"""ASE calculator backed by the g-xTB v2 command-line executable."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
from ase import units
from ase.calculators.calculator import (
    Calculator,
    CalculatorError,
    PropertyNotImplementedError,
    all_changes,
)

from ._parsers import (
    OutputParseError,
    parse_charges,
    parse_engrad,
    parse_hessian,
    parse_molden,
    parse_stdout_properties,
    parse_vibspectrum,
    parse_wbo,
)


class GXTBExecutionError(RuntimeError):
    """Raised when the g-xTB process does not produce a valid calculation."""


class XTB(Calculator):
    """ASE calculator for standard xTB Hamiltonians and g-xTB v2.

    The calculator uses a fresh scratch directory for every invocation.  The
    base ASE properties are returned in ASE units; optional electronic and
    vibrational properties are requested lazily by the explicit methods below.
    """

    implemented_properties = [
        "energy",
        "forces",
        "charges",
        "dipole",
        "quadrupole",
        "bond_orders",
        "orbital_energies",
        "orbital_occupations",
        "homo_lumo_gap",
        "hessian",
        "vibrational_frequencies",
    ]

    _optional_properties = {
        "energy",
        "forces",
        "charges",
        "dipole",
        "quadrupole",
        "bond_orders",
        "orbital_energies",
        "orbital_occupations",
        "homo_lumo_gap",
        "hessian",
        "vibrational_frequencies",
    }

    _method_aliases = {
        "gxtb": "gxtb",
        "g-xtb": "gxtb",
        "gfn0": "gfn0-xtb",
        "gfn0-xtb": "gfn0-xtb",
        "gfn0xtb": "gfn0-xtb",
        "gfn1": "gfn1-xtb",
        "gfn1-xtb": "gfn1-xtb",
        "gfn1xtb": "gfn1-xtb",
        "gfn2": "gfn2-xtb",
        "gfn2-xtb": "gfn2-xtb",
        "gfn2xtb": "gfn2-xtb",
    }

    def __init__(
        self,
        command: str | os.PathLike[str] | Sequence[str] = "xtb",
        *,
        method: str = "gfn2-xtb",
        charge: int = 0,
        uhf: int | None = None,
        accuracy: float = 1.0,
        etemp: float | None = None,
        solvation_model: str | None = None,
        solvent: str | None = None,
        directory: str | os.PathLike[str] | None = None,
        keep_files: bool = False,
        timeout: float | None = None,
        parallel: int = 1,
        properties: Sequence[str] | None = None,
        extra_args: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        command_parts = self._normalize_command(command)
        method = self._normalize_method(method)
        requested_properties = tuple(properties or ())
        invalid = set(requested_properties) - self._optional_properties
        if invalid:
            raise ValueError(f"unsupported GXTB properties: {sorted(invalid)}")
        if solvation_model is not None:
            solvation_model = solvation_model.lower()
            if solvation_model not in {"gbe", "cosmo"}:
                raise ValueError("solvation_model must be 'gbe', 'cosmo', or None")
            if solvent is None:
                raise ValueError("solvent is required when solvation_model is set")
        if parallel < 1:
            raise ValueError("parallel must be positive")
        if uhf is not None and uhf < 0:
            raise ValueError("uhf must be non-negative or None")
        if etemp is not None and etemp < 0:
            raise ValueError("etemp must be non-negative or None")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive or None")

        self._scratch_root = Path(directory) if directory is not None else None
        self._last_run_directory: Path | None = None
        self._last_output: dict[str, Any] | None = None
        self._artifact_paths: dict[str, Path] = {}

        super().__init__(
            command=command_parts,
            method=method,
            charge=int(charge),
            uhf=None if uhf is None else int(uhf),
            accuracy=float(accuracy),
            etemp=None if etemp is None else float(etemp),
            solvation_model=solvation_model,
            solvent=None if solvent is None else str(solvent),
            keep_files=bool(keep_files),
            timeout=timeout,
            parallel=int(parallel),
            properties=requested_properties,
            extra_args=tuple(str(arg) for arg in extra_args),
            **kwargs,
        )

    def set(self, **kwargs: Any):
        changed = super().set(**kwargs)
        if changed and hasattr(self, "_artifact_paths"):
            self.results = {}
            self._artifact_paths.clear()
            self._last_run_directory = None
            self._last_output = None
        return changed

    @staticmethod
    def _normalize_method(method: str) -> str:
        try:
            return XTB._method_aliases[str(method).lower()]
        except KeyError as exc:
            allowed = ", ".join(sorted(set(XTB._method_aliases.values())))
            raise ValueError(
                f"unsupported xTB method {method!r}; use {allowed}"
            ) from exc

    @staticmethod
    def _normalize_command(
        command: str | os.PathLike[str] | Sequence[str],
    ) -> tuple[str, ...]:
        if isinstance(command, (str, os.PathLike)):
            result = (os.fspath(command),)
        else:
            result = tuple(os.fspath(part) for part in command)
        if not result or not result[0]:
            raise ValueError("command must not be empty")
        return result

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ) -> None:
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise CalculatorError("GXTB.calculate requires an Atoms object")

        requested = set(properties) | set(self.parameters.properties)
        invalid = requested - self._optional_properties
        if invalid:
            raise PropertyNotImplementedError(
                f"unsupported GXTB properties: {sorted(invalid)}"
            )
        hessian_requested = bool(
            requested & {"hessian", "vibrational_frequencies"}
        )
        if hessian_requested and "forces" in requested:
            raise ValueError(
                "g-xTB numerical Hessian and forces require separate calculations"
            )

        if system_changes:
            self.results = {}
            self._artifact_paths.clear()
        super().calculate(atoms, properties, system_changes)
        self._validate_atoms(atoms)

        run_directory = self._make_run_directory()
        self._last_run_directory = run_directory if self.parameters.keep_files else None
        try:
            input_name = "structure.xyz"
            self._write_xyz(run_directory / input_name, atoms)
            operation = "hessian" if hessian_requested else "gradient"
            command = self._build_command(input_name, requested, operation)
            try:
                completed = subprocess.run(
                    command,
                    cwd=run_directory,
                    capture_output=True,
                    text=True,
                    timeout=self.parameters.timeout,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise GXTBExecutionError(
                    f"g-xTB executable not found: {command[0]!r}"
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise GXTBExecutionError(
                    f"g-xTB timed out after {self.parameters.timeout} seconds"
                ) from exc

            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            (run_directory / "stdout").write_text(stdout)
            (run_directory / "stderr").write_text(stderr)
            self._last_output = {
                "command": tuple(command),
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
            self._check_process(completed.returncode, stdout, stderr, command)

            if operation == "gradient":
                parsed_gradient = parse_engrad(
                    run_directory / "structure.engrad", len(atoms)
                )
                self.results["energy"] = parsed_gradient.energy_hartree * units.Hartree
                self.results["forces"] = (
                    -parsed_gradient.gradient_hartree_per_bohr
                    * units.Hartree
                    / units.Bohr
                )
            else:
                stdout_properties = parse_stdout_properties(stdout, len(atoms))
                if stdout_properties.total_energy_hartree is None:
                    raise OutputParseError("g-xTB output contains no total energy")
                self.results["energy"] = (
                    stdout_properties.total_energy_hartree * units.Hartree
                )

            self._parse_requested_properties(
                run_directory, stdout, len(atoms), requested
            )
            if self.parameters.keep_files:
                for name, filename in {
                    "molden": "molden.input",
                    "hessian": "hessian",
                    "vibspectrum": "vibspectrum",
                    "charges": "charges",
                    "wbo": "wbo",
                    "engrad": "structure.engrad",
                }.items():
                    artifact = run_directory / filename
                    if artifact.is_file():
                        self._artifact_paths[name] = artifact
        except FileNotFoundError as exc:
            missing = exc.filename or "unknown output"
            if not self.parameters.keep_files:
                shutil.rmtree(run_directory, ignore_errors=True)
            raise GXTBExecutionError(
                f"required xTB output file missing: {missing}"
            ) from exc
        except Exception:
            if not self.parameters.keep_files:
                shutil.rmtree(run_directory, ignore_errors=True)
            raise
        else:
            if not self.parameters.keep_files:
                shutil.rmtree(run_directory, ignore_errors=True)

    def _validate_atoms(self, atoms) -> None:
        if len(atoms) == 0:
            raise ValueError("g-xTB requires at least one atom")
        if atoms.pbc.any():
            raise NotImplementedError(
                "g-xTB ASE wrapper does not claim periodic-cell support yet"
            )
        positions = np.asarray(atoms.positions, dtype=float)
        if not np.isfinite(positions).all():
            raise ValueError("atomic positions must be finite")

    def _make_run_directory(self) -> Path:
        root = self._scratch_root or Path(tempfile.gettempdir())
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="xtb-ase-", dir=root))

    @staticmethod
    def _write_xyz(path: Path, atoms) -> None:
        lines = [str(len(atoms)), "generated by xtb-ase"]
        for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions):
            x, y, z = (float(value) for value in position)
            lines.append(f"{symbol:2s} {x:.16g} {y:.16g} {z:.16g}")
        path.write_text("\n".join(lines) + "\n")

    def _build_command(
        self,
        input_name: str,
        requested: set[str],
        operation: str,
    ) -> list[str]:
        parameters = self.parameters
        command = list(parameters.command) + [input_name, "--no-restart"]
        if parameters.method == "gxtb":
            command.append("--gxtb")
        else:
            command += ["--gfn", parameters.method[3]]
        command += ["--chrg", str(parameters.charge)]
        if parameters.uhf is not None:
            command += ["--uhf", str(parameters.uhf)]
        command += ["--acc", f"{parameters.accuracy:g}"]
        if parameters.etemp is not None:
            command += ["--etemp", f"{parameters.etemp:g}"]
        command += ["--parallel", str(parameters.parallel)]
        command.append("--hess" if operation == "hessian" else "--grad")

        if "charges" in requested:
            command.append("--pop")
        if "dipole" in requested:
            command.append("--dipole")
        if "quadrupole" in requested:
            command.append("--quadrupole")
        if "bond_orders" in requested:
            command.append("--wbo")
        if requested & {"orbital_energies", "orbital_occupations"}:
            command.append("--molden")
        if parameters.solvation_model is not None:
            command += [f"--{parameters.solvation_model}", parameters.solvent]
        command.extend(parameters.extra_args)
        return command

    @staticmethod
    def _check_process(
        returncode: int,
        stdout: str,
        stderr: str,
        command: Sequence[str],
    ) -> None:
        combined = f"{stdout}\n{stderr}"
        nonconverged = re.search(
            r"(?:not\s+converged|did\s+not\s+converge|scf\s+failed)",
            combined,
            flags=re.IGNORECASE,
        )
        if returncode != 0 or nonconverged:
            detail = stderr.strip() or stdout.strip()
            raise GXTBExecutionError(
                f"g-xTB failed (returncode={returncode}) for {list(command)!r}: "
                f"{detail[-2000:]}"
            )

    def _parse_requested_properties(
        self,
        run_directory: Path,
        stdout: str,
        natoms: int,
        requested: set[str],
    ) -> None:
        stdout_properties = None
        if requested & {
            "charges",
            "dipole",
            "quadrupole",
            "homo_lumo_gap",
            "hessian",
            "vibrational_frequencies",
        }:
            stdout_properties = parse_stdout_properties(stdout, natoms)

        if "charges" in requested:
            self.results["charges"] = parse_charges(run_directory / "charges", natoms)
        if "dipole" in requested:
            assert stdout_properties is not None
            if stdout_properties.dipole_au is not None:
                self.results["dipole"] = stdout_properties.dipole_au * units.Bohr
            elif stdout_properties.dipole_debye_vector is not None:
                self.results["dipole"] = (
                    stdout_properties.dipole_debye_vector * units.Debye
                )
            else:
                raise OutputParseError("g-xTB output contains no dipole vector")
        if "quadrupole" in requested:
            assert stdout_properties is not None
            if stdout_properties.quadrupole_au is None:
                raise OutputParseError("xTB output contains no quadrupole tensor")
            xx, xy, yy, xz, yz, zz = stdout_properties.quadrupole_au
            self.results["quadrupole"] = np.asarray(
                [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float
            ) * units.Bohr**2
        if "homo_lumo_gap" in requested:
            assert stdout_properties is not None
            if stdout_properties.homo_lumo_gap_ev is None:
                raise OutputParseError("g-xTB output contains no HOMO-LUMO gap")
            self.results["homo_lumo_gap"] = stdout_properties.homo_lumo_gap_ev
        if "bond_orders" in requested:
            self.results["bond_orders"] = parse_wbo(run_directory / "wbo", natoms)
        if requested & {"orbital_energies", "orbital_occupations"}:
            orbitals = parse_molden(run_directory / "molden.input")
            self.results["orbital_energies"] = orbitals.energies_hartree * units.Hartree
            self.results["orbital_occupations"] = orbitals.occupations
        if "hessian" in requested:
            self.results["hessian"] = (
                parse_hessian(run_directory / "hessian", natoms)
                * units.Hartree
                / units.Bohr**2
            )
        if "vibrational_frequencies" in requested:
            self.results["vibrational_frequencies"] = parse_vibspectrum(
                run_directory / "vibspectrum"
            )

    def _ensure_property(self, name: str, atoms=None):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(f"{name} property not implemented")
        return self.get_property(name, atoms)

    def get_bond_orders(self, atoms=None) -> np.ndarray:
        return self._ensure_property("bond_orders", atoms)

    def get_wbo(self, atoms=None) -> np.ndarray:
        """Alias for :meth:`get_bond_orders`."""

        return self.get_bond_orders(atoms)

    def get_orbital_energies(self, atoms=None, unit: str = "eV") -> np.ndarray:
        energies_ev = np.asarray(
            self._ensure_property("orbital_energies", atoms), dtype=float
        )
        normalized = unit.lower().replace(" ", "")
        if normalized in {"ev", "electronvolt", "electronvolts"}:
            return energies_ev.copy()
        if normalized in {"eh", "ha", "hartree", "hartrees"}:
            return energies_ev / units.Hartree
        raise ValueError("unit must be 'eV' or 'hartree'")

    def get_orbital_occupations(self, atoms=None) -> np.ndarray:
        return np.asarray(
            self._ensure_property("orbital_occupations", atoms), dtype=float
        ).copy()

    def get_homo_lumo_gap(self, atoms=None, unit: str = "eV") -> float:
        gap_ev = float(self._ensure_property("homo_lumo_gap", atoms))
        normalized = unit.lower().replace(" ", "")
        if normalized in {"ev", "electronvolt", "electronvolts"}:
            return gap_ev
        if normalized in {"eh", "ha", "hartree", "hartrees"}:
            return gap_ev / units.Hartree
        raise ValueError("unit must be 'eV' or 'hartree'")

    def get_quadrupole(self, atoms=None, unit: str = "eA2") -> np.ndarray:
        quadrupole_ea2 = np.asarray(
            self._ensure_property("quadrupole", atoms), dtype=float
        )
        normalized = unit.lower().replace(" ", "")
        if normalized in {"ea2", "eangstrom2", "eangstroms2"}:
            return quadrupole_ea2.copy()
        if normalized in {"au", "ea02", "ebohr2", "ebohrs2"}:
            return quadrupole_ea2 / units.Bohr**2
        raise ValueError("unit must be 'eA2' or 'au'")

    def get_hessian(self, atoms=None) -> np.ndarray:
        return np.asarray(self._ensure_property("hessian", atoms), dtype=float).copy()

    def get_vibrational_frequencies(self, atoms=None) -> np.ndarray:
        return np.asarray(
            self._ensure_property("vibrational_frequencies", atoms), dtype=float
        ).copy()

    def get_molden_path(self, atoms=None) -> Path:
        self._ensure_property("orbital_energies", atoms)
        path = self._artifact_paths.get("molden")
        if path is None:
            raise PropertyNotImplementedError(
                "Molden path is unavailable when keep_files=False"
            )
        if not path.is_file():
            raise PropertyNotImplementedError("g-xTB did not emit molden.input")
        return path

    def get_run_directory(self) -> Path | None:
        return self._last_run_directory

    def get_raw_output(self) -> dict[str, Any]:
        if self._last_output is None:
            raise CalculatorError("GXTB has not run yet")
        return dict(self._last_output)


class GXTB(XTB):
    """Convenience calculator fixed to ``method='gxtb'``."""

    def __init__(
        self,
        command: str | os.PathLike[str] | Sequence[str] = "xtb",
        **kwargs: Any,
    ) -> None:
        if "method" in kwargs:
            raise TypeError("GXTB always uses method='gxtb'")
        kwargs.setdefault("etemp", 0.0)
        super().__init__(command=command, method="gxtb", **kwargs)
