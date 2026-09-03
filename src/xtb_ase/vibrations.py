"""ASE vibration data and molecular thermochemistry helpers.

The helpers in this module keep two calculation routes explicit:

* an analytic Hessian supplied by a calculator such as g-XTB or MACE;
* ASE finite differences of ``get_forces()`` for calculators such as GFN-FF.

The thermochemistry helper delegates the statistical-mechanics conventions to
ASE's :class:`~ase.thermochemistry.IdealGasThermo`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import Vibrations


def _load_vibrations_data():
    """Load ``VibrationsData`` across supported ASE module layouts."""

    try:
        from ase.vibrations.data import VibrationsData
    except ImportError:
        try:
            from ase.vibrationsdata import VibrationsData
        except ImportError as exc:  # pragma: no cover - old ASE fallback
            raise ImportError(
                "this ASE version does not provide VibrationsData; "
                "upgrade ASE to use the Hessian vibration adapter"
            ) from exc
    return VibrationsData


def _validate_atoms(atoms: Atoms, *, reject_periodic: bool = False) -> None:
    if not isinstance(atoms, Atoms):
        raise TypeError("atoms must be an ase.Atoms object")
    if len(atoms) == 0:
        raise ValueError("atoms must contain at least one atom")
    if reject_periodic and atoms.pbc.any():
        raise NotImplementedError(
            "the molecular vibration helper does not support periodic cells"
        )
    positions = np.asarray(atoms.positions, dtype=float)
    if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
        raise ValueError("atomic positions must be finite with shape (N, 3)")


def _normalize_indices(
    indices: Sequence[int] | np.ndarray | None,
    natoms: int,
) -> np.ndarray:
    if indices is None:
        return np.arange(natoms, dtype=int)
    try:
        values = np.asarray(indices)
    except (TypeError, ValueError) as exc:
        raise ValueError("indices must be a one-dimensional integer sequence") from exc
    if values.ndim != 1 or values.size == 0:
        raise ValueError("indices must be a non-empty one-dimensional sequence")
    if values.dtype.kind not in "iu":
        raise ValueError("indices must be a one-dimensional integer sequence")
    values = values.astype(int, copy=False)
    if np.any(values < 0) or np.any(values >= natoms):
        raise ValueError("indices must refer to atoms in the input structure")
    if np.unique(values).size != values.size:
        raise ValueError("indices must not contain duplicates")
    return values.copy()


def _normalize_hessian(atoms: Atoms, hessian: Any) -> np.ndarray:
    natoms = len(atoms)
    try:
        matrix = np.asarray(hessian, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("hessian must be a finite square numeric array") from exc
    if matrix.ndim == 4 and matrix.shape == (natoms, 3, natoms, 3):
        matrix = matrix.reshape(3 * natoms, 3 * natoms)
    expected_shape = (3 * natoms, 3 * natoms)
    if matrix.shape != expected_shape:
        raise ValueError(f"hessian must have shape {expected_shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("hessian must be finite")
    return np.ascontiguousarray(matrix)


def hessian_to_vibrations_data(
    atoms: Atoms,
    hessian: Sequence[Sequence[float]] | np.ndarray,
    indices: Sequence[int] | np.ndarray | None = None,
):
    """Convert an ASE-unit Cartesian Hessian to ``VibrationsData``.

    ``hessian`` may be a full ``(3N, 3N)`` matrix or an ASE-style
    ``(N, 3, N, 3)`` array.  When ``indices`` is supplied, the corresponding
    Cartesian submatrix is selected and the returned data records those active
    atoms.  The input ``Atoms`` object is copied and never receives a
    calculator through this function.
    """

    _validate_atoms(atoms)
    matrix = _normalize_hessian(atoms, hessian)
    active_indices = _normalize_indices(indices, len(atoms))
    flat_indices = np.repeat(3 * active_indices, 3) + np.tile(np.arange(3), active_indices.size)
    active_matrix = matrix[np.ix_(flat_indices, flat_indices)]

    vibrations_data_class = _load_vibrations_data()
    equilibrium = atoms.copy()
    equilibrium.calc = None
    from_2d = getattr(vibrations_data_class, "from_2d", None)
    if callable(from_2d):
        return from_2d(
            equilibrium,
            active_matrix,
            indices=active_indices.tolist(),
        )
    active_natoms = active_indices.size
    return vibrations_data_class(
        equilibrium,
        active_matrix.reshape(active_natoms, 3, active_natoms, 3),
        indices=active_indices.tolist(),
    )


def run_vibrations(
    atoms: Atoms,
    calculator: object | None = None,
    *,
    indices: Sequence[int] | np.ndarray | None = None,
    delta: float = 0.01,
    nfree: int = 2,
    name: str | os.PathLike[str] | None = None,
    method: str = "standard",
    direction: str = "central",
):
    """Run ASE finite-difference vibrations and return ``VibrationsData``.

    A private temporary cache is used when ``name`` is omitted.  The returned
    data is detached from that cache, while an explicit ``name`` leaves ASE's
    normal reusable vibration cache in place.
    """

    _validate_atoms(atoms, reject_periodic=True)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be positive")
    if isinstance(nfree, bool) or nfree not in {2, 4}:
        raise ValueError("nfree must be 2 or 4")
    active_calculator = calculator if calculator is not None else atoms.calc
    if active_calculator is None:
        raise ValueError("a calculator with get_forces is required")
    if not callable(getattr(active_calculator, "get_forces", None)):
        raise TypeError("calculator must provide get_forces(atoms)")
    active_indices = _normalize_indices(indices, len(atoms)) if indices is not None else None

    working = atoms.copy()
    working.calc = active_calculator

    def calculate(cache_name: str | os.PathLike[str]):
        vibration = Vibrations(
            working,
            indices=None if active_indices is None else active_indices.tolist(),
            name=os.fspath(cache_name),
            delta=float(delta),
            nfree=int(nfree),
        )
        vibration.run()
        return vibration.get_vibrations(method=method, direction=direction)

    if name is not None:
        return calculate(name)
    with tempfile.TemporaryDirectory(prefix="xtb-ase-vib-") as temporary_root:
        return calculate(Path(temporary_root) / "vib")


def get_vibrations_data(
    atoms: Atoms,
    calculator: object | None = None,
    *,
    indices: Sequence[int] | np.ndarray | None = None,
    delta: float = 0.01,
    nfree: int = 2,
    name: str | os.PathLike[str] | None = None,
    use_analytic: bool = True,
    method: str = "standard",
    direction: str = "central",
):
    """Obtain vibration data from an ASE calculator.

    If the selected calculator exposes ``get_hessian(atoms)``, that analytic
    Hessian is used by default, matching MACE and g-XTB.  Otherwise the helper
    falls back to :func:`run_vibrations`, which differentiates ASE forces as
    used for GFN-FF.  Set ``use_analytic=False`` to force finite differences.
    """

    _validate_atoms(atoms)
    active_calculator = calculator if calculator is not None else atoms.calc
    if active_calculator is None:
        raise ValueError("a calculator is required")
    hessian_getter = getattr(active_calculator, "get_hessian", None)
    if use_analytic and callable(hessian_getter):
        try:
            return hessian_to_vibrations_data(
                atoms,
                hessian_getter(atoms),
                indices=indices,
            )
        except NotImplementedError:
            pass
    return run_vibrations(
        atoms,
        calculator=active_calculator,
        indices=indices,
        delta=delta,
        nfree=nfree,
        name=name,
        method=method,
        direction=direction,
    )


@dataclass(frozen=True)
class ASEVibrationalThermochemistry:
    """ASE `IdealGasThermo` results in explicit SI/ASE units."""

    potential_energy_eV: float
    zero_point_energy_eV: float
    enthalpy_eV: float
    entropy_eV_per_K: float
    gibbs_free_energy_eV: float
    free_energy_eV: float
    thermochemical_correction_eV: float
    temperature_K: float
    pressure_Pa: float

    def __post_init__(self) -> None:
        names = (
            "potential_energy_eV",
            "zero_point_energy_eV",
            "enthalpy_eV",
            "entropy_eV_per_K",
            "gibbs_free_energy_eV",
            "free_energy_eV",
            "thermochemical_correction_eV",
            "temperature_K",
            "pressure_Pa",
        )
        for name in names:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be positive")
        if self.pressure_Pa <= 0.0:
            raise ValueError("pressure_Pa must be positive")


def _vibration_energies(vibrations: object, natoms: int) -> np.ndarray:
    getter = getattr(vibrations, "get_energies", None)
    if not callable(getter):
        raise TypeError("vibrations must provide get_energies()")
    try:
        values = np.asarray(getter(), dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError("vibrational energies must be a one-dimensional array") from exc
    if values.ndim != 1:
        raise ValueError("vibrational energies must be a one-dimensional array")
    if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
        raise ValueError("vibrational energies must be finite")
    atoms_getter = getattr(vibrations, "get_atoms", None)
    if callable(atoms_getter):
        vibration_atoms = atoms_getter()
        if len(vibration_atoms) != natoms:
            raise ValueError("vibrations and atoms must contain the same number of atoms")
    return values


def ase_vibrational_thermochemistry(
    atoms: Atoms,
    vibrations: object,
    *,
    temperature_K: float = 298.15,
    pressure: float = units.bar,
    geometry: str,
    symmetrynumber: int,
    spin: float,
    potential_energy: float | None = None,
    vib_selection: str = "highest",
    ignore_imag_modes: bool = False,
) -> ASEVibrationalThermochemistry:
    """Evaluate ASE ideal-gas RRHO thermochemistry from vibration data.

    ``pressure`` follows ASE's thermochemistry convention, so use
    ``ase.units.bar`` (an internal ASE pressure value).  The returned
    ``pressure_Pa`` is converted to SI for unambiguous reporting.  The
    thermochemical correction is defined as ``G - potential_energy``.
    """

    _validate_atoms(atoms, reject_periodic=True)
    if not np.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if not np.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("pressure must be positive")
    if geometry not in {"linear", "nonlinear", "monatomic"}:
        raise ValueError("geometry must be linear, nonlinear, or monatomic")
    if isinstance(symmetrynumber, bool) or not isinstance(symmetrynumber, (int, np.integer)):
        raise ValueError("symmetrynumber must be a positive integer")
    if symmetrynumber <= 0:
        raise ValueError("symmetrynumber must be a positive integer")
    if not np.isfinite(spin) or spin < 0.0:
        raise ValueError("spin must be finite and non-negative")
    if vib_selection not in {"all", "exact", "highest", "abs_highest"}:
        raise ValueError(
            "vib_selection must be all, exact, highest, or abs_highest"
        )
    if not isinstance(ignore_imag_modes, (bool, np.bool_)):
        raise ValueError("ignore_imag_modes must be boolean")

    energies = _vibration_energies(vibrations, len(atoms))
    if potential_energy is None:
        try:
            potential_energy = float(atoms.get_potential_energy())
        except Exception as exc:
            raise ValueError(
                "potential energy is required when atoms have no usable calculator"
            ) from exc
    potential_energy = float(potential_energy)
    if not np.isfinite(potential_energy):
        raise ValueError("potential energy must be finite")

    thermo = IdealGasThermo(
        vib_energies=energies,
        geometry=geometry,
        potentialenergy=potential_energy,
        atoms=atoms,
        symmetrynumber=int(symmetrynumber),
        spin=float(spin),
        vib_selection=vib_selection,
        ignore_imag_modes=bool(ignore_imag_modes),
    )
    zero_point_energy = float(thermo.get_ZPE_correction())
    enthalpy = float(thermo.get_enthalpy(float(temperature_K), verbose=False))
    entropy = float(
        thermo.get_entropy(float(temperature_K), pressure=float(pressure), verbose=False)
    )
    gibbs_free_energy = float(
        thermo.get_gibbs_energy(
            float(temperature_K),
            float(pressure),
            verbose=False,
        )
    )
    pressure_pa = float(pressure / units.Pascal)
    return ASEVibrationalThermochemistry(
        potential_energy_eV=potential_energy,
        zero_point_energy_eV=zero_point_energy,
        enthalpy_eV=enthalpy,
        entropy_eV_per_K=entropy,
        gibbs_free_energy_eV=gibbs_free_energy,
        free_energy_eV=gibbs_free_energy,
        thermochemical_correction_eV=gibbs_free_energy - potential_energy,
        temperature_K=float(temperature_K),
        pressure_Pa=pressure_pa,
    )
