"""Lazy ASE façade for the standalone GFN-FF Python package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ._runtime import (
    EnvironmentValue,
    normalize_environment,
    normalize_optional_int,
    temporary_environment,
)


class GFNFFDependencyError(ImportError):
    """Raised when the optional standalone GFN-FF backend is unavailable."""


class GFNFFConfigurationError(RuntimeError):
    """Raised when native GFN-FF runtime configuration is unavailable."""


def _load_standalone_gfnff():
    """Load the real ASE backend without making it a hard import dependency."""

    try:
        from gfnff.ase_calculator import GFNFF as standalone_class
    except ImportError as exc:
        raise GFNFFDependencyError(
            "The GFN-FF backend is optional; install it with "
            "`pip install 'xtb-ase[gfnff]'` (or `pip install 'gfnff[ase]'`)."
        ) from exc
    return standalone_class


def _set_native_threads(threads: int | None) -> None:
    """Set the OpenMP thread count in the loaded standalone backend.

    The standalone package does not expose this setting in its Python class,
    but its ctypes loader keeps the native library handle available.  The
    OpenMP setting is process-wide, which is why independent configurations
    should use :class:`xtb_ase.CalculatorPool` workers.
    """

    if threads is None:
        return
    try:
        from gfnff import _lib as low_level

        library = low_level._lib
        setter = getattr(library, "omp_set_num_threads", None)
        if setter is None:
            setter = getattr(library, "omp_set_num_threads_", None)
        if setter is None:
            raise AttributeError("omp_set_num_threads")
        setter.argtypes = [ctypes.c_int]
        setter.restype = None
        setter(ctypes.c_int(threads))
    except (AttributeError, ImportError, OSError) as exc:
        raise GFNFFConfigurationError(
            "the loaded GFN-FF backend does not expose an OpenMP thread setter"
        ) from exc


class GFNFF(Calculator):
    """ASE calculator façade over ``gfnff.ase_calculator.GFNFF``.

    The standalone backend performs the unit conversion and owns the native
    GFN-FF state.  This class keeps the dependency optional and provides a
    stable import from :mod:`xtb_ase`.
    """

    implemented_properties = ["energy", "forces", "stress"]
    default_parameters = {
        "charge": 0,
        "solvent": "",
        "printlevel": 0,
    }

    def __init__(
        self,
        charge: int = 0,
        solvent: str = "",
        printlevel: int = 0,
        fragments: Sequence[int] | None = None,
        ref_charges: Sequence[float] | None = None,
        threads: int | None = None,
        env: Mapping[str, EnvironmentValue] | None = None,
        **kwargs: Any,
    ) -> None:
        if "processes" in kwargs:
            raise TypeError(
                "processes belongs to CalculatorPool, not an individual Calculator"
            )
        electronic_parameters = {
            name
            for name in (
                "uhf",
                "spin",
                "unpaired_electrons",
                "etemp",
                "electronic_temperature",
            )
            if name in kwargs
        }
        if electronic_parameters:
            raise TypeError(
                "GFN-FF does not support electronic parameters: "
                + ", ".join(sorted(electronic_parameters))
            )
        threads = normalize_optional_int(threads, "threads", minimum=1)
        env = normalize_environment(env)
        self._backend = None
        self._last_numbers: np.ndarray | None = None
        self._last_pbc: np.ndarray | None = None
        self._last_charge: int | None = None
        self._fragments = fragments
        self._ref_charges = ref_charges
        super().__init__(**kwargs)
        self.parameters.update(
            charge=int(charge),
            solvent=str(solvent),
            printlevel=int(printlevel),
            threads=threads,
            env=env,
        )

    def set(self, **kwargs: Any):
        if "processes" in kwargs:
            raise TypeError(
                "processes belongs to CalculatorPool, not an individual Calculator"
            )
        electronic_parameters = {
            name
            for name in (
                "uhf",
                "spin",
                "unpaired_electrons",
                "etemp",
                "electronic_temperature",
            )
            if name in kwargs
        }
        if electronic_parameters:
            raise TypeError(
                "GFN-FF does not support electronic parameters: "
                + ", ".join(sorted(electronic_parameters))
            )
        if "threads" in kwargs:
            kwargs["threads"] = normalize_optional_int(
                kwargs["threads"], "threads", minimum=1
            )
        if "env" in kwargs:
            kwargs["env"] = normalize_environment(kwargs["env"])
        changed = super().set(**kwargs)
        if changed and self._backend is not None:
            self._dispose_backend()
            self.results = {}
        return changed

    def check_state(self, atoms, tol=1e-15):
        """Also invalidate cached results when ``atoms.info['charge']`` changes."""

        changes = super().check_state(atoms, tol=tol)
        if self._backend is not None:
            charge = int(atoms.info.get("charge", self.parameters.charge))
            if charge != self._last_charge and "charge" not in changes:
                changes.append("charge")
        return changes

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ) -> None:
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("GFNFF.calculate requires an Atoms object")
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms
        if self._needs_reinit(atoms):
            self._dispose_backend()
            self._backend = self._make_backend(atoms)
            self._last_numbers = np.asarray(atoms.numbers, dtype=np.int32).copy()
            self._last_pbc = np.asarray(atoms.pbc, dtype=bool).copy()
            self._last_charge = int(atoms.info.get("charge", self.parameters.charge))

        _set_native_threads(self.parameters.threads)
        self._backend.calculate(atoms, properties, system_changes)
        for name in self.implemented_properties:
            if name in self._backend.results:
                value = self._backend.results[name]
                self.results[name] = value.copy() if hasattr(value, "copy") else value

    def _needs_reinit(self, atoms) -> bool:
        if self._backend is None:
            return True
        numbers = np.asarray(atoms.numbers, dtype=np.int32)
        pbc = np.asarray(atoms.pbc, dtype=bool)
        charge = int(atoms.info.get("charge", self.parameters.charge))
        return (
            not np.array_equal(numbers, self._last_numbers)
            or not np.array_equal(pbc, self._last_pbc)
            or charge != self._last_charge
        )

    def _make_backend(self, atoms):
        with temporary_environment(self.parameters.env):
            standalone_class = _load_standalone_gfnff()
            _set_native_threads(self.parameters.threads)
            fragments = atoms.info.get("fragments", self._fragments)
            ref_charges = atoms.info.get("ref_charges", self._ref_charges)
            return standalone_class(
                charge=int(atoms.info.get("charge", self.parameters.charge)),
                solvent=self.parameters.solvent,
                printlevel=self.parameters.printlevel,
                fragments=fragments,
                ref_charges=ref_charges,
            )

    def _dispose_backend(self) -> None:
        backend = self._backend
        self._backend = None
        self._last_numbers = None
        self._last_pbc = None
        self._last_charge = None
        if backend is None:
            return
        native = getattr(backend, "_gfnff", None)
        if native is not None:
            native.deallocate()

    def __del__(self):
        try:
            self._dispose_backend()
        except Exception:
            pass
