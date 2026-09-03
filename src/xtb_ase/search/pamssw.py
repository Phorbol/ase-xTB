"""Optional adapter for PAM-SSW global-search comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms


@dataclass(frozen=True)
class PAMSSWComparisonConfig:
    """Explicit PAM-SSW comparison settings in the native eV convention."""

    target_uphill_energy_eV: float = 0.05
    max_trials: int = 32
    max_steps_per_walk: int = 8
    rng_seed: int = 0
    max_force_evals: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.target_uphill_energy_eV) or self.target_uphill_energy_eV <= 0.0:
            raise ValueError("target_uphill_energy_eV must be positive")
        for name, value in (
            ("max_trials", self.max_trials),
            ("max_steps_per_walk", self.max_steps_per_walk),
            ("rng_seed", self.rng_seed),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.max_trials <= 0:
            raise ValueError("max_trials must be positive")
        if self.max_steps_per_walk <= 0:
            raise ValueError("max_steps_per_walk must be positive")
        if self.rng_seed < 0:
            raise ValueError("rng_seed must be non-negative")
        if self.max_force_evals is not None:
            if (
                isinstance(self.max_force_evals, bool)
                or not isinstance(self.max_force_evals, (int, np.integer))
                or self.max_force_evals <= 0
            ):
                raise ValueError("max_force_evals must be a positive integer or None")


@dataclass(frozen=True)
class PAMSSWComparisonResult:
    """Detached ASE structures and summary values returned by PAM-SSW."""

    best_atoms: Atoms
    best_energy_eV: float
    minima: tuple[Atoms, ...]
    archive_energies_eV: np.ndarray
    raw_result: Any

    def __post_init__(self) -> None:
        if not isinstance(self.best_atoms, Atoms):
            raise TypeError("best_atoms must be an ase.Atoms object")
        if not np.isfinite(self.best_energy_eV):
            raise ValueError("best_energy_eV must be finite")
        best = self.best_atoms.copy()
        best.calc = None
        object.__setattr__(self, "best_atoms", best)
        minima = []
        for atoms in self.minima:
            if not isinstance(atoms, Atoms):
                raise TypeError("minima must contain ase.Atoms objects")
            copy = atoms.copy()
            copy.calc = None
            minima.append(copy)
        object.__setattr__(self, "minima", tuple(minima))
        energies = np.asarray(self.archive_energies_eV, dtype=float)
        if energies.ndim != 1 or not np.isfinite(energies).all():
            raise ValueError("archive_energies_eV must be a finite 1D array")
        object.__setattr__(self, "archive_energies_eV", energies.copy())
        object.__setattr__(self, "best_energy_eV", float(self.best_energy_eV))


def _load_pamssw():
    try:
        from pamssw import SSWConfig, State, run_ssw
        from pamssw.calculators import ASECalculator
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "PAM-SSW comparison requires the pamssw package; expose the "
            "`/home/gengjianrui/bin/pam-ssw` checkout or install it first"
        ) from exc
    return State, SSWConfig, run_ssw, ASECalculator


def _state_to_atoms(state) -> Atoms:
    cell = None if getattr(state, "cell", None) is None else np.asarray(state.cell, dtype=float).copy()
    atoms = Atoms(
        numbers=np.asarray(state.numbers, dtype=int).copy(),
        positions=np.asarray(state.positions, dtype=float).copy(),
        cell=cell,
        pbc=tuple(bool(value) for value in getattr(state, "pbc", (False, False, False))),
    )
    atoms.info.update(dict(getattr(state, "metadata", {})))
    atoms.calc = None
    return atoms


def run_pamssw(
    initial_atoms: Atoms,
    calculator: object,
    config: PAMSSWComparisonConfig | None = None,
) -> PAMSSWComparisonResult:
    """Run PAM-SSW from an ASE structure using a supplied ASE calculator."""

    if not isinstance(initial_atoms, Atoms):
        raise TypeError("initial_atoms must be an ase.Atoms object")
    if len(initial_atoms) == 0:
        raise ValueError("initial_atoms must contain at least one atom")
    if calculator is None:
        raise TypeError("calculator must be an ASE calculator")
    config = PAMSSWComparisonConfig() if config is None else config
    State, SSWConfig, run_ssw_function, ASECalculator = _load_pamssw()

    cell = None
    if initial_atoms.cell.any():
        cell = np.asarray(initial_atoms.cell, dtype=float).copy()
    state = State(
        numbers=np.asarray(initial_atoms.numbers, dtype=int).copy(),
        positions=np.asarray(initial_atoms.positions, dtype=float).copy(),
        cell=cell,
        pbc=tuple(bool(value) for value in initial_atoms.pbc),
        metadata=dict(initial_atoms.info),
    )
    search_config = SSWConfig(
        target_uphill_energy=config.target_uphill_energy_eV,
        max_trials=config.max_trials,
        max_steps_per_walk=config.max_steps_per_walk,
        rng_seed=config.rng_seed,
        max_force_evals=config.max_force_evals,
    )
    raw_result = run_ssw_function(
        state,
        ASECalculator(calculator),
        search_config,
    )
    entries = list(raw_result.archive.entries)
    minima = tuple(_state_to_atoms(entry.state) for entry in entries)
    archive_energies = np.asarray([float(entry.energy) for entry in entries], dtype=float)
    return PAMSSWComparisonResult(
        best_atoms=_state_to_atoms(raw_result.best_state),
        best_energy_eV=float(raw_result.best_energy),
        minima=minima,
        archive_energies_eV=archive_energies,
        raw_result=raw_result,
    )
