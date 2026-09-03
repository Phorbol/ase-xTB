"""Reproducible ASE molecular-dynamics helpers for search trajectories."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.constraints import FixCom
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary

try:
    from ase.md.velocitydistribution import thermalize_momenta
except ImportError:  # pragma: no cover - compatibility with older ASE
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    def thermalize_momenta(atoms, temperature_K, *, rng):
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)


@dataclass(frozen=True)
class LangevinConfig:
    """Control a non-periodic Langevin trajectory in molecular units."""

    temperature_K: float = 600.0
    timestep_fs: float = 0.5
    friction_per_fs: float = 0.01
    steps: int = 1000
    sample_interval: int = 10
    rng_seed: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature_K) or self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be positive")
        if not np.isfinite(self.timestep_fs) or self.timestep_fs <= 0.0:
            raise ValueError("timestep_fs must be positive")
        if not np.isfinite(self.friction_per_fs) or self.friction_per_fs <= 0.0:
            raise ValueError("friction_per_fs must be positive")
        for name, value in (
            ("steps", self.steps),
            ("sample_interval", self.sample_interval),
            ("rng_seed", self.rng_seed),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.sample_interval <= 0:
            raise ValueError("sample_interval must be positive")
        if self.rng_seed < 0:
            raise ValueError("rng_seed must be non-negative")


def iter_langevin_frames(
    atoms: Atoms,
    calculator: Calculator,
    config: LangevinConfig,
) -> Iterator[Atoms]:
    """Yield detached frames from a reproducible non-periodic Langevin run."""

    if not isinstance(atoms, Atoms):
        raise TypeError("atoms must be an ase.Atoms object")
    if len(atoms) == 0:
        raise ValueError("atoms must contain at least one atom")
    if atoms.pbc.any():
        raise NotImplementedError(
            "the baseline Langevin sampler does not support periodic cells"
        )
    if calculator is None:
        raise TypeError("calculator must be an ASE calculator")

    working = atoms.copy()
    working.calc = calculator
    rng = np.random.default_rng(int(config.rng_seed))
    thermalize_momenta(
        working,
        temperature_K=float(config.temperature_K),
        rng=rng,
    )
    Stationary(working)
    working.set_constraint([*working.constraints, FixCom()])
    dynamics = Langevin(
        working,
        timestep=float(config.timestep_fs) * units.fs,
        temperature_K=float(config.temperature_K),
        friction=float(config.friction_per_fs) / units.fs,
        fixcm=False,
        rng=rng,
    )

    for step in range(int(config.steps) + 1):
        if step % int(config.sample_interval) == 0:
            frame = working.copy()
            frame.calc = None
            frame.info["md_step"] = step
            frame.info["temperature_K"] = float(config.temperature_K)
            yield frame
        if step < int(config.steps):
            dynamics.run(1)


def sample_langevin_frames(
    atoms: Atoms,
    calculator: Calculator,
    config: LangevinConfig,
) -> list[Atoms]:
    """Materialize :func:`iter_langevin_frames` as a list."""

    return list(iter_langevin_frames(atoms, calculator, config))
