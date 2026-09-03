import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones

from xtb_ase.search.sampling import (
    LangevinConfig,
    iter_langevin_frames,
    sample_langevin_frames,
)


def lj_dimer() -> Atoms:
    return Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])


def test_langevin_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="temperature"):
        LangevinConfig(temperature_K=0.0)
    with pytest.raises(ValueError, match="sample_interval"):
        LangevinConfig(sample_interval=0)


def test_langevin_sampling_has_deterministic_count_and_detached_frames():
    config = LangevinConfig(
        temperature_K=50.0,
        timestep_fs=0.2,
        friction_per_fs=0.05,
        steps=4,
        sample_interval=2,
        rng_seed=7,
    )
    frames = sample_langevin_frames(lj_dimer(), LennardJones(), config)
    assert len(frames) == 3
    assert all(frame.calc is None for frame in frames)
    assert all(np.isfinite(frame.positions).all() for frame in frames)


def test_langevin_iterator_rejects_periodic_cells():
    atoms = lj_dimer()
    atoms.cell = np.eye(3) * 10.0
    atoms.pbc = True
    with pytest.raises(NotImplementedError, match="periodic"):
        list(iter_langevin_frames(atoms, LennardJones(), LangevinConfig(steps=1)))
