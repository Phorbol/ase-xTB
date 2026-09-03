import numpy as np
import pytest

from xtb_ase.search.selection import (
    available_fps_backends,
    energy_stratified_fps,
    farthest_point_sampling,
)


def test_numpy_fps_starts_at_requested_index_and_returns_unique_indices():
    features = np.asarray([[0.0], [1.0], [4.0], [8.0]])
    selected = farthest_point_sampling(features, 3, backend="numpy", start_index=1)
    assert selected.tolist() == [1, 3, 2]
    assert len(set(selected.tolist())) == 3


def test_energy_stratified_fps_keeps_lowest_energy_and_respects_window():
    features = np.asarray([[0.0], [0.2], [1.0], [3.0], [9.0]])
    energies = np.asarray([0.0, 0.01, 0.04, 0.20, 0.80])
    selected = energy_stratified_fps(
        features,
        energies,
        n_samples=3,
        energy_window_kcal_mol=6.0,
        energy_bins=3,
        backend="numpy",
    )
    assert 0 in selected
    assert 4 not in selected


def test_numba_backend_is_explicitly_unavailable_when_numba_is_missing():
    if "numba" in available_fps_backends():
        pytest.skip("Numba is installed in this environment")
    with pytest.raises(ImportError, match="numba"):
        farthest_point_sampling(np.zeros((2, 1)), 1, backend="numba")


def test_fps_rejects_invalid_features_and_sample_count():
    with pytest.raises(ValueError, match="2D"):
        farthest_point_sampling(np.zeros(3), 1)
    with pytest.raises(ValueError, match="n_samples"):
        farthest_point_sampling(np.zeros((2, 1)), 3)
