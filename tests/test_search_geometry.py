import numpy as np
import pytest
from ase import Atoms

from xtb_ase.search.geometry import (
    ordered_kabsch_rmsd,
    pair_distance_fingerprint,
    standardize_features,
)


def water() -> Atoms:
    return Atoms(
        "OH2",
        positions=[
            [0.0, 0.0, 0.0],
            [0.76, 0.0, 0.50],
            [-0.76, 0.0, 0.50],
        ],
    )


def test_pair_distance_fingerprint_is_rigid_and_same_element_permutation_invariant():
    first = water()
    second = first.copy()
    second.rotate(37.0, "z")
    second.translate([3.0, -1.0, 0.4])
    second = second[[0, 2, 1]]

    np.testing.assert_allclose(
        pair_distance_fingerprint(first), pair_distance_fingerprint(second)
    )


def test_ordered_kabsch_rmsd_is_zero_for_a_rigid_transform():
    first = water()
    second = first.copy()
    second.rotate(37.0, "z")
    second.translate([3.0, -1.0, 0.4])

    assert ordered_kabsch_rmsd(first, second) == pytest.approx(0.0, abs=1e-12)


def test_geometry_rejects_different_composition():
    with pytest.raises(ValueError, match="composition"):
        ordered_kabsch_rmsd(water(), Atoms("H3", positions=np.zeros((3, 3))))


def test_standardize_features_handles_constant_columns_and_rejects_nonfinite():
    result = standardize_features(np.asarray([[1.0, 4.0], [3.0, 4.0]]))
    np.testing.assert_allclose(result[:, 1], 0.0)
    with pytest.raises(ValueError, match="finite"):
        standardize_features(np.asarray([[np.nan, 1.0]]))
