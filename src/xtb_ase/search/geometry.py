"""Geometry descriptors and explicit, fixed-order RMSD helpers."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from numpy.typing import ArrayLike


def _validated_geometry(atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(atoms, Atoms):
        raise TypeError("geometry helpers expect ase.Atoms objects")
    numbers = np.asarray(atoms.numbers, dtype=np.int32)
    positions = np.asarray(atoms.positions, dtype=float)
    if numbers.ndim != 1 or positions.shape != (len(numbers), 3):
        raise ValueError("atomic numbers and positions must describe an (N, 3) geometry")
    if len(numbers) == 0:
        raise ValueError("geometry must contain at least one atom")
    if not np.isfinite(positions).all():
        raise ValueError("atomic positions must be finite")
    return numbers, positions


def pair_distance_fingerprint(atoms: Atoms) -> np.ndarray:
    """Return sorted pair distances grouped by unordered atomic-number pair.

    The result is invariant to rigid translation/rotation and to reordering
    atoms within the same element class.  It is a cheap prefilter only: a
    distance multiset is not a complete molecular graph invariant and is not
    an RMSD.
    """

    numbers, positions = _validated_geometry(atoms)
    unique_numbers = np.unique(numbers)
    blocks: list[np.ndarray] = []
    for first_index, first_number in enumerate(unique_numbers):
        first_atoms = np.flatnonzero(numbers == first_number)
        for second_number in unique_numbers[first_index:]:
            second_atoms = np.flatnonzero(numbers == second_number)
            if first_number == second_number:
                if len(first_atoms) < 2:
                    continue
                deltas = positions[first_atoms][:, None, :] - positions[first_atoms][None, :, :]
                distances = np.linalg.norm(deltas, axis=-1)
                values = distances[np.triu_indices(len(first_atoms), k=1)]
            else:
                deltas = positions[first_atoms][:, None, :] - positions[second_atoms][None, :, :]
                values = np.linalg.norm(deltas, axis=-1).reshape(-1)
            blocks.append(np.sort(np.asarray(values, dtype=float)))

    if not blocks:
        return np.zeros(1, dtype=float)
    return np.concatenate(blocks).astype(float, copy=False)


def ordered_kabsch_rmsd(first: Atoms, second: Atoms) -> float:
    """Compute optimal rigid RMSD while preserving atom order.

    This function deliberately does not solve atom assignment.  It is an
    explicit fallback for controlled tests; use the optional ``irmsd``
    backend for symmetry/permutation-aware conformer classification.
    """

    first_numbers, first_positions = _validated_geometry(first)
    second_numbers, second_positions = _validated_geometry(second)
    if len(first_numbers) != len(second_numbers):
        raise ValueError("geometries must have the same atom count")
    if not np.array_equal(first_numbers, second_numbers):
        raise ValueError("geometries must have the same ordered composition")
    if len(first_numbers) == 1:
        return 0.0

    first_centered = first_positions - first_positions.mean(axis=0)
    second_centered = second_positions - second_positions.mean(axis=0)
    covariance = second_centered.T @ first_centered
    left, _, right_transposed = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_transposed) < 0.0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_transposed
    residual = second_centered @ rotation - first_centered
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def standardize_features(features: ArrayLike) -> np.ndarray:
    """Standardize columns of a finite two-dimensional feature matrix."""

    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a 2D array")
    if matrix.shape[0] == 0:
        raise ValueError("features must contain at least one row")
    if not np.isfinite(matrix).all():
        raise ValueError("features must be finite")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (matrix - mean) / scale
