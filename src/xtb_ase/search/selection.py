"""Diversity selection primitives for trajectory post-processing."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase import units
from numpy.typing import ArrayLike


try:
    from numba import njit as _njit
except ImportError:  # pragma: no cover - exercised through public behavior
    _njit = None


if _njit is not None:

    @_njit(cache=True)
    def _fps_numba(features: np.ndarray, n_samples: int, start_index: int) -> np.ndarray:
        n_rows = features.shape[0]
        selected = np.empty(n_samples, dtype=np.int64)
        chosen = np.zeros(n_rows, dtype=np.bool_)
        min_distances = np.full(n_rows, np.inf, dtype=np.float64)

        selected[0] = start_index
        chosen[start_index] = True
        for row in range(n_rows):
            distance = 0.0
            for column in range(features.shape[1]):
                delta = features[row, column] - features[start_index, column]
                distance += delta * delta
            min_distances[row] = distance
        min_distances[start_index] = -np.inf

        for selection_index in range(1, n_samples):
            best_index = 0
            best_distance = -np.inf
            for row in range(n_rows):
                if not chosen[row] and min_distances[row] > best_distance:
                    best_index = row
                    best_distance = min_distances[row]
            selected[selection_index] = best_index
            chosen[best_index] = True
            min_distances[best_index] = -np.inf
            for row in range(n_rows):
                if chosen[row]:
                    continue
                distance = 0.0
                for column in range(features.shape[1]):
                    delta = features[row, column] - features[best_index, column]
                    distance += delta * delta
                if distance < min_distances[row]:
                    min_distances[row] = distance
        return selected

else:
    _fps_numba = None


def available_fps_backends() -> tuple[str, ...]:
    """Return FPS implementations available in the current environment."""

    return ("numpy", "numba") if _fps_numba is not None else ("numpy",)


def _validate_features(features: ArrayLike) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a 2D array")
    if matrix.shape[0] == 0:
        raise ValueError("features must contain at least one row")
    if matrix.shape[1] == 0:
        raise ValueError("features must contain at least one column")
    if not np.isfinite(matrix).all():
        raise ValueError("features must be finite")
    return np.ascontiguousarray(matrix)


def _validate_sample_count(n_samples: int, n_rows: int) -> int:
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError("n_samples must be a positive integer")
    n_samples = int(n_samples)
    if not 0 < n_samples <= n_rows:
        raise ValueError("n_samples must be between 1 and the number of rows")
    return n_samples


def _validate_backend(backend: str) -> str:
    backend = str(backend).lower()
    if backend not in {"auto", "numpy", "numba"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'numba'")
    if backend == "auto":
        return "numba" if _fps_numba is not None else "numpy"
    if backend == "numba" and _fps_numba is None:
        raise ImportError(
            "Numba FPS was requested but numba is not installed; "
            "install it with `pip install 'xtb-ase[search]'`"
        )
    return backend


def farthest_point_sampling(
    features: ArrayLike,
    n_samples: int,
    *,
    backend: str = "auto",
    start_index: int = 0,
) -> np.ndarray:
    """Select a farthest-point subset using Euclidean feature distance."""

    matrix = _validate_features(features)
    n_samples = _validate_sample_count(n_samples, matrix.shape[0])
    if isinstance(start_index, bool) or not isinstance(start_index, (int, np.integer)):
        raise ValueError("start_index must be an integer")
    start_index = int(start_index)
    if not 0 <= start_index < matrix.shape[0]:
        raise ValueError("start_index must be a valid feature row")
    backend = _validate_backend(backend)
    if backend == "numba":
        return np.asarray(_fps_numba(matrix, n_samples, start_index), dtype=int)

    selected = np.empty(n_samples, dtype=int)
    chosen = np.zeros(matrix.shape[0], dtype=bool)
    min_distances = np.sum((matrix - matrix[start_index]) ** 2, axis=1)
    selected[0] = start_index
    chosen[start_index] = True
    min_distances[start_index] = -np.inf
    for selection_index in range(1, n_samples):
        candidate = int(np.argmax(min_distances))
        selected[selection_index] = candidate
        chosen[candidate] = True
        distances = np.sum((matrix - matrix[candidate]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, distances)
        min_distances[chosen] = -np.inf
    return selected


def _allocate_strata(counts: np.ndarray, n_samples: int) -> np.ndarray:
    quotas = np.zeros_like(counts, dtype=int)
    for _ in range(n_samples):
        scores = np.full(counts.shape, -np.inf, dtype=float)
        for index, count in enumerate(counts):
            if quotas[index] < count:
                scores[index] = 1.0 / ((index + 1) * (quotas[index] + 1))
        chosen = int(np.argmax(scores))
        quotas[chosen] += 1
    return quotas


def energy_stratified_fps(
    features: ArrayLike,
    energies_eV: Sequence[float] | ArrayLike,
    n_samples: int,
    *,
    energy_window_kcal_mol: float | None = 6.0,
    energy_bins: int = 4,
    backend: str = "auto",
) -> np.ndarray:
    """Select diverse rows while retaining low-energy strata.

    Energies and returned indices refer to the original row order.  The
    relative energy window is measured from the lowest finite energy.
    """

    matrix = _validate_features(features)
    energies = np.asarray(energies_eV, dtype=float)
    if energies.ndim != 1 or energies.shape[0] != matrix.shape[0]:
        raise ValueError("energies_eV must contain one value per feature row")
    if not np.isfinite(energies).all():
        raise ValueError("energies_eV must be finite")
    if isinstance(energy_bins, bool) or not isinstance(energy_bins, (int, np.integer)):
        raise ValueError("energy_bins must be a positive integer")
    energy_bins = int(energy_bins)
    if energy_bins <= 0:
        raise ValueError("energy_bins must be a positive integer")
    if energy_window_kcal_mol is not None:
        if not np.isfinite(energy_window_kcal_mol) or energy_window_kcal_mol < 0:
            raise ValueError("energy_window_kcal_mol must be non-negative or None")

    lowest = float(np.min(energies))
    delta = energies - lowest
    if energy_window_kcal_mol is None:
        eligible = np.arange(len(energies), dtype=int)
    else:
        window_eV = float(energy_window_kcal_mol) * units.kcal / units.mol
        eligible = np.flatnonzero(delta <= window_eV + 1e-15)
    if eligible.size == 0:
        raise ValueError("no structures fall within the requested energy window")

    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError("n_samples must be a positive integer")
    if int(n_samples) <= 0:
        raise ValueError("n_samples must be a positive integer")
    n_samples = min(int(n_samples), eligible.size)
    eligible_delta = delta[eligible]
    maximum_delta = float(np.max(eligible_delta))
    if maximum_delta <= 1e-15:
        bin_ids = np.zeros(eligible.size, dtype=int)
        n_bins = 1
    else:
        n_bins = energy_bins
        bin_ids = np.floor(eligible_delta / maximum_delta * n_bins).astype(int)
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    counts = np.bincount(bin_ids, minlength=n_bins)
    quotas = _allocate_strata(counts, n_samples)

    selected: list[int] = []
    for bin_index, quota in enumerate(quotas):
        if quota == 0:
            continue
        local_positions = np.flatnonzero(bin_ids == bin_index)
        local_indices = eligible[local_positions]
        local_order = np.lexsort((local_indices, energies[local_indices]))
        local_indices = local_indices[local_order]
        local_features = matrix[local_indices]
        local_selected = farthest_point_sampling(
            local_features,
            int(quota),
            backend=backend,
            start_index=0,
        )
        selected.extend(int(local_indices[index]) for index in local_selected)

    lowest_index = int(np.lexsort((eligible, energies[eligible]))[0])
    lowest_source_index = int(eligible[lowest_index])
    if lowest_source_index not in selected:
        selected[0] = lowest_source_index
    return np.asarray(selected, dtype=int)
