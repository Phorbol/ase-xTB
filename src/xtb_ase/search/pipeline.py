"""End-to-end trajectory selection and conformer deduplication."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms, units

from .geometry import (
    ordered_kabsch_rmsd,
    pair_distance_fingerprint,
    standardize_features,
)
from .selection import (
    available_fps_backends,
    energy_stratified_fps,
)


DescriptorFunction = Callable[[Atoms], Sequence[float] | np.ndarray]


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for trajectory down-selection and conformer grouping."""

    energy_window_kcal_mol: float | None = 6.0
    max_selected: int = 32
    energy_bins: int = 4
    fps_backend: str = "auto"
    rmsd_backend: str = "irmsd"
    rmsd_tolerance_angstrom: float = 0.125
    irmsd_inversion: int = 0

    def __post_init__(self) -> None:
        if self.energy_window_kcal_mol is not None:
            if not np.isfinite(self.energy_window_kcal_mol) or self.energy_window_kcal_mol < 0.0:
                raise ValueError("energy_window_kcal_mol must be non-negative or None")
        for name, value in (
            ("max_selected", self.max_selected),
            ("energy_bins", self.energy_bins),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.fps_backend not in {"auto", "numpy", "numba"}:
            raise ValueError("fps_backend must be 'auto', 'numpy', or 'numba'")
        if self.rmsd_backend not in {"irmsd", "distance_fingerprint", "ordered"}:
            raise ValueError(
                "rmsd_backend must be 'irmsd', 'distance_fingerprint', or 'ordered'"
            )
        if not np.isfinite(self.rmsd_tolerance_angstrom) or self.rmsd_tolerance_angstrom <= 0.0:
            raise ValueError("rmsd_tolerance_angstrom must be positive")
        if self.irmsd_inversion not in {0, 1, 2}:
            raise ValueError("irmsd_inversion must be 0, 1, or 2")


@dataclass(frozen=True)
class Candidate:
    """One detached input frame with its source index, energy, and descriptor."""

    index: int
    atoms: Atoms
    energy_eV: float
    descriptor: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.index, (int, np.integer)) or self.index < 0:
            raise ValueError("candidate index must be a non-negative integer")
        if not np.isfinite(self.energy_eV):
            raise ValueError("candidate energy must be finite")
        if not isinstance(self.atoms, Atoms):
            raise TypeError("candidate atoms must be an ase.Atoms object")
        descriptor = np.asarray(self.descriptor, dtype=float)
        if descriptor.ndim != 1 or descriptor.size == 0:
            raise ValueError("candidate descriptor must be a non-empty 1D array")
        if not np.isfinite(descriptor).all():
            raise ValueError("candidate descriptor must be finite")
        detached = self.atoms.copy()
        detached.calc = None
        object.__setattr__(self, "atoms", detached)
        object.__setattr__(self, "energy_eV", float(self.energy_eV))
        object.__setattr__(self, "descriptor", descriptor.copy())


@dataclass(frozen=True)
class ConformerGroup:
    """A lowest-energy representative and its source-frame members."""

    representative_index: int
    member_indices: tuple[int, ...]
    representative: Candidate


@dataclass(frozen=True)
class ConformerSearchResult:
    """Selection output with provenance and method diagnostics."""

    candidates: tuple[Candidate, ...]
    prefilter_indices: tuple[int, ...]
    groups: tuple[ConformerGroup, ...]
    diagnostics: dict[str, Any]

    @property
    def selected_indices(self) -> tuple[int, ...]:
        return tuple(group.representative_index for group in self.groups)

    @property
    def representatives(self) -> tuple[Atoms, ...]:
        return tuple(group.representative.atoms.copy() for group in self.groups)


class ConformerSearch:
    """Run the deterministic baseline selection pipeline on ASE frames."""

    def __init__(self, config: SearchConfig | None = None) -> None:
        self.config = SearchConfig() if config is None else config

    def select(
        self,
        frames: Iterable[Atoms],
        *,
        energies: Sequence[float] | np.ndarray | None = None,
        descriptors: Sequence[Sequence[float]] | np.ndarray | None = None,
        descriptor_fn: DescriptorFunction | None = None,
    ) -> ConformerSearchResult:
        frame_list = list(frames)
        self._validate_frames(frame_list)
        if descriptors is not None and descriptor_fn is not None:
            raise ValueError("provide descriptors or descriptor_fn, not both")

        energy_values = self._extract_energies(frame_list, energies)
        descriptor_matrix = self._build_descriptors(
            frame_list,
            descriptors=descriptors,
            descriptor_fn=descriptor_fn,
        )
        normalized_descriptors = standardize_features(descriptor_matrix)
        prefilter_count = min(
            len(frame_list),
            max(int(self.config.max_selected), 4 * int(self.config.max_selected)),
        )
        prefilter_indices = energy_stratified_fps(
            normalized_descriptors,
            energy_values,
            prefilter_count,
            energy_window_kcal_mol=self.config.energy_window_kcal_mol,
            energy_bins=int(self.config.energy_bins),
            backend=self.config.fps_backend,
        )
        candidates = tuple(
            Candidate(
                index=index,
                atoms=frame_list[index],
                energy_eV=energy_values[index],
                descriptor=normalized_descriptors[index],
            )
            for index in range(len(frame_list))
        )

        irmsd_function = None
        if self.config.rmsd_backend == "irmsd":
            irmsd_function = self._load_irmsd()
        geometry_fingerprints: dict[int, np.ndarray] = {}
        ordered_indices = sorted(
            (int(index) for index in prefilter_indices),
            key=lambda index: (candidates[index].energy_eV, index),
        )
        group_members: list[list[int]] = []
        group_representatives: list[int] = []
        for index in ordered_indices:
            candidate = candidates[index]
            duplicate_group = None
            for group_index, representative_index in enumerate(group_representatives):
                distance = self._dedup_distance(
                    candidate.atoms,
                    candidates[representative_index].atoms,
                    geometry_fingerprints,
                    candidate.index,
                    representative_index,
                    irmsd_function,
                )
                if distance <= self.config.rmsd_tolerance_angstrom:
                    duplicate_group = group_index
                    break
            if duplicate_group is None:
                group_representatives.append(index)
                group_members.append([index])
                if len(group_members) >= self.config.max_selected:
                    break
            else:
                group_members[duplicate_group].append(index)

        groups = tuple(
            ConformerGroup(
                representative_index=representative_index,
                member_indices=tuple(members),
                representative=candidates[representative_index],
            )
            for representative_index, members in zip(
                group_representatives, group_members
            )
        )
        diagnostics = {
            "n_input": len(frame_list),
            "n_in_window": int(
                np.count_nonzero(
                    energy_values
                    <= np.min(energy_values)
                    + (
                        np.inf
                        if self.config.energy_window_kcal_mol is None
                        else self.config.energy_window_kcal_mol * units.kcal / units.mol
                    )
                )
            ),
            "n_prefilter": len(prefilter_indices),
            "n_groups": len(groups),
            "fps_backend": (
                "numba"
                if self.config.fps_backend == "auto"
                and "numba" in available_fps_backends()
                else self.config.fps_backend
            ),
            "dedup_backend": self.config.rmsd_backend,
            "dedup_exact": self.config.rmsd_backend == "irmsd",
            "energy_window_kcal_mol": self.config.energy_window_kcal_mol,
        }
        return ConformerSearchResult(
            candidates=candidates,
            prefilter_indices=tuple(int(index) for index in prefilter_indices),
            groups=groups,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _validate_frames(frames: Sequence[Atoms]) -> None:
        if not frames:
            raise ValueError("frames must contain at least one structure")
        first_numbers = None
        for index, atoms in enumerate(frames):
            if not isinstance(atoms, Atoms):
                raise TypeError(f"frames[{index}] must be an ase.Atoms object")
            if len(atoms) == 0:
                raise ValueError("frames must not contain empty structures")
            if atoms.pbc.any():
                raise NotImplementedError(
                    "the baseline selector does not support periodic frames"
                )
            positions = np.asarray(atoms.positions, dtype=float)
            if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
                raise ValueError(f"frames[{index}] positions must be finite with shape (N, 3)")
            numbers = np.asarray(atoms.numbers, dtype=np.int32)
            sorted_numbers = np.sort(numbers)
            if first_numbers is None:
                first_numbers = sorted_numbers
            elif not np.array_equal(first_numbers, sorted_numbers):
                raise ValueError("all frames must have the same composition")

    @staticmethod
    def _extract_energies(
        frames: Sequence[Atoms],
        energies: Sequence[float] | np.ndarray | None,
    ) -> np.ndarray:
        if energies is not None:
            values = np.asarray(energies, dtype=float)
            if values.ndim != 1 or len(values) != len(frames):
                raise ValueError("energies must contain one value per frame")
        else:
            values = np.empty(len(frames), dtype=float)
            for index, atoms in enumerate(frames):
                value = atoms.info.get("energy")
                if value is None and atoms.calc is not None:
                    value = getattr(atoms.calc, "results", {}).get("energy")
                if value is None:
                    try:
                        value = atoms.get_potential_energy()
                    except Exception as exc:
                        raise ValueError(
                            f"frame {index} has no cached energy and cannot be evaluated"
                        ) from exc
                values[index] = value
        if not np.isfinite(values).all():
            raise ValueError("energies must be finite")
        return values

    @staticmethod
    def _build_descriptors(
        frames: Sequence[Atoms],
        *,
        descriptors: Sequence[Sequence[float]] | np.ndarray | None,
        descriptor_fn: DescriptorFunction | None,
    ) -> np.ndarray:
        if descriptors is not None:
            matrix = np.asarray(descriptors, dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != len(frames):
                raise ValueError("descriptors must have shape (n_frames, n_features)")
            if matrix.shape[1] == 0 or not np.isfinite(matrix).all():
                raise ValueError("descriptors must be finite and non-empty")
            return np.ascontiguousarray(matrix)
        function = pair_distance_fingerprint if descriptor_fn is None else descriptor_fn
        rows: list[np.ndarray] = []
        for index, atoms in enumerate(frames):
            row = np.asarray(function(atoms), dtype=float)
            if row.ndim != 1 or row.size == 0:
                raise ValueError(f"descriptor for frame {index} must be a non-empty 1D array")
            if not np.isfinite(row).all():
                raise ValueError(f"descriptor for frame {index} must be finite")
            rows.append(row)
        lengths = {row.size for row in rows}
        if len(lengths) != 1:
            raise ValueError("all descriptors must have the same length")
        return np.ascontiguousarray(np.vstack(rows))

    @staticmethod
    def _load_irmsd():
        try:
            from irmsd import get_irmsd_ase
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "rmsd_backend='irmsd' requires the optional irmsd package; "
                "install it with `pip install 'xtb-ase[search]'`"
            ) from exc
        return get_irmsd_ase

    def _dedup_distance(
        self,
        candidate: Atoms,
        representative: Atoms,
        fingerprint_cache: dict[int, np.ndarray],
        candidate_index: int,
        representative_index: int,
        irmsd_function,
    ) -> float:
        if self.config.rmsd_backend == "ordered":
            return ordered_kabsch_rmsd(candidate, representative)
        if self.config.rmsd_backend == "distance_fingerprint":
            if candidate_index not in fingerprint_cache:
                fingerprint_cache[candidate_index] = pair_distance_fingerprint(candidate)
            if representative_index not in fingerprint_cache:
                fingerprint_cache[representative_index] = pair_distance_fingerprint(representative)
            first = fingerprint_cache[candidate_index]
            second = fingerprint_cache[representative_index]
            if first.shape != second.shape:
                return float("inf")
            return float(np.sqrt(np.mean((first - second) ** 2)))
        if irmsd_function is None:  # pragma: no cover - guarded in select
            raise RuntimeError("iRMSD function was not initialized")
        result = irmsd_function(
            candidate,
            representative,
            iinversion=self.config.irmsd_inversion,
        )
        distance = float(result[0])
        if not np.isfinite(distance):
            raise ValueError("iRMSD backend returned a non-finite distance")
        return distance
