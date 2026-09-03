"""Optional-dependency-safe conformer search helpers."""

from .geometry import ordered_kabsch_rmsd, pair_distance_fingerprint
from .pipeline import (
    Candidate,
    ConformerGroup,
    ConformerSearch,
    ConformerSearchResult,
    SearchConfig,
)
from .pamssw import (
    PAMSSWComparisonConfig,
    PAMSSWComparisonResult,
    run_pamssw,
)
from .sampling import LangevinConfig, iter_langevin_frames, sample_langevin_frames
from .selection import (
    available_fps_backends,
    energy_stratified_fps,
    farthest_point_sampling,
)

__all__ = [
    "Candidate",
    "ConformerGroup",
    "ConformerSearch",
    "ConformerSearchResult",
    "LangevinConfig",
    "PAMSSWComparisonConfig",
    "PAMSSWComparisonResult",
    "SearchConfig",
    "available_fps_backends",
    "energy_stratified_fps",
    "farthest_point_sampling",
    "iter_langevin_frames",
    "ordered_kabsch_rmsd",
    "pair_distance_fingerprint",
    "run_pamssw",
    "sample_langevin_frames",
]
