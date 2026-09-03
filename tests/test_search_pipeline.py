import importlib.util

import numpy as np
import pytest
from ase import Atoms

from xtb_ase.search.pipeline import ConformerSearch, SearchConfig


def water(x: float = 0.0) -> Atoms:
    return Atoms(
        "OH2",
        positions=[
            [x, 0.0, 0.0],
            [x + 0.76, 0.0, 0.5],
            [x - 0.76, 0.0, 0.5],
        ],
    )


def test_selector_reads_cached_info_energy_and_keeps_lowest_representative():
    frames = [water(), water(1.0), water(2.0)]
    for frame, energy in zip(frames, [0.10, 0.00, 0.20]):
        frame.info["energy"] = energy
    result = ConformerSearch(
        SearchConfig(
            energy_window_kcal_mol=None,
            max_selected=3,
            energy_bins=1,
            rmsd_backend="ordered",
            rmsd_tolerance_angstrom=1e-10,
        )
    ).select(frames)
    assert result.groups[0].representative_index == 1
    assert result.representatives[0].info["energy"] == pytest.approx(0.0)
    assert result.groups[0].member_indices == (1, 0, 2)


def test_selector_marks_distance_backend_as_approximate():
    result = ConformerSearch(
        SearchConfig(
            energy_window_kcal_mol=None,
            max_selected=2,
            energy_bins=1,
            rmsd_backend="distance_fingerprint",
            rmsd_tolerance_angstrom=1e-8,
        )
    ).select([water(), water(1.0)], energies=[0.0, 0.1])
    assert result.diagnostics["dedup_exact"] is False
    assert len(result.groups) == 1


def test_selector_accepts_an_injected_descriptor_callable():
    frames = [water(), water(1.0)]
    result = ConformerSearch(
        SearchConfig(
            energy_window_kcal_mol=None,
            max_selected=2,
            energy_bins=1,
            rmsd_backend="distance_fingerprint",
        )
    ).select(
        frames,
        energies=[0.0, 0.1],
        descriptor_fn=lambda atoms: np.asarray([atoms.positions[1, 0]]),
    )
    assert result.candidates[0].descriptor.shape == (1,)
    assert result.prefilter_indices == (0, 1)


def test_irmsd_backend_fails_closed_without_optional_package():
    if importlib.util.find_spec("irmsd") is not None:
        pytest.skip("irmsd is installed in this environment")
    frame = water()
    frame.info["energy"] = 0.0
    with pytest.raises(ImportError, match="irmsd"):
        ConformerSearch(SearchConfig(rmsd_backend="irmsd")).select([frame])


def test_search_namespace_exports_implemented_symbols():
    from xtb_ase.search import ConformerSearch as ExportedSearch
    from xtb_ase.search import SearchConfig as ExportedConfig

    assert ExportedSearch is ConformerSearch
    assert ExportedConfig is SearchConfig
