# Conformer Search Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dependency-light, ASE-native GFN-FF trajectory sampling and conformer selection baseline with optional iRMSD/Numba backends, g-xTB thermochemistry results, and a PAM-SSW global-search comparison adapter.

**Architecture:** Keep trajectory generation, geometry descriptors, selection/deduplication, optional refinement, and external global-search comparison as separate modules. The baseline consumes or generates ASE `Atoms`, uses eV internally, and exposes source indices and diagnostics so MACE/TorchSim can be injected later without changing selection semantics. Optional scientific backends are loaded lazily and fail closed when an exact capability is requested but unavailable.

**Tech Stack:** Python 3.10+, ASE, NumPy, optional Numba, optional `irmsd`, optional local/installed `pamssw`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-conformer-search-baseline-design.md`

## Global Constraints

- Preserve the existing `xtb_ase` package name and calculator APIs.
- Keep NumPy/ASE as the base dependencies; Numba, iRMSD, and PAM-SSW remain optional.
- Use ASE eV, eV/Å, and Å internally; convert the public energy window from kcal/mol with `ase.units.kcal / ase.units.mol`.
- Never label a pair-distance fingerprint as RMSD or exact structural equivalence.
- `rmsd_backend="irmsd"` must raise an actionable optional-dependency error when `irmsd` is absent.
- Do not modify the `pam-ssw` checkout; only import its public `State`, `SSWConfig`, `run_ssw`, and `ASECalculator` interfaces.
- Every new behavior starts with a failing test and ends with a focused test plus the full suite.
- Do not claim benchmark or scientific improvement; this plan only delivers the executable baseline and comparison hooks.

---

### Task 1: Add the optional search dependency group and package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/xtb_ase/search/__init__.py`
- Test: `tests/test_public_api.py`

**Interfaces:**
- Produces the import namespace `xtb_ase.search` and the optional extra `xtb-ase[search]`.
- Does not import Numba, iRMSD, MACE, TorchSim, or PAM-SSW at package import time.

- [ ] **Step 1: Write the failing public-import test**

```python
def test_search_namespace_is_importable_without_optional_backends():
    import xtb_ase.search as search

    assert search.__name__ == "xtb_ase.search"
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `python -m pytest -q tests/test_public_api.py::test_search_namespace_is_importable_without_optional_backends`

Expected: FAIL with `ModuleNotFoundError: No module named 'xtb_ase.search'`.

- [ ] **Step 3: Add the optional dependency group and package exports**

Add to `pyproject.toml`:

```toml
search = [
  "numba>=0.58",
  "irmsd>=0.1.1",
]
```

Create `src/xtb_ase/search/__init__.py` as an empty package marker. The public
symbol re-exports are added after their implementation modules exist in Task 5;
this task only proves that importing the namespace does not load optional backends.

```python
"""Optional-dependency-safe conformer search helpers."""

__all__: list[str] = []
```

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `python -m pytest -q tests/test_public_api.py::test_search_namespace_is_importable_without_optional_backends`

Expected: PASS.

- [ ] **Step 5: Commit the package skeleton**

```bash
git add pyproject.toml src/xtb_ase/search/__init__.py tests/test_public_api.py
git commit -m "feat: add conformer search package namespace"
```

### Task 2: Implement geometry fingerprints and explicit ordered RMSD

**Files:**
- Create: `src/xtb_ase/search/geometry.py`
- Test: `tests/test_search_geometry.py`

**Interfaces:**
- `pair_distance_fingerprint(atoms: ase.Atoms) -> numpy.ndarray`
- `ordered_kabsch_rmsd(first: ase.Atoms, second: ase.Atoms) -> float`
- `standardize_features(features: array-like) -> numpy.ndarray`

- [ ] **Step 1: Write failing geometry tests**

```python
import numpy as np
import pytest
from ase import Atoms

from xtb_ase.search.geometry import (
    ordered_kabsch_rmsd,
    pair_distance_fingerprint,
    standardize_features,
)


def water():
    return Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.76, 0.0, 0.50], [-0.76, 0.0, 0.50]],
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
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest -q tests/test_search_geometry.py`

Expected: FAIL because `xtb_ase.search.geometry` does not exist.

- [ ] **Step 3: Implement geometry helpers**

Validate non-empty finite `(N, 3)` coordinates and use sorted unordered atomic-number pairs. For pair type `(zi, zj)`, collect all `i < j` distances with those numbers and concatenate sorted blocks in lexicographic pair order. Raise `ValueError` when the input has no atoms, malformed coordinates, or non-finite values. Implement Kabsch alignment with centering, SVD, determinant correction, and fixed input atom order; reject different atom counts or ordered atomic numbers.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python -m pytest -q tests/test_search_geometry.py`

Expected: PASS.

- [ ] **Step 5: Commit geometry helpers**

```bash
git add src/xtb_ase/search/geometry.py tests/test_search_geometry.py
git commit -m "feat: add invariant geometry fingerprints"
```

### Task 3: Implement NumPy/Numba FPS and energy-stratified selection primitives

**Files:**
- Create: `src/xtb_ase/search/selection.py`
- Test: `tests/test_search_selection.py`

**Interfaces:**
- `farthest_point_sampling(features, n_samples, backend="auto", start_index=0) -> numpy.ndarray`
- `energy_stratified_fps(features, energies_eV, n_samples, energy_window_kcal_mol, energy_bins, backend="auto") -> numpy.ndarray`
- `available_fps_backends() -> tuple[str, ...]`

- [ ] **Step 1: Write failing FPS and energy-window tests**

```python
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
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest -q tests/test_search_selection.py`

Expected: FAIL because the selection module and functions do not exist.

- [ ] **Step 3: Implement the minimal selection primitives**

Normalize a finite two-dimensional feature matrix. The NumPy implementation maintains the minimum squared distance to the selected set and uses `np.argmax` with the first-index tie rule. The optional Numba implementation must produce the same indices for the same input and avoid Python callbacks inside the kernel. `backend="auto"` selects Numba when importable and otherwise NumPy; `backend="numba"` raises an installation error when unavailable.

For energy selection, convert kcal/mol to eV, retain `delta_e <= window`, assign equal-width bins between zero and the largest retained delta, allocate slots by repeatedly choosing the non-empty bin with the largest `1 / ((bin_index + 1) * (count + 1))` score, and run FPS within each bin from its lowest-energy member. Always include the global lowest-energy retained index. Return source indices in deterministic bin/selection order and never return an index outside the energy window.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python -m pytest -q tests/test_search_selection.py`

Expected: PASS.

- [ ] **Step 5: Commit selection primitives**

```bash
git add src/xtb_ase/search/selection.py tests/test_search_selection.py
git commit -m "feat: add energy-stratified FPS selection"
```

### Task 4: Implement Langevin trajectory generation

**Files:**
- Create: `src/xtb_ase/search/sampling.py`
- Test: `tests/test_search_sampling.py`

**Interfaces:**
- `LangevinConfig(temperature_K=..., timestep_fs=..., friction_per_fs=..., steps=..., sample_interval=..., rng_seed=...)`
- `iter_langevin_frames(atoms, calculator, config) -> Iterator[ase.Atoms]`
- `sample_langevin_frames(atoms, calculator, config) -> list[ase.Atoms]`

- [ ] **Step 1: Write failing sampling tests**

```python
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones

from xtb_ase.search.sampling import (
    LangevinConfig,
    iter_langevin_frames,
    sample_langevin_frames,
)


def lj_dimer():
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
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest -q tests/test_search_sampling.py`

Expected: FAIL because `xtb_ase.search.sampling` does not exist.

- [ ] **Step 3: Implement the ASE Langevin iterator**

Validate positive temperature, timestep, friction, and sample interval, non-negative steps, and non-negative integer seed. Reject non-empty periodic cells in this helper with `NotImplementedError` because the baseline is a non-periodic molecular/cluster sampler. Copy the input, attach the supplied calculator, initialize Maxwell-Boltzmann velocities with the supplied NumPy generator, remove center-of-mass momentum with `Stationary`, and use ASE `Langevin` with internal-unit conversions. Yield step zero and every `sample_interval` step through `steps`; each yielded copy must have `calc = None` and set `frame.info["md_step"]` and `frame.info["temperature_K"]`. The list helper simply materializes the iterator.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python -m pytest -q tests/test_search_sampling.py`

Expected: PASS.

- [ ] **Step 5: Commit trajectory generation**

```bash
git add src/xtb_ase/search/sampling.py tests/test_search_sampling.py
git commit -m "feat: add reproducible ASE Langevin sampler"
```

### Task 5: Implement candidate construction, iRMSD/distance deduplication, and result objects

**Files:**
- Create: `src/xtb_ase/search/pipeline.py`
- Test: `tests/test_search_pipeline.py`

**Interfaces:**
- `SearchConfig(...)`
- `Candidate(index, atoms, energy_eV, descriptor)`
- `ConformerSearch(config).select(frames, energies=None, descriptors=None, descriptor_fn=None) -> ConformerSearchResult`
- `ConformerSearchResult.representatives -> tuple[ase.Atoms, ...]`

- [ ] **Step 1: Write failing pipeline tests**

```python
import numpy as np
import pytest
from ase import Atoms

from xtb_ase.search.pipeline import ConformerSearch, SearchConfig


def water(x=0.0):
    return Atoms("OH2", positions=[[x, 0, 0], [x + 0.76, 0, 0.5], [x - 0.76, 0, 0.5]])


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


def test_search_namespace_exports_implemented_symbols():
    from xtb_ase.search import ConformerSearch, SearchConfig

    assert ConformerSearch is not None
    assert SearchConfig is not None


def test_selector_marks_distance_backend_as_approximate():
    frames = [water(), water(1.0)]
    result = ConformerSearch(
        SearchConfig(
            energy_window_kcal_mol=None,
            max_selected=2,
            energy_bins=1,
            rmsd_backend="distance_fingerprint",
            rmsd_tolerance_angstrom=1e-8,
        )
    ).select(frames, energies=[0.0, 0.1])
    assert result.diagnostics["dedup_exact"] is False


def test_irmsd_backend_fails_closed_without_optional_package():
    if __import__("importlib.util").util.find_spec("irmsd") is not None:
        pytest.skip("irmsd is installed in this environment")
    frame = water()
    frame.info["energy"] = 0.0
    with pytest.raises(ImportError, match="irmsd"):
        ConformerSearch(SearchConfig(rmsd_backend="irmsd")).select([frame])
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest -q tests/test_search_pipeline.py`

Expected: FAIL because the pipeline module and result objects do not exist.

- [ ] **Step 3: Implement the pipeline**

Define `SearchConfig` as a frozen dataclass with validated fields:

```python
energy_window_kcal_mol: float | None = 6.0
max_selected: int = 32
energy_bins: int = 4
fps_backend: str = "auto"
rmsd_backend: str = "irmsd"
rmsd_tolerance_angstrom: float = 0.125
```

Build each `Candidate` from a copied `Atoms`, finite energy, and either supplied descriptor, `descriptor_fn`, or `pair_distance_fingerprint`. Validate equal atom count and sorted composition across frames. Standardize the descriptor matrix, call `energy_stratified_fps` with an oversample size of `min(n_frames, max(4 * max_selected, max_selected))`, then sort preselected indices by `(energy_eV, source_index)` before deduplication.

Implement deduplication against current group representatives:

- `ordered`: call `ordered_kabsch_rmsd` and require ordered atomic numbers.
- `distance_fingerprint`: compare RMS distance of the default fingerprints and set `diagnostics["dedup_exact"] = False`.
- `irmsd`: lazy import `irmsd.get_irmsd_ase`, call it with `iinversion=0`, and use the returned first value; raise `ImportError("install xtb-ase[search] ...")` if unavailable.

Each group stores the representative source index, member source indices, and representative `Candidate`; because candidates are processed in ascending energy, the representative is the lowest-energy member. The result stores all candidates, prefilter indices, groups, selected indices, and diagnostics including `n_input`, `n_in_window`, `n_prefilter`, `n_groups`, `fps_backend`, and `dedup_exact`. Returned structures are copies with no shared mutable calculator.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python -m pytest -q tests/test_search_pipeline.py`

Expected: PASS.

- [ ] **Step 5: Commit the selection pipeline**

```bash
git add src/xtb_ase/search/pipeline.py tests/test_search_pipeline.py
git commit -m "feat: add conformer selection and dedup pipeline"
```

### Task 6: Add g-xTB thermochemistry parser and calculator results

**Files:**
- Modify: `src/xtb_ase/_parsers.py`
- Modify: `src/xtb_ase/gxtb.py`
- Modify: `src/xtb_ase/__init__.py`
- Modify: `tests/test_parsers.py`
- Modify: `tests/test_gxtb_calculator.py`
- Modify: `tests/test_public_api.py`

**Interfaces:**
- `ParsedThermochemistry`
- `parse_thermochemistry(text) -> ParsedThermochemistry`
- `GXTB.get_gibbs_free_energy(atoms=None, unit="eV") -> float`
- `GXTB.get_enthalpy(atoms=None, unit="eV") -> float`
- `GXTB.get_zero_point_energy(atoms=None, unit="eV") -> float`

- [ ] **Step 1: Write failing parser and calculator tests**

Add this fixture to `tests/test_parsers.py`:

```python
THERMO_STDOUT = """
         :: total energy             -76.432503870995 Eh    ::
         :: total free energy         -76.427574590877 Eh   ::
         :: zero point energy           0.022378390734 Eh   ::
          | TOTAL ENTHALPY            -76.406340598102 Eh   |
          | TOTAL FREE ENERGY         -76.427574590877 Eh   |
    298.15    0.378488E-02    0.261633E-01    0.212340E-01    0.492928E-02
"""


def test_parse_thermochemistry_reads_boxed_gxtb_values():
    parsed = parse_thermochemistry(THERMO_STDOUT)
    assert parsed.total_free_energy_hartree == pytest.approx(-76.427574590877)
    assert parsed.total_enthalpy_hartree == pytest.approx(-76.406340598102)
    assert parsed.zero_point_energy_hartree == pytest.approx(0.022378390734)
    assert parsed.temperature_kelvin == pytest.approx(298.15)
```

Extend the fake Hessian executable in `tests/test_gxtb_calculator.py` to print the same block and add:

```python
def test_gxtb_exposes_gibbs_free_energy_in_ev(tmp_path):
    executable = make_fake_xtb(tmp_path / "fake-xtb")
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0.7, 0, 0], [-0.7, 0, 0]])
    calc = GXTB(command=str(executable), directory=tmp_path, properties=("gibbs_free_energy",))
    atoms.calc = calc
    assert calc.get_gibbs_free_energy(atoms) == pytest.approx(-76.427574590877 * units.Hartree)
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest -q tests/test_parsers.py::test_parse_thermochemistry_reads_boxed_gxtb_values tests/test_gxtb_calculator.py::test_gxtb_exposes_gibbs_free_energy_in_ev`

Expected: FAIL because the parser, property, and accessor do not exist.

- [ ] **Step 3: Implement the parser and calculator property plumbing**

Add a frozen `ParsedThermochemistry` with optional Hartree fields for total energy, free energy, enthalpy, zero-point energy, RRHO contribution, and optional Kelvin temperature. Match the last occurrence of each label so the final thermodynamic block wins; accept both `:: label value Eh ::` and `| LABEL value Eh |`. Parse the temperature from a row beginning with a finite number followed by `H(0)-H(T)+PV`-compatible output; the fixture may use a dedicated regex that accepts the first numeric row only when a `T/K` header precedes it.

Add the numeric properties `free_energy`, `gibbs_free_energy`, `enthalpy`, `zero_point_energy`, and `thermochemical_correction` to the xTB optional property set. A request for any thermochemistry property triggers the existing Hessian command. Convert Hartree to eV in `self.results`; set both `free_energy` and `gibbs_free_energy` to the parsed total free energy. Add accessors accepting `unit="eV"` or `unit="hartree"`, and raise `OutputParseError` if the requested field is absent. Do not add entropy yet.

- [ ] **Step 4: Run the focused and existing calculator tests**

Run: `python -m pytest -q tests/test_parsers.py tests/test_gxtb_calculator.py tests/test_public_api.py`

Expected: PASS.

- [ ] **Step 5: Commit g-xTB thermochemistry support**

```bash
git add src/xtb_ase/_parsers.py src/xtb_ase/gxtb.py src/xtb_ase/__init__.py tests/test_parsers.py tests/test_gxtb_calculator.py tests/test_public_api.py
git commit -m "feat: expose g-xTB thermochemistry results"
```

### Task 7: Add the PAM-SSW comparison adapter

**Files:**
- Create: `src/xtb_ase/search/pamssw.py`
- Modify: `src/xtb_ase/search/__init__.py`
- Test: `tests/test_pamssw_adapter.py`

**Interfaces:**
- `PAMSSWComparisonConfig(target_uphill_energy_eV=0.05, max_trials=32, max_steps_per_walk=8, rng_seed=0, max_force_evals=None)`
- `PAMSSWComparisonResult(best_atoms, best_energy_eV, minima, archive_energies_eV, raw_result)`
- `run_pamssw(initial_atoms, calculator, config=PAMSSWComparisonConfig()) -> PAMSSWComparisonResult`

- [ ] **Step 1: Write failing adapter tests**

```python
import numpy as np
import pytest
from ase import Atoms

from xtb_ase.search.pamssw import PAMSSWComparisonConfig, run_pamssw


def test_pamssw_config_rejects_nonpositive_uphill_height():
    with pytest.raises(ValueError, match="target_uphill_energy"):
        PAMSSWComparisonConfig(target_uphill_energy_eV=0.0)


def test_pamssw_adapter_maps_state_and_archive(monkeypatch):
    import xtb_ase.search.pamssw as module

    seen = {}

    class FakeState:
        def __init__(self, **kwargs):
            seen["state"] = kwargs

    class FakeConfig:
        def __init__(self, **kwargs):
            seen["config"] = kwargs

    class FakeEntry:
        def __init__(self, state, energy):
            self.state = state
            self.energy = energy

    class FakeResult:
        best_state = FakeState(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
        best_energy = -1.0
        archive = type("Archive", (), {"entries": [FakeEntry(best_state, -1.0)]})()

    monkeypatch.setattr(module, "_load_pamssw", lambda: (FakeState, FakeConfig, lambda *args: FakeResult(), object))
    result = run_pamssw(Atoms("H", positions=[[0, 0, 0]]), object())
    assert seen["config"]["target_uphill_energy"] == pytest.approx(0.05)
    assert result.best_energy_eV == pytest.approx(-1.0)
    assert len(result.minima) == 1
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest -q tests/test_pamssw_adapter.py`

Expected: FAIL because the adapter module and configuration objects do not exist.

- [ ] **Step 3: Implement the lazy public-API adapter**

Validate positive trial/step counts and positive `target_uphill_energy_eV`. `_load_pamssw()` must import `pamssw.State`, `pamssw.SSWConfig`, `pamssw.run_ssw`, and `pamssw.calculators.ASECalculator`, returning those four objects; missing imports raise `ImportError("install or expose the pamssw checkout ...")`.

Convert `Atoms` to `State(numbers, positions, cell, pbc)` and pass `ASECalculator(calculator)` to `run_ssw`. Construct `SSWConfig(target_uphill_energy=config.target_uphill_energy_eV, max_trials=..., max_steps_per_walk=..., rng_seed=..., max_force_evals=...)`. Convert `result.best_state` and every `archive.entries[*].state` to detached ASE `Atoms`; preserve `pbc` and cell when present. Return archive energies as a finite NumPy array and retain the raw PAM-SSW result for diagnostics. Do not run PAM-SSW during `ConformerSearch.select`.

- [ ] **Step 4: Run the focused tests and optional local integration test**

Run: `python -m pytest -q tests/test_pamssw_adapter.py`

Expected: PASS for monkeypatched tests; a real backend test is skipped unless `pamssw` is importable.

- [ ] **Step 5: Commit the PAM-SSW adapter**

```bash
git add src/xtb_ase/search/pamssw.py src/xtb_ase/search/__init__.py tests/test_pamssw_adapter.py
git commit -m "feat: add PAM-SSW global search comparison adapter"
```

### Task 8: Document the baseline and add the full verification gate

**Files:**
- Create: `docs/conformer-search-baseline.md`
- Modify: `README.md`
- Test: all existing tests plus new `tests/test_search_*.py` and `tests/test_pamssw_adapter.py`

**Interfaces:**
- Document `sample_langevin_frames`, `ConformerSearch`, `SearchConfig`, optional extras, and `run_pamssw`.
- Document that MACE/TorchSim is an injected descriptor/refiner boundary, not a hard dependency in this release.

- [ ] **Step 1: Write a documentation smoke test**

```python
def test_search_documentation_mentions_explicit_approximation_boundary():
    text = Path("docs/conformer-search-baseline.md").read_text()
    assert "distance-fingerprint" in text
    assert "iRMSD" in text
    assert "PAM-SSW" in text
```

- [ ] **Step 2: Run the documentation test to verify it fails**

Run: `python -m pytest -q tests/test_search_docs.py`

Expected: FAIL because the documentation file does not exist.

- [ ] **Step 3: Add usage documentation**

Document a complete dependency-light example:

```python
from xtb_ase import GFNFF
from xtb_ase.search import ConformerSearch, SearchConfig, sample_langevin_frames

frames = sample_langevin_frames(
    atoms,
    GFNFF(charge=0, threads=4),
    LangevinConfig(temperature_K=600, steps=10_000, sample_interval=20, rng_seed=11),
)
result = ConformerSearch(
    SearchConfig(
        energy_window_kcal_mol=6.0,
        max_selected=64,
        rmsd_backend="irmsd",
    )
).select(frames)
```

Explain that the input frames must be quenched or energy-evaluated consistently before free-energy ranking, that pair-distance fallback is approximate, that MACE pooled features enter through `descriptor_fn`, and that `PAMSSWComparisonConfig.target_uphill_energy_eV` is an explicit comparison parameter rather than an optimized result.

- [ ] **Step 4: Run the complete verification suite and inspect the diff**

Run: `python -m pytest -q`

Expected: all tests pass, optional backend tests skip only when their packages/binaries are absent, and no warnings/errors are emitted beyond declared skips.

Run: `git diff --check && git status --short && git log --oneline --decorate -12`

Expected: no whitespace errors; only the intended branch commits and no unrelated files.

- [ ] **Step 5: Commit the documentation and verified release point**

```bash
git add README.md docs/conformer-search-baseline.md tests/test_search_docs.py
git commit -m "docs: describe conformer search baseline"
```

## Final verification checklist

- [ ] `python -m pytest -q` passes with fresh output.
- [ ] `rmsd_backend="irmsd"` is exact-or-fails-closed; approximate fallback is labeled.
- [ ] The lowest-energy in-window frame is always retained.
- [ ] All result structures are detached copies.
- [ ] g-xTB thermochemistry values are parsed in Hartree and exposed in eV.
- [ ] PAM-SSW uses the explicit small uphill-height parameter and remains optional.
- [ ] No MACE/TorchSim/PLUMED/BO claim is made beyond the injected interface boundary.
- [ ] No files in `/home/gengjianrui/bin/pam-ssw` were modified.
