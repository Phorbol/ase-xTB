# Conformer-search baseline

This document describes the first executable baseline for flexible molecules and
molecular clusters. It is deliberately a trajectory post-processing pipeline:

```text
GFNFF high-temperature Langevin MD
        -> energy window
        -> energy-stratified FPS
        -> optional MACE/TorchSim descriptor
        -> exact iRMSD or explicit approximate deduplication
        -> optional relaxation and g-xTB --hess thermochemistry
```

The implementation is an ASE-native data path. It does not claim that this
baseline is already better than CREST, Molclus, Bayesian optimization, or an
enhanced-sampling method. Those comparisons require equal seeds, force-evaluation
budgets, relaxation settings, and a fixed conformer reference set.

## Install

The search helpers do not make Numba, iRMSD, PAM-SSW, MACE, or TorchSim hard
dependencies:

```bash
python -m pip install -e '.[gfnff,search]'
```

`numba` accelerates FPS when available. `irmsd` is required only for the exact
permutation-aware RMSD deduplication backend. PAM-SSW is an external comparison
package and is loaded only when `run_pamssw` is called.

## Baseline API

`LangevinConfig` controls a reproducible non-periodic ASE trajectory. The frames
are returned detached from the calculator, so evaluate their GFNFF energies as a
separate, explicit step before selection:

```python
import numpy as np
from ase import Atoms

from xtb_ase import GFNFF
from xtb_ase.search import (
    ConformerSearch,
    LangevinConfig,
    SearchConfig,
    sample_langevin_frames,
)

start = Atoms("CCO", positions=initial_positions)
frames = sample_langevin_frames(
    start,
    GFNFF(charge=0, solvent="", threads=8),
    LangevinConfig(
        temperature_K=600.0,
        timestep_fs=0.5,
        friction_per_fs=0.01,
        steps=50_000,
        sample_interval=20,
        rng_seed=20260903,
    ),
)

def gfnff_energy(frame):
    evaluated = frame.copy()
    evaluated.calc = GFNFF(charge=0, solvent="", threads=8)
    return evaluated.get_potential_energy()

energies = np.asarray([gfnff_energy(frame) for frame in frames])  # eV
result = ConformerSearch(
    SearchConfig(
        energy_window_kcal_mol=6.0,
        max_selected=32,
        energy_bins=4,
        fps_backend="auto",
        rmsd_backend="irmsd",
        rmsd_tolerance_angstrom=0.125,
    )
).select(frames, energies=energies)

for group in result.groups:
    print(group.representative_index, group.member_indices)
```

The implementation uses `energy_stratified_fps` for the diversity-selection
stage. The selector returns `ConformerSearchResult` with source indices, detached
representatives, all candidate energies/descriptors, and diagnostics. Energies
are eV internally; `energy_window_kcal_mol` is converted with ASE's unit
constant. The lowest-energy structure inside the window is always retained.

For a dependency-light smoke test, use the explicit approximate
`distance-fingerprint` backend:

```python
config = SearchConfig(rmsd_backend="distance_fingerprint")
```

The pair-distance fingerprint is rigid-transform and same-element permutation
invariant, but it is not a complete graph invariant and is not exact RMSD.
Diagnostics therefore report `dedup_exact=False`. Use `rmsd_backend="irmsd"` for
the final permutation/symmetry-aware deduplication and install the search extra;
missing iRMSD fails closed rather than silently falling back.

## MACE/TorchSim extension point

The baseline does not load a particular MACE-OMOL or POLAR model. A descriptor
callable can be injected without changing selection semantics:

```python
result = ConformerSearch(config).select(
    frames,
    energies=energies,
    descriptor_fn=lambda atoms: mace_pooling_feature(atoms),
)
```

The callable must return one finite one-dimensional vector per structure, with a
consistent length. A future batch relaxer should consume
`result.representatives` and return new structures plus energies; it must be
reported as a separate refinement stage rather than mixed into the GFNFF
coverage statistic.

## PAM-SSW comparison

PAM-SSW is a comparison arm, not part of the baseline. The adapter maps ASE
`Atoms` to PAM-SSW's public `State` and returns detached ASE minima:

```python
from xtb_ase.search import PAMSSWComparisonConfig, run_pamssw

ssw_result = run_pamssw(
    start,
    GFNFF(charge=0, solvent="", threads=8),
    PAMSSWComparisonConfig(
        target_uphill_energy_eV=0.05,  # deliberately small comparison setting
        max_trials=32,
        max_steps_per_walk=8,
        rng_seed=20260903,
    ),
)
```

`target_uphill_energy_eV` is passed to PAM-SSW's
`SSWConfig.target_uphill_energy`. The adapter supports fixed-cell or molecular
ASE states as accepted by PAM-SSW, but does not add variable-cell behavior. The
PAM-SSW checkout is never modified by this package. Record the explicit uphill
height, trial/step limits, calculator settings, and total force calls in any
benchmark.

## g-xTB thermochemistry closeout

After optional relaxation, request thermochemistry from `GXTB` with the g-xTB
Hessian operation:

```python
from xtb_ase import GXTB

calculator = GXTB(
    command="/opt/gxtb/xtb",
    properties=("free_energy", "enthalpy", "zero_point_energy"),
    threads=8,
)
optimized.calc = calculator
free_energy_eV = calculator.get_gibbs_free_energy(optimized)
enthalpy_eV = calculator.get_enthalpy()
zpe_eV = calculator.get_zero_point_energy()
```

The values are parsed from g-xTB's `--hess` output and exposed in eV (or in
Hartree with `unit="hartree"`). This is an RRHO/Hessian thermochemistry value
for the optimized structure; it is not a free-energy estimator for an arbitrary
high-temperature MD snapshot. The returned `thermochemical_correction` is
`total free energy - total electronic energy` when both output fields exist.

## Benchmark boundary and next experiments

The first benchmark should compare direct GFNFF high-temperature MD plus this
pipeline against CREST and the PAM-SSW arm under the same molecule set, initial
structures, random seeds where meaningful, wall-time/CPU-hour and force-call
budgets. Report at least:

1. unique conformers after the same exact deduplication and energy threshold;
2. lowest and best-known-relative energies after the same relaxation level;
3. coverage of a frozen reference set, with the denominator shown;
4. force evaluations, wall time, peak memory, and failure counts;
5. sensitivity to temperature, trajectory length, FPS size, RMSD tolerance, and
   the PAM-SSW uphill height.

Only after this baseline is measured should we add a MACE latent-space arm,
Graph-CV, novelty/extrapolation detection, PLUMED/PySAGES/SPONGE methods, or a
Bayesian optimizer. A global method that spends its budget learning a surrogate
or biasing a CV is justified only if it improves coverage or best-known energy
under the same expensive-calculator budget; “more global” is not itself a
success criterion.
