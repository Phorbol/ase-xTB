# xtb-ase

ASE calculators for the g-xTB v2 executable and the standalone GFN-FF
library.  This package deliberately does not add methods to `tblite-python`:
g-xTB v2 is currently distributed as a modified `xtb` executable, while the
public tblite API does not contain a g-xTB constructor.

## Installation

Install ASE and this package:

```bash
python -m pip install .
```

Install the optional native GFN-FF backend as well:

```bash
python -m pip install '.[gfnff]'
```

Download a matching g-xTB binary from the official g-xTB release and pass its
`xtb` path to `GXTB`.  The binary must accept `--gxtb`; the v2.0.1 Linux
package is based on modified xtb 6.7.1.

## xTB and g-xTB calculator

`XTB` selects the standard GFN0/1/2-xTB Hamiltonian or g-xTB from one ASE
interface.  `GXTB` is the same calculator fixed to `method="gxtb"`:

```python
from xtb_ase import XTB, GXTB

atoms.calc = XTB(command="/opt/xtb/bin/xtb", method="gfn2-xtb")
atoms.calc = GXTB(command="/opt/gxtb/xtb")
```

For `XTB`, `etemp=None` preserves the standard xtb CLI default.  `GXTB`
defaults to 0 K, matching g-xTB v2's default.  Passing `uhf=None` preserves
the executable's automatic spin choice; passing an integer explicitly enables
the unrestricted path.  `spin` and `unpaired_electrons` are Python-friendly
aliases for `uhf`; they mean the number of unpaired electrons, not the spin
multiplicity.

```python
from ase import Atoms
from xtb_ase import GXTB

atoms = Atoms(
    "OH2",
    positions=[[0.0, 0.0, 0.0], [0.7586, 0.0, 0.5043],
               [-0.7586, 0.0, 0.5043]],
)
atoms.calc = GXTB(
    command="/opt/gxtb/xtb",
    charge=0,
    # None preserves g-xTB's automatic closed/open-shell default.
    spin=None,
    threads=8,
    env={"OMP_STACKSIZE": "4G", "OMP_MAX_ACTIVE_LEVELS": "1"},
    keep_files=True,
)

energy = atoms.get_potential_energy()       # eV
forces = atoms.get_forces()                 # eV / Angstrom
charges = atoms.get_charges()               # e
dipole = atoms.get_dipole_moment()          # e * Angstrom
quadrupole = atoms.calc.get_quadrupole()    # e * Angstrom^2
gap = atoms.calc.get_homo_lumo_gap()        # eV
wbo = atoms.calc.get_bond_orders()          # symmetric atom-pair matrix
orbital_e = atoms.calc.get_orbital_energies()  # eV
occupancy = atoms.calc.get_orbital_occupations()
```

`threads` controls the OpenMP/MKL thread count for one xTB child process and
is emitted as `--parallel N`; the older `parallel=N` spelling remains valid.
When `threads=None` (the default), the wrapper omits `--parallel` so inherited
`OMP_NUM_THREADS` and `MKL_NUM_THREADS` settings can take effect.  `env` is a
per-child environment overlay and does not modify Python's global
`os.environ`.  `electronic_temperature` is an alias for `etemp`.

`processes` has a different meaning: it is the number of independent
calculations, not a setting for one ASE Calculator.  Use `CalculatorPool` for
batch work and configure the per-task thread count in the calculator factory:

```python
from functools import partial
from xtb_ase import CalculatorPool, GXTB

pool = CalculatorPool(
    partial(GXTB, command="/opt/gxtb/xtb", threads=4),
    processes=8,
)
results = pool.map(atoms_list, properties=("energy", "forces"))
```

This requests up to `8` independent processes with `4` threads each.  The
factory must be pickleable when `processes > 1`; use a module-level factory or
`functools.partial`, not a lambda.  An ASE optimization, MD, or NEB trajectory
still uses one stateful Calculator rather than the pool.  The pool defaults to
the `spawn` multiprocessing context for native-library safety; pass
`mp_context="fork"` only when that tradeoff is intentional.

The calculator invokes a fresh scratch directory per calculation.  Set
`directory` to choose the scratch root and `keep_files=True` to retain
`structure.xyz`, `structure.engrad`, `charges`, `wbo`, `molden.input`, logs,
and other xTB artifacts.  `get_raw_output()` returns the command, return code,
stdout, and stderr from the last run.

### Supported g-xTB results

| API | Source | Units | Notes |
| --- | --- | --- | --- |
| `get_potential_energy()` | `structure.engrad` | eV | Always part of a normal run |
| `get_forces()` | `structure.engrad` gradient | eV/Å | Forces are the negative gradient |
| `get_charges()` | `charges` | e | Mulliken/charge file emitted by xTB |
| `get_dipole_moment()` | stdout dipole block | eÅ | Requested with `--dipole` |
| `get_quadrupole()` | stdout quadrupole block | eÅ² | Symmetric 3×3 tensor; requested with `--quadrupole` |
| `get_bond_orders()` | `wbo` | dimensionless | Symmetric atom-pair matrix; xTB emits thresholded pairs |
| `get_orbital_energies()` | `molden.input` | eV | Use `unit="hartree"` for Eh |
| `get_orbital_occupations()` | `molden.input` | electrons | Alpha/beta separation is not yet modeled |
| `get_homo_lumo_gap()` | stdout summary | eV | Use `unit="hartree"` for Eh |
| `get_hessian()` | `hessian` | eV/Å² | Numerical Hessian from analytic g-xTB gradients |
| `get_vibrational_frequencies()` | `vibspectrum` | cm⁻¹ | Includes translational/rotational near-zero modes |
| `get_molden_path()` | `molden.input` | path | Requires `keep_files=True` |

### ASE vibrations and thermochemistry

The native Hessian can be exposed through ASE's `VibrationsData` contract:

```python
from xtb_ase import GXTB

atoms.calc = GXTB(command="/opt/gxtb/xtb", threads=8)
vibrations = atoms.calc.get_vibrations_data(atoms)
frequencies_cm1 = vibrations.get_frequencies()
zpe_eV = vibrations.get_zero_point_energy()
```

The generic helper prefers a calculator's `get_hessian(atoms)` method, as in
g-XTB and MACE, and otherwise runs ASE finite differences of `get_forces()`;
the latter is the route for force-only calculators such as GFN-FF:

```python
from xtb_ase import GFNFF, get_vibrations_data

vibrations = get_vibrations_data(
    atoms,
    calculator=GFNFF(charge=0, solvent="", threads=8),
)
```

Both routes are molecular vibration calculations and reject periodic cells in
the finite-difference path.  `VibrationsData.get_frequencies()` returns cm⁻¹
and `get_zero_point_energy()` returns eV.  The finite-difference cache is
temporary unless an explicit `name=` is supplied.

For an ASE `IdealGasThermo` RRHO result, provide the molecular conventions
explicitly:

```python
from ase import units
from xtb_ase import ase_vibrational_thermochemistry

thermo = ase_vibrational_thermochemistry(
    atoms,
    vibrations,
    temperature_K=298.15,
    pressure=units.bar,
    geometry="nonlinear",
    symmetrynumber=1,
    spin=0,
    potential_energy=atoms.get_potential_energy(),
)
thermo.gibbs_free_energy_eV
thermo.enthalpy_eV
thermo.zero_point_energy_eV
thermo.thermochemical_correction_eV
```

This generic result is separate from `GXTB.get_free_energy()` and
`GXTB.get_enthalpy()`: those methods report the native g-XTB `--hess` output,
whereas `ase_vibrational_thermochemistry()` delegates to ASE's
`IdealGasThermo` and reports entropy in eV/K.  Neither route should be used to
call a single high-temperature MD snapshot's electronic-plus-RRHO value a
conformational free energy.

For an xTB-compatible modified-s-rRHO result, use the separate quasi-RRHO
helper.  It uses ASE `MSRRHOThermo` with xTB's 50 cm⁻¹ rotor crossover, treats
imaginary modes in the xTB-compatible 20 cm⁻¹ window, and defaults to a 1 atm
standard pressure.  ASE pressure values are expected, so write 1 atm as
`101325 * units.Pascal`:

```python
from ase import units
from xtb_ase import ase_quasi_rrho_thermochemistry

quasi_thermo = ase_quasi_rrho_thermochemistry(
    atoms,
    vibrations,
    temperature_K=298.15,
    pressure=101325 * units.Pascal,
    geometry="nonlinear",
    symmetrynumber=1,
    spin=0,
    potential_energy=atoms.get_potential_energy(),
    rotor_cutoff_cm1=50.0,
    imaginary_cutoff_cm1=20.0,
    frequency_scale=1.0,
)
quasi_thermo.gibbs_free_energy_eV
```

`frequency_scale`, temperature, pressure, imaginary-mode cutoff, rotor
crossover, electronic spin entropy, and the optional internal-energy
interpolation are all Python parameters.  The default `treat_internal_energy`
is `False`, matching the original xTB-style scheme; set it to `True` only when
the ASE/Otlyotov-Minenkov internal-energy extension is intended.  This helper
does not implement xTB `modef` or one-dimensional DVR scans.  When electronic
entropy is enabled, `spin` uses ASE's spin quantum number (`uhf / 2` for an
xTB unpaired-electron count).

Optional outputs are lazy.  To collect several electronic properties in one
run, configure them up front:

```python
calc = GXTB(
    command="/opt/gxtb/xtb",
    properties=("charges", "dipole", "bond_orders",
                "orbital_energies", "homo_lumo_gap"),
)
```

The wrapper does not claim g-xTB periodic-cell stress, polarizability, orbital
localization, point-charge embedding, or cube generation.  g-xTB solvation is
available only through its limited `gbe`/`cosmo` CLI modes:

```python
calc = GXTB(
    command="/opt/gxtb/xtb",
    solvation_model="cosmo",
    solvent="water",
)
```

The upstream g-xTB documentation warns that the current solvation gradients
need careful evaluation, especially during optimization.

## GFN-FF calculator

`GFNFF` is a thin façade around `gfnff.ase_calculator.GFNFF`; it keeps GFN-FF
optional and exposes the native library's energy, forces, stress, solvent and
periodic-cell behavior:

```python
from xtb_ase import GFNFF

atoms.calc = GFNFF(charge=0, solvent="", printlevel=0, threads=8)
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

The standalone GFN-FF package is maintained separately from tblite and should
be version-pinned when numerical equivalence with an official xtb release is
required.  Its `threads` setting calls the loaded native OpenMP setter and is
process-wide for that native library; use `CalculatorPool` to isolate different
thread configurations.  `env` can configure the backend during initialization,
but it is also process-local rather than instance-isolated.  GFN-FF supports
charge, solvent, fragments, and reference charges, but rejects xTB electronic
parameters such as `spin`, `uhf`, and `etemp` because it is a force field.

## Validation

Run parser/unit tests without an external backend:

```bash
python -m pytest -q
```

Run g-xTB integration tests with the checked binary (or set
`GXTB_COMMAND`):

```bash
python -m pytest -q -m integration
```

Run GFN-FF integration tests after installing its optional wheel:

```bash
PYTHONPATH=/path/to/gfnff/site-packages:src python -m pytest -q -m integration
```

## Conformer-search baseline

The optional `xtb_ase.search` package provides a reproducible non-periodic
GFNFF high-temperature Langevin trajectory, energy-windowed NumPy/Numba FPS,
and conformer deduplication. Install it with:

```bash
python -m pip install -e '.[gfnff,search]'
```

The exact permutation/symmetry-aware `iRMSD` backend is optional;
`rmsd_backend="distance_fingerprint"` is an explicitly approximate fallback
for smoke tests.

```python
from xtb_ase import GFNFF
from xtb_ase.search import (
    ConformerSearch,
    LangevinConfig,
    SearchConfig,
    sample_langevin_frames,
)

frames = sample_langevin_frames(
    start_atoms,
    GFNFF(charge=0, solvent="", threads=8),
    LangevinConfig(temperature_K=600, steps=50_000, sample_interval=20,
                   rng_seed=20260903),
)
# Evaluate detached frames explicitly with GFNFF and pass one eV value per frame.
result = ConformerSearch(SearchConfig(
    energy_window_kcal_mol=6.0,
    max_selected=32,
    rmsd_backend="irmsd",
)).select(frames, energies=energies_eV)
```

The result retains source indices and diagnostics. A small-uphill PAM-SSW
comparison arm is available through `run_pamssw` and defaults to
`target_uphill_energy_eV=0.05`; it is not part of the baseline and does not
modify the separate `pam-ssw` checkout. See
[`docs/conformer-search-baseline.md`](docs/conformer-search-baseline.md) for
the full pipeline, MACE/TorchSim descriptor injection point, g-xTB Hessian
thermochemistry closeout, and fair-budget benchmark protocol.
