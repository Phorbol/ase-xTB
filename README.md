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
the unrestricted path.

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
    uhf=None,
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

atoms.calc = GFNFF(charge=0, solvent="", printlevel=0)
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

The standalone GFN-FF package is maintained separately from tblite and should
be version-pinned when numerical equivalence with an official xtb release is
required.

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
