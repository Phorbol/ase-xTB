# xTB/g-xTB electronic-property matrix

The matrix below was checked against the pinned Linux g-xTB v2.0.1 binary
(`xtb 6.7.1`, build `26dd68d`) on 2026-09-03.  Standard xTB uses the same
`XTB` wrapper with `method="gfn0"`, `"gfn1"`, or `"gfn2"`; `GXTB` fixes the
method to `"gxtb"`.

| CLI feature | Output observed | API status |
| --- | --- | --- |
| `--grad` | `<label>.engrad` | `energy`, `forces` |
| `--pop` | `charges` | `Atoms.get_charges()` |
| `--dipole` | g-xTB atomic-unit block; standard-xTB molecular `full` block | `Atoms.get_dipole_moment()` |
| `--quadrupole` | g-xTB atomic-unit total; standard-xTB molecular `full` block | `get_quadrupole()` as symmetric 3x3 eÅ² tensor |
| `--wbo` | atom-pair `wbo` list | `get_bond_orders()` / `get_wbo()` |
| `--molden` | `molden.input` with basis and MO records | orbital energies/occupations plus retained file path |
| `--hess` | `hessian`, `vibspectrum` | `get_hessian()`, `get_vibrational_frequencies()` |
| `--chrg` | charge used by the Hamiltonian | `charge=` |
| `--uhf` | number of unpaired electrons | `uhf=`, `spin=`, `unpaired_electrons=` |
| `--etemp` | electronic temperature | `etemp=`, `electronic_temperature=` |
| `--gbe` / `--cosmo` | implicit-solvent calculation | `solvation_model=`, `solvent=` |
| `--parallel` | OpenMP/MKL threads for one invocation | `threads=` or legacy `parallel=` |
| `--alpha` | no parseable polarizability block in pinned g-xTB run | intentionally not exposed |
| `--esp` | no stable g-xTB grid artifact in pinned run | intentionally not exposed |
| `--stm` | no stable g-xTB STM artifact in pinned run | intentionally not exposed |
| `--lmo` | listed as a g-xTB limitation | intentionally not exposed |
| `--fod` | no g-xTB result contract established | intentionally not exposed |
| `--ceh` | separate `ceh.charges` calculation, no normal energy/gradient | not part of the base calculator contract |

## Unit contract

- xTB Hartree → ASE eV: `units.Hartree`
- xTB Eh/Bohr gradient → ASE eV/Å: `units.Hartree / units.Bohr`, with a minus sign for forces
- atomic-unit dipole → ASE eÅ: `units.Bohr`
- Debye dipole → ASE eÅ: `units.Debye`
- Eh/Bohr² Hessian → ASE eV/Å²: `units.Hartree / units.Bohr**2`

Optional outputs are lazy.  Configure a tuple through `GXTB(properties=...)`
when several properties should be collected from one SCF run; otherwise an
explicit property method starts the smallest compatible additional CLI run.

`get_vibrations_data()` converts the eV/Å² Hessian to ASE `VibrationsData`.
The shared `get_vibrations_data(atoms, calculator=...)` helper prefers an
analytic `get_hessian()` method (g-XTB/MACE) and falls back to ASE force finite
differences for force-only calculators such as GFN-FF.  The companion
`ase_vibrational_thermochemistry()` delegates to ASE `IdealGasThermo`; its
Gibbs/enthalpy/ZPE/correction values are a separate generic ASE route from
the native g-XTB thermochemistry properties above.

`threads` is a per-invocation setting.  `processes` belongs to the public
`CalculatorPool` for independent structures, where a pickleable calculator
factory creates one isolated Calculator per worker.  The resource product is
approximately `processes * threads`; the pool does not change the semantics of
stateful ASE trajectories.

## Boundaries

`GXTB`/`XTB` currently fail closed for periodic cells and do not claim stress.
`GFNFF` delegates to the standalone `gfnff.ase_calculator.GFNFF`, which has a
separate native contract for energy, forces, stress, solvent and PBC.
GFN-FF's OpenMP thread setter is process-wide, so different GFN-FF thread
counts should be placed in separate pool workers.
