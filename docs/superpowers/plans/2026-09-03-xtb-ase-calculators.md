# xTB and g-xTB ASE Calculators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide an isolated Python package with ASE-compatible standard xTB/g-xTB subprocess calculators and a GFN-FF adapter, exposing stable high-frequency electronic-structure results without changing tblite's public method set.

**Architecture:** `XTB` owns an isolated per-calculation scratch directory, selects standard GFN0/1/2-xTB or g-xTB, and parses structured output files plus narrowly scoped stdout blocks; `GXTB` fixes that class to g-xTB. `GFNFF` delegates to the standalone `gfnff` library when installed. Electronic properties are split into ASE-native results (`energy`, `forces`, `charges`, `dipole`) and explicit calculator methods for bond orders, orbitals, Hessians, vibrational data, and raw artifacts.

**Tech Stack:** Python 3.10+, ASE 3.29+, NumPy, subprocess, pathlib, pytest; optional `gfnff[ase]`.

**Spec:** Confirmed user design: exact g-xTB v2 binary wrapper plus GFN-FF standalone-library adapter; include high-frequency electronic-structure interfaces discovered from `xtb --help` and validate against direct CLI output.

## Global Constraints

- Do not modify or pretend to extend `tblite-python`; its built-in method list remains unchanged.
- Standard xTB calculations use `--gfn 0/1/2`; g-xTB calculations use `--gxtb`; GFN-FF calculations use the standalone `gfnff` adapter unless an explicit subprocess fallback is requested.
- Energy is returned in eV, forces in eV/Å, charges in e, and dipoles in eÅ as required by ASE.
- Every g-xTB calculation uses a unique scratch directory and fails closed on missing or malformed required output.
- `energy`, `forces`, `charges`, and `dipole` are ASE properties; optional expensive results are requested by explicit methods and are never silently approximated.
- `quadrupole` is exposed as a symmetric 3x3 tensor through `get_quadrupole()` because ASE has no standard atom method for it.
- No PBC/stress claim for g-xTB until a direct binary test proves the behavior; GFN-FF PBC/stress follows the dependency's tested API.
- g-xTB solvation is limited to explicit `gbe`/`cosmo` selection and is documented as gradient/optimization caution; unsupported CLI features are rejected or left as raw `extra_args` rather than silently mapped.

---

### Task 1: Package contract and parser tests

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/xtb_ase/__init__.py`
- Create: `src/xtb_ase/_parsers.py`
- Test: `tests/test_parsers.py`

**Interfaces:**
- `parse_engrad(path, natoms) -> ParsedGradient`
- `parse_charges(path, natoms) -> np.ndarray`
- `parse_stdout_properties(text, natoms) -> ParsedProperties`
- `parse_wbo(path, natoms) -> np.ndarray`
- `parse_molden(path) -> ParsedOrbitals`
- `parse_hessian(path, natoms) -> np.ndarray`
- `parse_vibspectrum(path) -> np.ndarray`

- [ ] **Step 1: Write failing parser tests** for `.engrad`, charge vectors, dipole/HOMO-LUMO output, sparse WBO pair output, Molden orbital records, Hessian blocks, and vibrational frequencies.
- [ ] **Step 2: Run `pytest tests/test_parsers.py -q` and verify the failures are missing parser symbols or behavior, not fixture errors.**
- [ ] **Step 3: Implement only the parsers needed by the failing tests, with explicit atom-count and numeric-field validation.**
- [ ] **Step 4: Re-run parser tests and confirm green output.**
- [ ] **Step 5: Add README property/unit documentation and commit the parser contract.**

### Task 2: GXTB ASE calculator core

**Files:**
- Create: `src/xtb_ase/gxtb.py`
- Test: `tests/test_gxtb_calculator.py`
- Test: `tests/test_gxtb_integration.py`

**Interfaces:**
- `XTB(command, method="gfn2-xtb", charge=0, uhf=None, accuracy=1.0, etemp=0.0, solvation_model=None, solvent=None, directory=None, keep_files=False, timeout=None, parallel=1, properties=None, extra_args=())`
- `GXTB(command, ...)` is a convenience subclass fixed to `method="gxtb"`.
- ASE properties: `energy`, `forces`, `charges`, `dipole`
- Additional ASE result: `quadrupole` (retrieved by `get_quadrupole()`)
- Explicit methods: `get_bond_orders()`, `get_orbital_energies()`, `get_orbital_occupations()`, `get_homo_lumo_gap()`, `get_hessian()`, `get_vibrational_frequencies()`, `get_molden_path()`, `get_run_directory()`, `get_raw_output()`

- [ ] **Step 1: Write failing tests** for command construction, XYZ unit conversion, energy/gradient conversion, ASE `get_potential_energy/get_forces/get_charges/get_dipole_moment`, and a failed-process/missing-file error.
- [ ] **Step 2: Run the focused tests and verify they fail because `GXTB` is absent.**
- [ ] **Step 3: Implement the minimal calculator using a unique scratch directory, `--gxtb --grad`, `.engrad`, and optional output flags only when requested.**
- [ ] **Step 4: Run focused tests and confirm green; add cache invalidation when positions, numbers, charge, spin, or cell state changes.**
- [ ] **Step 5: Add direct-binary integration tests for water energy, forces, charges, dipole, WBO, orbitals, Hessian, and frequencies.**
- [ ] **Step 6: Run integration tests with the pinned g-xTB binary and document unsupported flags such as polarizability, localization, point-charge embedding, and cube generation.**

### Task 3: GXTB electronic-property API and artifacts

**Files:**
- Modify: `src/xtb_ase/gxtb.py`
- Modify: `src/xtb_ase/_parsers.py`
- Test: `tests/test_electronic_properties.py`

**Interfaces:**
- `get_bond_orders() -> np.ndarray` returns a symmetric atom-pair matrix for pairs emitted by the CLI; absent pairs are zero and thresholding is documented.
- `get_orbital_energies(unit="eV") -> np.ndarray` and `get_orbital_occupations() -> np.ndarray` return Molden/CLI orbitals.
- `get_homo_lumo_gap(unit="eV") -> float` returns the parsed CLI gap.
- `get_quadrupole(unit="eA2") -> np.ndarray` returns the parsed symmetric molecular quadrupole tensor.
- `get_hessian() -> np.ndarray` returns the non-mass-weighted Hessian in eV/Å² after conversion from Eh/Bohr².
- `get_vibrational_frequencies() -> np.ndarray` returns cm⁻¹ values from `vibspectrum`.
- `get_molden_path() -> pathlib.Path` returns a retained artifact path only after `--molden` was requested.

- [ ] **Step 1: Write failing tests** for lazy optional flags, unit conversions, explicit `PropertyNotImplementedError` behavior, and raw artifact retention.
- [ ] **Step 2: Run tests and verify the expected failures.**
- [ ] **Step 3: Implement lazy result collection and explicit property methods; do not expose `alpha` unless the binary emits a parseable polarizability block.**
- [ ] **Step 4: Run the electronic-property tests and confirm green.**
- [ ] **Step 5: Test two independent calculators concurrently and verify no cross-run file reuse.**

### Task 4: GFN-FF adapter

**Files:**
- Create: `src/xtb_ase/gfnff.py`
- Modify: `src/xtb_ase/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/test_gfnff_adapter.py`

**Interfaces:**
- `GFNFF(**parameters)` is an ASE calculator façade that imports `gfnff.ase.GFNFF` lazily and forwards `charge`, `solvent`, `printlevel`, and supported PBC/stress parameters.
- Expose `energy`, `forces`, and `stress`; raise an actionable optional-dependency error if `gfnff` is absent.

- [ ] **Step 1: Write failing adapter tests** using a real or minimal dependency probe, checking import errors and parameter forwarding.
- [ ] **Step 2: Run the tests and verify they fail because the adapter does not exist.**
- [ ] **Step 3: Implement lazy delegation without copying or reimplementing GFN-FF internals.**
- [ ] **Step 4: Run adapter tests and, when the wheel is available, compare energy/forces/stress against direct official xtb GFN-FF output for a small molecule and periodic cell.**

### Task 5: Documentation, packaging, and final verification

**Files:**
- Modify: `README.md`
- Create: `tests/test_public_api.py`

- [ ] **Step 1: Write failing public-API tests** for imports, version-independent calculator construction, and documented unit/property names.
- [ ] **Step 2: Run the public-API tests and verify the expected failures.**
- [ ] **Step 3: Complete packaging metadata and usage examples for g-xTB and optional GFN-FF.**
- [ ] **Step 4: Run the complete test suite, integration tests, compile checks, and a clean install into a temporary virtual environment.**
- [ ] **Step 5: Inspect `git diff`, record exact binary/dependency versions and test counts, then request review before declaring completion.**
