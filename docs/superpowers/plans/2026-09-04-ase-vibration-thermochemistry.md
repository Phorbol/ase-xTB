# ASE Vibration and Thermochemistry Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing g-XTB Hessian results and any ASE calculator's force surface usable through ASE `VibrationsData` and a shared ASE thermochemistry API.

**Architecture:** Add a small `xtb_ase.vibrations` module with two layers: a Hessian-to-`VibrationsData` adapter for calculators such as g-XTB and MACE, and a finite-difference `Vibrations` runner for calculators such as GFNFF that provide forces only. Add a separate thermochemistry helper that delegates the statistical-mechanics calculation to ASE `IdealGasThermo`, returning named eV/eV/K values without changing the existing g-XTB-native RRHO properties.

**Tech Stack:** Python 3.10+, ASE `Vibrations`, `VibrationsData`, `IdealGasThermo`, NumPy, pytest.

**Spec:** The approved design in the preceding conversation: native g-XTB Hessian conversion, standard ASE finite-difference vibration compatibility, and an ASE-compatible free-energy/enthalpy/ZPE/thermochemical-correction result for MACE/GFNFF/xTB force providers.

## Global Constraints

- Preserve the existing `GXTB.get_free_energy()` and related native g-XTB output semantics.
- Use ASE energy units: eV for energies, eV/K for entropy, Å for coordinates, and eV/Å² for Hessians.
- Never call an electronic-energy-plus-RRHO correction a free energy of an arbitrary MD snapshot.
- `VibrationsData` is a data adapter; it must not silently run finite differences or mutate the caller's `Atoms` object.
- The finite-difference helper must use a private cache directory by default and remove it after assembling `VibrationsData`.
- Thermochemistry requires explicit molecular `geometry`, `symmetrynumber`, and `spin` inputs when Gibbs free energy is requested; no unsupported periodic thermochemistry is claimed.
- Run focused tests first, then the complete suite and the existing g-XTB integration tests.

---

### Task 1: Add failing tests for the public vibration and thermochemistry API

**Files:**
- Create: `tests/test_vibrations.py`
- Modify: `tests/test_gxtb_calculator.py`
- Modify: `tests/test_public_api.py`

**Interfaces under test:**
- `xtb_ase.vibrations.hessian_to_vibrations_data(atoms, hessian, indices=None)`
- `xtb_ase.vibrations.run_vibrations(atoms, calculator=None, ...)`
- `xtb_ase.vibrations.ase_vibrational_thermochemistry(...)`
- `GXTB.get_vibrations_data(atoms=None, indices=None)`

- [x] **Step 1: Write tests that import the new functions and assert behavior**

Cover these real behaviors:

```python
def test_hessian_adapter_returns_vibrations_data_and_zpe():
    data = hessian_to_vibrations_data(atoms, diagonal_hessian)
    assert data.get_hessian_2d().shape == (3 * len(atoms), 3 * len(atoms))
    assert np.isfinite(data.get_frequencies()).all()
    assert data.get_zero_point_energy() > 0.0


def test_run_vibrations_uses_any_ase_force_calculator_without_mutation():
    original = atoms.copy()
    data = run_vibrations(
        atoms,
        calculator=harmonic_calculator,
        delta=0.005,
        name=None,
    )
    np.testing.assert_allclose(atoms.positions, original.positions)
    assert data.get_atoms().calc is None


def test_ase_thermochemistry_matches_ideal_gas_thermo():
    result = ase_vibrational_thermochemistry(
        atoms,
        data,
        temperature_K=298.15,
        geometry="nonlinear",
        symmetrynumber=1,
        spin=0,
    )
    assert result.gibbs_free_energy_eV == pytest.approx(expected_gibbs)
    assert result.thermochemical_correction_eV == pytest.approx(
        result.gibbs_free_energy_eV - result.potential_energy_eV
    )
```

Use a small analytic ASE calculator or a positive diagonal Hessian for deterministic tests; do not require MACE, GFNFF, or g-XTB to be installed for unit tests. Add a fake g-XTB Hessian test proving `GXTB.get_vibrations_data()` produces the same `VibrationsData` contract as MACE's `get_hessian()` output.

- [x] **Step 2: Run the focused tests and verify the expected missing-API failures**

Run:

```bash
python -m pytest -q tests/test_vibrations.py tests/test_gxtb_calculator.py::test_gxtb_returns_vibrations_data tests/test_public_api.py
```

Expected: failure because the new module/helper and `GXTB` method do not yet exist.

---

### Task 2: Implement Hessian conversion and calculator-backed finite differences

**Files:**
- Create: `src/xtb_ase/vibrations.py`
- Modify: `src/xtb_ase/gxtb.py`

**Interfaces:**
- `hessian_to_vibrations_data(atoms, hessian, indices=None) -> VibrationsData`
- `run_vibrations(atoms, calculator=None, indices=None, delta=0.01, nfree=2, name=None) -> VibrationsData`
- `GXTB.get_vibrations_data(atoms=None, indices=None) -> VibrationsData`

- [x] **Step 1: Implement a version-compatible `VibrationsData` loader**

Import `ase.vibrations.data.VibrationsData` on current ASE and fall back to the older `ase.vibrationsdata.VibrationsData` location. Convert a full `(3N, 3N)` Hessian to the active-index submatrix when `indices` is supplied, validate finite shape and non-empty indices, and call `VibrationsData.from_2d` when available. Preserve the equilibrium `Atoms` as an internal copy and never attach a calculator to the returned data.

- [x] **Step 2: Implement the finite-difference runner**

Copy the input `Atoms`, attach the supplied calculator or its existing calculator, construct ASE `Vibrations`, and run it. If `name` is omitted, use a `TemporaryDirectory` cache path; assemble `VibrationsData` before cleanup. Reject periodic atoms for this molecular helper, invalid `delta`, and calculators with no `get_forces` capability. Ensure the caller's positions, constraints, calculator, and info remain unchanged.

- [x] **Step 3: Add the native g-XTB method**

Add `GXTB.get_vibrations_data()` next to `get_hessian()`. Call `get_hessian(atoms)`, then pass the resulting eV/Å² matrix to `hessian_to_vibrations_data`. Support an active `indices` subset without changing the underlying g-XTB Hessian output. Keep `get_hessian()` and `get_vibrational_frequencies()` backward-compatible.

- [x] **Step 4: Run focused tests and refactor only after green**

Run:

```bash
python -m pytest -q tests/test_vibrations.py tests/test_gxtb_calculator.py tests/test_public_api.py
```

---

### Task 3: Implement generic ASE thermochemistry results

**Files:**
- Modify: `src/xtb_ase/vibrations.py`
- Modify: `src/xtb_ase/__init__.py`
- Modify: `tests/test_vibrations.py`

**Interfaces:**
- `ASEVibrationalThermochemistry`
- `ase_vibrational_thermochemistry(atoms, vibrations, temperature_K=298.15, pressure=units.bar, geometry=..., symmetrynumber=..., spin=..., potential_energy=None, vib_selection="highest", ignore_imag_modes=False) -> ASEVibrationalThermochemistry`

- [x] **Step 1: Implement the result dataclass and ASE delegation**

Extract complex vibrational energies in eV from `VibrationsData.get_energies()`. Instantiate `ase.thermochemistry.IdealGasThermo` with the supplied geometry, atoms, symmetry number, spin, potential energy, vibration-selection mode, and imaginary-mode policy. Evaluate `get_zero_point_energy()`, `get_enthalpy(temperature_K)`, `get_entropy(temperature_K, pressure)`, and `get_gibbs_energy(temperature_K, pressure)`. Return a frozen result with:

```text
potential_energy_eV
zero_point_energy_eV
enthalpy_eV
entropy_eV_per_K
gibbs_free_energy_eV
free_energy_eV  # alias value for callers using the existing vocabulary
thermochemical_correction_eV  # Gibbs - potential energy
temperature_K
pressure_Pa
```

Require `geometry`, `symmetrynumber`, and `spin` explicitly in the public helper unless the caller requests only ZPE; raise a clear `ValueError` for invalid temperature/pressure and `RuntimeError` from ASE when Gibbs inputs are incomplete. Do not add a fake `entropy` field to the native g-XTB CLI parser.

- [x] **Step 2: Test exact delegation and units**

Compare the helper's values to direct `IdealGasThermo` calls on the same `VibrationsData.get_energies()` and assert the correction identity, eV/K entropy units, Hartree-independent behavior, and missing-potential-energy errors. Test a MACE-style calculator by feeding its Hessian-shaped array into `hessian_to_vibrations_data`; no MACE import is needed.

- [x] **Step 3: Export the stable public symbols and document the distinction**

Export the adapter/helper/result from `xtb_ase`, add a README section and extend `docs/conformer-search-baseline.md` so that native g-XTB RRHO output and generic ASE `IdealGasThermo` output are separately named and not directly mixed in benchmark tables.

---

### Task 4: Verification and integration evidence

**Files:**
- Modify: `tests/test_gxtb_integration.py`
- Modify: `tests/test_vibrations.py`
- Modify: `README.md`
- Modify: `docs/conformer-search-baseline.md`

- [x] **Step 1: Add optional real-backend smoke coverage**

When the checked g-XTB binary exists, call `GXTB.get_vibrations_data()` and compare its mode count, finite frequencies, and ZPE to the existing native Hessian path. Keep this test skipped when the binary is absent. Run `run_vibrations()` with the installed GFNFF backend only when its optional package is available.

- [x] **Step 2: Run all verification commands**

```bash
python -m pytest -q
PYTHONPATH=src:/home/gengjianrui/bin/pam-ssw python -m pytest -q
git diff --check
git status --short
git -C /home/gengjianrui/bin/pam-ssw status --short
```

The baseline suite must remain green; optional backends may skip only under their existing explicit conditions. Report that finite-difference vibration values depend on calculator forces and that native g-XTB RRHO values are a separate calculation path.

- [ ] **Step 3: Commit the extension as a separate verified change**

```bash
git add src/xtb_ase/vibrations.py src/xtb_ase/gxtb.py src/xtb_ase/__init__.py tests/test_vibrations.py tests/test_gxtb_calculator.py tests/test_gxtb_integration.py tests/test_public_api.py README.md docs/conformer-search-baseline.md docs/superpowers/plans/2026-09-04-ase-vibration-thermochemistry.md
git commit -m "feat: integrate ASE vibrations and thermochemistry"
```

## Evidence boundary

Green API tests prove the adapter contract and unit conversion only. They do not prove that a MACE model, GFNFF, or g-XTB is scientifically accurate, nor that finite-difference and analytic Hessians agree beyond the tested calculator. Coverage, convergence, and free-energy ranking remain separate benchmark questions.
