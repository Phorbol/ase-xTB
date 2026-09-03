# Python Resource and Electronic-Parameter Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Python-controlled thread/environment/electronic-parameter configuration for xTB-family ASE calculators and a process pool for independent calculations.

**Architecture:** Keep `threads` as a per-child xTB/gXTB setting and retain `parallel` as a compatibility alias. Add a small standard-library `CalculatorPool` whose pickleable factory creates isolated calculators in independent worker processes; expose GFN-FF's process-wide OpenMP setter with documentation that worker isolation is required for concurrent configurations.

**Tech Stack:** Python 3.10+, ASE, NumPy, `subprocess`, `ctypes`, `concurrent.futures.ProcessPoolExecutor`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-python-resource-configuration.md`

## Global Constraints

- Preserve the optional nature of the `gfnff` dependency.
- Do not mutate the parent process environment for xTB/gXTB child-process configuration.
- Preserve `parallel`, `uhf`, and `etemp` as backward-compatible spellings.
- Do not claim that xTB `--parallel` is an independent MPI/process count.
- Keep GFN-FF electronic-only parameters rejected rather than silently ignored.
- Use only standard-library additions; do not add a runtime dependency for pooling.

---

### Task 1: Add xTB/gXTB resource and parameter aliases

**Files:**
- Modify: `src/xtb_ase/gxtb.py`
- Test: `tests/test_gxtb_calculator.py`
- Test: `tests/test_public_api.py`

**Interfaces:**
- Consumes: existing `XTB.__init__`, `XTB.set`, and `_build_command` behavior.
- Produces: `XTB(..., threads=None, parallel=None, env=None, spin=None, unpaired_electrons=None, electronic_temperature=None)` and the same aliases through `set()`.

- [ ] **Step 1: Write failing argument/environment tests**

Extend the fake executable so it writes selected environment values, then add tests equivalent to:

```python
def test_xtb_python_resource_and_electronic_aliases(tmp_path):
    executable = make_fake_xtb(tmp_path / "fake-xtb", record_environment=True)
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calc = GXTB(
        command=str(executable),
        directory=tmp_path,
        keep_files=True,
        threads=4,
        spin=2,
        electronic_temperature=300.0,
        env={"OMP_STACKSIZE": "8G", "XTBASE_TEST_ENV": "child-only"},
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    args = (calc.get_run_directory() / "args.txt").read_text().splitlines()
    assert args[args.index("--parallel") + 1] == "4"
    assert args[args.index("--uhf") + 1] == "2"
    assert args[args.index("--etemp") + 1] == "300"
    assert (calc.get_run_directory() / "env.txt").read_text().strip() == "child-only"
```

Also add tests that `threads=None` omits `--parallel`, `parallel=3` remains valid, `set(threads=2)` updates the next command, and conflicting aliases raise `ValueError` before execution.

- [ ] **Step 2: Run the focused tests and verify the expected RED failures**

Run:

```bash
python -m pytest -q tests/test_gxtb_calculator.py tests/test_public_api.py
```

Expected: the new tests fail because the constructor has no canonical `threads`/alias resolution and the child environment is not forwarded.

- [ ] **Step 3: Implement canonical alias resolution and environment merging**

Add typed mapping support and helpers in `gxtb.py` that resolve each alias group, validate positive/non-negative values, normalize `env` to a serializable mapping, and mirror canonical values into `self.parameters` for ASE inspection. Change the constructor default to `threads=None`, preserve `parallel` as a mirrored alias, pass `env=merged_environment` to `subprocess.run`, and append `--parallel` only when `parameters.threads` is not `None`.

Normalize aliases before `super().set(**kwargs)` so `set(threads=4)`, `set(parallel=4)`, `set(spin=1)`, and `set(electronic_temperature=300)` invalidate cached results and update the canonical parameter. Keep `GXTB`'s 0 K default only when neither `etemp` nor `electronic_temperature` was supplied.

- [ ] **Step 4: Run the focused tests and existing unit suite**

Run:

```bash
python -m pytest -q tests/test_gxtb_calculator.py tests/test_public_api.py
python -m pytest -q
```

Expected: all xTB unit tests pass and the full suite has no new failures.

- [ ] **Step 5: Commit the xTB configuration slice**

```bash
git add src/xtb_ase/gxtb.py tests/test_gxtb_calculator.py tests/test_public_api.py
git commit -m "feat: expose Python xTB resource configuration"
```

### Task 2: Add GFN-FF thread/environment configuration and guards

**Files:**
- Modify: `src/xtb_ase/gfnff.py`
- Test: `tests/test_gfnff_adapter.py`

**Interfaces:**
- Consumes: the optional standalone `gfnff.ase_calculator.GFNFF` backend and Task 1 environment normalization semantics.
- Produces: `GFNFF(..., threads=None, env=None)` with explicit rejection of electronic-only parameters.

- [ ] **Step 1: Write failing GFN-FF configuration tests**

Add tests that construct `GFNFF(threads=4)` and verify a monkeypatched native-thread helper receives 4 before backend creation, that `GFNFF(uhf=1)`, `GFNFF(spin=1)`, and `GFNFF(etemp=300)` raise `TypeError`, and that non-positive thread counts raise `ValueError`.

- [ ] **Step 2: Run the GFN-FF focused tests to verify RED**

```bash
python -m pytest -q tests/test_gfnff_adapter.py
```

Expected: new constructor arguments are currently forwarded as unknown ASE parameters or silently accepted, so the new assertions fail.

- [ ] **Step 3: Implement process-local GFN-FF configuration**

Normalize `threads` and `env` in the façade. Apply `env` with a small context manager while importing/initializing the optional backend and restore the parent environment in `finally`. After the backend is loaded, call `omp_set_num_threads(int(threads))` through the backend's loaded ctypes handle when requested; raise an actionable configuration error if the setter is unavailable. Apply the setter immediately before the native singlepoint call so sequential calculators can change the process-wide setting. Preserve charge/solvent precedence and cache invalidation.

- [ ] **Step 4: Run GFN-FF tests with and without the optional backend**

```bash
python -m pytest -q tests/test_gfnff_adapter.py
PYTHONPATH=vendor/site-packages:src python -m pytest -q tests/test_gfnff_adapter.py
```

Expected: missing-dependency tests remain actionable; optional-backend integration tests pass when the vendored backend is available.

- [ ] **Step 5: Commit the GFN-FF configuration slice**

```bash
git add src/xtb_ase/gfnff.py tests/test_gfnff_adapter.py
git commit -m "feat: configure GFN-FF OpenMP threads from Python"
```

### Task 3: Add independent-process `CalculatorPool`

**Files:**
- Create: `src/xtb_ase/pool.py`
- Modify: `src/xtb_ase/__init__.py`
- Create: `tests/test_pool.py`
- Test: `tests/test_public_api.py`

**Interfaces:**
- Consumes: pickleable calculator factories such as `functools.partial(GXTB, command=..., threads=4)`.
- Produces: `CalculatorPool(calculator_factory, processes=1, mp_context=None).map(atoms_iterable, properties=("energy",)) -> list[dict[str, Any]]`.

- [ ] **Step 1: Write failing pool tests**

Use a top-level `functools.partial` factory around the existing fake xTB executable and add tests equivalent to:

```python
def test_pool_preserves_order_and_returns_requested_results(tmp_path):
    executable = make_fake_xtb(tmp_path / "fake-xtb")
    factory = partial(GXTB, command=str(executable), directory=tmp_path, threads=1)
    atoms_list = [Atoms("H", positions=[[0.0, 0.0, 0.0]]), Atoms("H", positions=[[0.7, 0.0, 0.0]])]
    results = CalculatorPool(factory, processes=2).map(atoms_list, properties=("energy", "forces"))
    assert [result["energy"] for result in results] == pytest.approx([-2.0 * units.Hartree] * 2)
    assert all(result["forces"].shape == (1, 3) for result in results)
```

Add validation tests for `processes=0`, empty property lists, and a non-pickleable lambda factory when `processes=2`.

- [ ] **Step 2: Run the pool tests to verify RED**

```bash
python -m pytest -q tests/test_pool.py tests/test_public_api.py
```

Expected: import and constructor failures because `CalculatorPool` does not yet exist.

- [ ] **Step 3: Implement ordered serial/process execution**

Implement a top-level worker that copies each `Atoms`, constructs one calculator from the factory, calls `calculate(..., system_changes=all_changes)`, and returns copied requested entries from `calculator.results`. Use a serial path for `processes=1`; use `ProcessPoolExecutor` with optional `multiprocessing.get_context(mp_context)` and `executor.map` for `processes>1`. Validate positive integer process counts, normalize non-empty property tuples, and raise an actionable `TypeError` for an unpickleable multi-process factory.

- [ ] **Step 4: Run pool tests and the complete unit suite**

```bash
python -m pytest -q tests/test_pool.py tests/test_public_api.py
python -m pytest -q
```

Expected: ordered two-worker results and all prior unit tests pass.

- [ ] **Step 5: Commit the process-pool slice**

```bash
git add src/xtb_ase/pool.py src/xtb_ase/__init__.py tests/test_pool.py tests/test_public_api.py
git commit -m "feat: add process pool for independent ASE calculations"
```

### Task 4: Document the public contract and update integration coverage

**Files:**
- Modify: `README.md`
- Modify: `docs/electronic-property-matrix.md`
- Modify: `tests/test_gxtb_integration.py`
- Modify: `tests/test_electronic_properties.py`
- Modify: `tests/test_gfnff_adapter.py`

**Interfaces:**
- Consumes: completed calculator and pool APIs from Tasks 1–3.
- Produces: user-facing examples distinguishing `processes`, `threads`, OpenMP environment, charge, unpaired-electron count, and solvation.

- [ ] **Step 1: Write/update regression expectations for explicit single-thread comparisons**

Change direct-vs-wrapper integration calculators to use `threads=1` as the stable comparison setting while retaining one test that verifies an omitted `threads` value omits `--parallel` in the fake executable.

- [ ] **Step 2: Run integration tests and record backend availability**

```bash
python -m pytest -q -m integration
```

Expected: real xTB/GFN-FF tests pass when their binaries/dependencies are present and otherwise are reported as skips by the existing guards.

- [ ] **Step 3: Update README and property matrix**

Document constructor examples for `threads`, `parallel`, `env`, `charge`, `spin` as unpaired-electron count, `electronic_temperature`, `solvation_model`, and `solvent`. Add a `CalculatorPool` example using `functools.partial`, explain `processes × threads`, and state that GFN-FF thread configuration is process-wide unless worker-isolated.

- [ ] **Step 4: Run package-wide verification**

```bash
python -m pytest -q
python -m compileall -q src tests
python -m build --wheel --outdir /tmp/xtb-ase-wheel
```

Expected: tests and compilation pass; the wheel is created without adding a hard GFN-FF dependency.

- [ ] **Step 5: Commit documentation and regression updates**

```bash
git add README.md docs/electronic-property-matrix.md tests/test_gxtb_integration.py tests/test_electronic_properties.py tests/test_gfnff_adapter.py
git commit -m "docs: describe Python resources and electronic settings"
```

### Task 5: Final review and remote synchronization

**Files:**
- Modify: none unless review finds a concrete defect.

**Interfaces:**
- Consumes: all committed implementation and verification artifacts.
- Produces: a clean branch with pushed commits and evidence separated into unit, integration, and packaging results.

- [ ] **Step 1: Review the diff and public import surface**

```bash
git diff origin/main...HEAD --stat
git diff origin/main...HEAD --check
python -c "from xtb_ase import CalculatorPool, GFNFF, GXTB, XTB; print(CalculatorPool, GFNFF, GXTB, XTB)"
```

- [ ] **Step 2: Re-run final tests from the source checkout**

```bash
python -m pytest -q
PYTHONPATH=vendor/site-packages:src python -m pytest -q
```

- [ ] **Step 3: Push the completed feature**

```bash
git push origin master:main
```

- [ ] **Step 4: Verify remote synchronization**

```bash
git status --short --branch
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

Expected: clean working tree, local and remote `main` point to the same commit, and the final report distinguishes implemented API behavior from backend-dependent integration evidence.
