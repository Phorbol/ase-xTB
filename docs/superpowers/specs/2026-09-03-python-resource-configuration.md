# Python Resource and Electronic-Parameter Configuration

## Goal

Expose xTB-family runtime resources and common electronic parameters through
Python without confusing per-calculation threads with independent task
processes.

## Scope

The feature covers the public `XTB`, `GXTB`, and `GFNFF` ASE calculators plus a
small process-pool helper for independent structures. It does not add GPU
support, MPI, periodic xTB stress, or new native GFN-FF capabilities.

## Interface contract

### `XTB` and `GXTB`

The constructors and `set()` accept:

```python
XTB(
    *,
    charge=0,
    uhf=None,
    spin=None,
    unpaired_electrons=None,
    etemp=None,
    electronic_temperature=None,
    threads=None,
    parallel=None,
    env=None,
)
```

- `threads` is the canonical per-invocation thread count. It emits
  `--parallel N`, which the xTB executable uses to configure its OpenMP/MKL
  execution. `parallel` remains an alias for compatibility.
- `threads=None` omits `--parallel`; inherited `OMP_NUM_THREADS` and
  `MKL_NUM_THREADS` can then control the executable.
- If `threads` and `parallel` are both provided with different values, raise
  `ValueError`. Equal values are accepted.
- `spin` and `unpaired_electrons` are aliases for xTB's `uhf` value: they mean
  the number of unpaired electrons, not a spin multiplicity. Conflicting
  aliases raise `ValueError`.
- `electronic_temperature` is an alias for `etemp`. Conflicting aliases raise
  `ValueError`.
- `env` is a mapping applied to the environment of the xTB child process. Its
  values are converted to strings; a value of `None` removes an inherited
  variable. The parent process environment is never mutated.
- Existing `charge`, `accuracy`, `solvation_model`, `solvent`, `extra_args`,
  and output-property behavior remain available.
- `GXTB` keeps the fixed `gxtb` method and 0 K default, while accepting all
  resource and electronic aliases above.

### `GFNFF`

`GFNFF` accepts `threads` and `env` in addition to its existing `charge`,
`solvent`, `printlevel`, `fragments`, and `ref_charges` parameters.

- `charge` and `solvent` are passed to the standalone native backend.
- `threads` calls the native OpenMP runtime setter before the calculation. The
  setting is process-wide for the loaded GFN-FF library, so different thread
  counts must be isolated with `CalculatorPool` workers.
- `env` is applied temporarily while the optional backend is loaded and a
  calculation is initialized. It is process-local but not instance-isolated;
  `CalculatorPool` is the safe route for concurrent configurations.
- `uhf`, `spin`, `unpaired_electrons`, `etemp`, and
  `electronic_temperature` are rejected with a clear `TypeError` because
  GFN-FF is a force field and has no electronic spin or electronic-temperature
  input.

### Independent task processes

Add a public `CalculatorPool`:

```python
from functools import partial
from xtb_ase import CalculatorPool, GXTB

pool = CalculatorPool(
    partial(GXTB, command="/opt/gxtb/xtb", threads=4),
    processes=8,
)
results = pool.map(atoms_list, properties=("energy", "forces"))
```

- `processes` is the number of independent worker processes and defaults to 1.
- Multi-process execution defaults to the `spawn` multiprocessing context for
  safe native-library isolation; `mp_context` can select another supported
  context explicitly.
- `calculator_factory` must be pickleable when `processes > 1`; importable
  functions and `functools.partial` are supported.
- `map()` preserves input order and returns one result dictionary per input,
  containing the requested property names and copied scalar/array values.
- Each worker constructs a fresh Calculator and receives a private process
  environment. The calculator's `directory`/temporary-directory behavior
  keeps xTB scratch files independent.
- `CalculatorPool` is for independent calculations; an ASE optimizer/MD/NEB
  still uses one Calculator for its sequential stateful trajectory.

## Validation rules

- `threads`, `parallel`, and GFN-FF `threads` must be positive integers or
  `None`.
- xTB `uhf`, `spin`, and `unpaired_electrons` must be non-negative integers or
  `None`.
- Existing solvation-model validation remains: `solvation_model` is `gbe` or
  `cosmo`, and requires `solvent`.
- `CalculatorPool.processes` must be a positive integer.

## Compatibility

The old `parallel=N`, `uhf=N`, and `etemp=T` spellings remain valid. The
default is changed from an explicit `--parallel 1` to no explicit thread
override, so the executable/OpenMP environment controls its default unless a
thread count is supplied.

## Verification evidence

Unit tests will inspect the fake executable's arguments and environment,
exercise alias conflicts and `set()`, validate GFN-FF configuration behavior,
and run a two-item `CalculatorPool`. Integration tests will retain the real
g-XTB and optional GFN-FF checks, with explicit `threads=1` where a stable
single-thread comparison is required.
