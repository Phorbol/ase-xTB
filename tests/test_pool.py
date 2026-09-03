from functools import partial
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, units

from xtb_ase import CalculatorPool, GXTB


def make_pool_fake_xtb(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys

args = sys.argv[1:]
Path('args.txt').write_text('\\n'.join(args) + '\\n')
Path('structure.engrad').write_text('''#
# The current total energy in Eh
#
-2.0
#
# The current gradient in Eh/bohr
#
0.1
-0.2
0.3
0.4
-0.5
0.6
0.7
-0.8
0.9
#
''')
print(':: total energy             -2.0 Eh    ::')
"""
    )
    path.chmod(0o755)
    return path


def water(x_shift: float) -> Atoms:
    return Atoms(
        "OH2",
        positions=np.asarray(
            [
                [x_shift, 0.0, 0.0],
                [x_shift + 0.7, 0.0, 0.5],
                [x_shift - 0.7, 0.0, 0.5],
            ]
        ),
    )


def test_pool_preserves_order_and_returns_requested_results(tmp_path: Path):
    executable = make_pool_fake_xtb(tmp_path / "fake-xtb")
    factory = partial(
        GXTB,
        command=str(executable),
        directory=tmp_path,
        threads=1,
    )

    results = CalculatorPool(factory, processes=2).map(
        [water(0.0), water(1.0)],
        properties=("energy", "forces"),
    )

    assert [result["energy"] for result in results] == pytest.approx(
        [-2.0 * units.Hartree, -2.0 * units.Hartree]
    )
    assert all(result["forces"].shape == (3, 3) for result in results)


def test_pool_validates_processes_and_properties(tmp_path: Path):
    executable = make_pool_fake_xtb(tmp_path / "fake-xtb")
    factory = partial(GXTB, command=str(executable), directory=tmp_path)

    with pytest.raises(ValueError, match="processes"):
        CalculatorPool(factory, processes=0)
    with pytest.raises(ValueError, match="processes"):
        CalculatorPool(factory, processes=None)

    pool = CalculatorPool(factory)
    with pytest.raises(ValueError, match="properties"):
        pool.map([water(0.0)], properties=())


def test_pool_defaults_to_spawn_for_native_backend_isolation(tmp_path: Path):
    executable = make_pool_fake_xtb(tmp_path / "fake-xtb")
    factory = partial(GXTB, command=str(executable), directory=tmp_path)

    pool = CalculatorPool(factory, processes=2)

    assert pool.mp_context == "spawn"


def test_pool_rejects_unpickleable_factory_for_multiple_processes(tmp_path: Path):
    executable = make_pool_fake_xtb(tmp_path / "fake-xtb")

    with pytest.raises(TypeError, match="pickle"):
        CalculatorPool(
            lambda: GXTB(command=str(executable), directory=tmp_path),
            processes=2,
        )
