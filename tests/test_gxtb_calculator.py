from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, units

from xtb_ase import GXTB, XTB
from xtb_ase.gxtb import GXTBExecutionError


def make_fake_xtb(
    path: Path,
    *,
    fail: bool = False,
    record_environment: bool = False,
) -> Path:
    script = f"""#!/usr/bin/env python3
import os
from pathlib import Path
import sys

args = sys.argv[1:]
Path('args.txt').write_text('\\n'.join(args) + '\\n')
if {record_environment!r}:
    Path('env.txt').write_text(os.environ.get('XTBASE_TEST_ENV', '<missing>'))
input_path = Path(args[0])
Path('seen.xyz').write_text(input_path.read_text())
if {fail!r}:
    print('fake SCF did not converge', file=sys.stderr)
    raise SystemExit(7)

if '--hess' in args:
    Path('hessian').write_text('$hessian\\n' + ' '.join(str(float(i)) for i in range(81)) + '\\n')
    Path('vibspectrum').write_text('$vibrational spectrum\\n 1 -0.00 0.0 -\\n$end\\n')
    print('''
         :: total free energy         -76.427574590877 Eh   ::
         :: zero point energy           0.022378390734 Eh   ::
          | TOTAL ENTHALPY            -76.406340598102 Eh   |
          | TOTAL FREE ENERGY         -76.427574590877 Eh   |
    298.15    0.378488E-02    0.261633E-01    0.212340E-01    0.492928E-02
''')
else:
    Path('structure.engrad').write_text('''#\\n# The current total energy in Eh\\n#\\n-2.0\\n#\\n# The current gradient in Eh/bohr\\n#\\n0.1\\n-0.2\\n0.3\\n0.4\\n-0.5\\n0.6\\n0.7\\n-0.8\\n0.9\\n#\\n''')
    Path('charges').write_text('-0.2\\n0.1\\n0.1\\n')
    if '--wbo' in args:
        Path('wbo').write_text('1 2 0.8\\n1 3 0.8\\n')
    if '--molden' in args:
        Path('molden.input').write_text('[Molden Format]\\n[MO]\\nEne= -0.5\\nOccup= 2.0\\n')

print(''':: total energy             -2.0 Eh    ::
:: HOMO-LUMO gap             0.5 Eh           13.6057 eV    ::
Atomic dipole moments (in atomic units):
           total     0.000000    0.000000     1.000000
         |total|     1.000000     2.541747 Debye
Atomic quadrupole moments (in atomic units):
           total     1.0     0.0    -1.0     0.0     0.0     0.0
''')
"""
    path.write_text(script)
    path.chmod(0o755)
    return path


def test_gxtb_calculator_exposes_ase_base_properties_and_units(tmp_path: Path):
    executable = make_fake_xtb(tmp_path / "fake-xtb")
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.5], [-0.7, 0.0, 0.5]],
    )
    calculator = GXTB(
        command=str(executable),
        charge=-1,
        uhf=2,
        accuracy=0.25,
        etemp=0.0,
        directory=tmp_path,
        keep_files=True,
    )
    atoms.calc = calculator

    assert atoms.get_potential_energy() == pytest.approx(-2.0 * units.Hartree)
    np.testing.assert_allclose(
        atoms.get_forces(),
        -np.asarray([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6], [0.7, -0.8, 0.9]])
        * units.Hartree
        / units.Bohr,
    )
    np.testing.assert_allclose(atoms.get_charges(), [-0.2, 0.1, 0.1])
    np.testing.assert_allclose(atoms.get_dipole_moment(), [0.0, 0.0, units.Bohr])

    run_directory = calculator.get_run_directory()
    assert run_directory is not None
    args = (run_directory / "args.txt").read_text().splitlines()
    assert "--gxtb" in args
    assert "--grad" in args
    assert args[args.index("--chrg") + 1] == "-1"
    assert args[args.index("--uhf") + 1] == "2"
    assert args[args.index("--acc") + 1] == "0.25"

    xyz = (run_directory / "seen.xyz").read_text().splitlines()
    assert xyz[0] == "3"
    assert xyz[2].split()[0] == "O"
    assert float(xyz[3].split()[1]) == pytest.approx(0.7)


def test_generic_xtb_selects_standard_gfn_method(tmp_path: Path):
    executable = make_fake_xtb(tmp_path / "fake-xtb")
    atoms = Atoms(
        "OH2",
        positions=np.asarray([[0.0, 0.0, 0.0], [0.7, 0.0, 0.5], [-0.7, 0.0, 0.5]]),
    )
    calculator = XTB(
        command=str(executable),
        method="GFN2-xTB",
        directory=tmp_path,
        keep_files=True,
    )
    atoms.calc = calculator

    atoms.get_potential_energy()

    run_directory = calculator.get_run_directory()
    assert run_directory is not None
    args = (run_directory / "args.txt").read_text().splitlines()
    assert args[args.index("--gfn") + 1] == "2"
    assert "--gxtb" not in args


def test_generic_xtb_preserves_standard_cli_electronic_temperature_default(
    tmp_path: Path,
):
    executable = make_fake_xtb(tmp_path / "fake-xtb")
    atoms = Atoms("OH2", positions=np.asarray([[0.0, 0.0, 0.0], [0.7, 0.0, 0.5], [-0.7, 0.0, 0.5]]))
    calculator = XTB(
        command=str(executable),
        method="gfn2",
        directory=tmp_path,
        keep_files=True,
    )
    atoms.calc = calculator

    atoms.get_potential_energy()

    run_directory = calculator.get_run_directory()
    assert run_directory is not None
    args = (run_directory / "args.txt").read_text().splitlines()
    assert "--etemp" not in args


def test_xtb_solvation_options_are_explicit(tmp_path: Path):
    executable = make_fake_xtb(tmp_path / "fake-xtb")
    atoms = Atoms("OH2", positions=np.asarray([[0.0, 0.0, 0.0], [0.7, 0.0, 0.5], [-0.7, 0.0, 0.5]]))
    calculator = GXTB(
        command=str(executable),
        solvation_model="COSMO",
        solvent="water",
        directory=tmp_path,
        keep_files=True,
    )
    atoms.calc = calculator

    atoms.get_potential_energy()

    run_directory = calculator.get_run_directory()
    assert run_directory is not None
    args = (run_directory / "args.txt").read_text().splitlines()
    assert args[args.index("--cosmo") + 1] == "water"


def test_gxtb_python_aliases_control_resources_and_electronic_settings(
    tmp_path: Path,
):
    executable = make_fake_xtb(
        tmp_path / "fake-xtb",
        record_environment=True,
    )
    atoms = Atoms(
        "OH2",
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [0.7, 0.0, 0.5], [-0.7, 0.0, 0.5]]
        ),
    )
    calculator = GXTB(
        command=str(executable),
        directory=tmp_path,
        keep_files=True,
        threads=4,
        spin=2,
        electronic_temperature=300.0,
        env={"XTBASE_TEST_ENV": "child-only"},
    )
    atoms.calc = calculator

    atoms.get_potential_energy()

    run_directory = calculator.get_run_directory()
    assert run_directory is not None
    args = (run_directory / "args.txt").read_text().splitlines()
    assert args[args.index("--parallel") + 1] == "4"
    assert args[args.index("--uhf") + 1] == "2"
    assert args[args.index("--etemp") + 1] == "300"
    assert (run_directory / "env.txt").read_text() == "child-only"


def test_gxtb_omits_parallel_when_threads_are_unspecified(tmp_path: Path):
    executable = make_fake_xtb(tmp_path / "fake-xtb")
    atoms = Atoms(
        "OH2",
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [0.7, 0.0, 0.5], [-0.7, 0.0, 0.5]]
        ),
    )
    calculator = GXTB(
        command=str(executable),
        directory=tmp_path,
        keep_files=True,
        threads=None,
    )
    atoms.calc = calculator

    atoms.get_potential_energy()

    run_directory = calculator.get_run_directory()
    assert run_directory is not None
    assert "--parallel" not in (run_directory / "args.txt").read_text().splitlines()


def test_gxtb_alias_conflicts_are_rejected():
    with pytest.raises(ValueError, match="threads.*parallel"):
        GXTB(command="/missing/xtb", threads=2, parallel=4)

    with pytest.raises(ValueError, match="uhf.*spin"):
        GXTB(command="/missing/xtb", uhf=1, spin=2)

    with pytest.raises(ValueError, match="etemp.*electronic_temperature"):
        GXTB(command="/missing/xtb", etemp=100.0, electronic_temperature=300.0)

    with pytest.raises(TypeError, match="CalculatorPool"):
        GXTB(command="/missing/xtb", processes=2)


def test_gxtb_set_accepts_thread_and_spin_aliases(tmp_path: Path):
    executable = make_fake_xtb(tmp_path / "fake-xtb")
    atoms = Atoms(
        "OH2",
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [0.7, 0.0, 0.5], [-0.7, 0.0, 0.5]]
        ),
    )
    calculator = GXTB(
        command=str(executable),
        directory=tmp_path,
        keep_files=True,
    )
    calculator.set(threads=3, spin=1)
    atoms.calc = calculator

    atoms.get_potential_energy()

    run_directory = calculator.get_run_directory()
    assert run_directory is not None
    args = (run_directory / "args.txt").read_text().splitlines()
    assert args[args.index("--parallel") + 1] == "3"
    assert args[args.index("--uhf") + 1] == "1"


def test_gxtb_calculator_reuses_results_for_energy_and_force_requests(tmp_path: Path):
    executable = make_fake_xtb(tmp_path / "fake-xtb")
    atoms = Atoms(
        "H2O",
        positions=np.asarray([[0, 0, 0], [0.7, 0, 0], [-0.7, 0, 0]]),
    )
    calculator = GXTB(command=str(executable), directory=tmp_path, keep_files=True)
    atoms.calc = calculator

    atoms.get_potential_energy()
    first_run = calculator.get_run_directory()
    atoms.get_forces()

    assert calculator.get_run_directory() == first_run
    assert len(list(tmp_path.glob("xtb-ase-*/args.txt"))) == 1


def test_gxtb_calculator_reports_process_failures(tmp_path: Path):
    executable = make_fake_xtb(tmp_path / "fake-xtb", fail=True)
    atoms = Atoms("H2", positions=np.asarray([[0, 0, 0], [0.7, 0, 0]]))
    atoms.calc = GXTB(command=str(executable), directory=tmp_path)

    with pytest.raises(GXTBExecutionError, match="did not converge"):
        atoms.get_potential_energy()


def test_gxtb_calculator_reports_missing_required_output(tmp_path: Path):
    atoms = Atoms("H2", positions=np.asarray([[0, 0, 0], [0.7, 0, 0]]))
    atoms.calc = GXTB(command="/usr/bin/true", directory=tmp_path)

    with pytest.raises(GXTBExecutionError, match="required xTB output file missing"):
        atoms.get_potential_energy()


def test_gxtb_keep_files_writes_raw_stdout_and_stderr(tmp_path: Path):
    executable = make_fake_xtb(tmp_path / "fake-xtb")
    atoms = Atoms(
        "H2O",
        positions=np.asarray([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [-0.7, 0.0, 0.0]]),
    )
    calculator = GXTB(command=str(executable), directory=tmp_path, keep_files=True)
    atoms.calc = calculator

    atoms.get_potential_energy()

    run_directory = calculator.get_run_directory()
    assert run_directory is not None
    assert (run_directory / "stdout").read_text().startswith(":: total energy")
    assert (run_directory / "stderr").is_file()


def test_gxtb_molden_artifact_survives_a_later_hessian_request(tmp_path: Path):
    executable = make_fake_xtb(tmp_path / "fake-xtb")
    atoms = Atoms(
        "H2O",
        positions=np.asarray([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [-0.7, 0.0, 0.0]]),
    )
    calculator = GXTB(command=str(executable), directory=tmp_path, keep_files=True)
    atoms.calc = calculator

    molden = calculator.get_molden_path(atoms)
    calculator.get_hessian(atoms)

    assert molden.is_file()
    assert calculator.get_molden_path(atoms) == molden


def test_gxtb_exposes_gibbs_free_energy_in_ev(tmp_path: Path):
    executable = make_fake_xtb(tmp_path / "fake-xtb")
    atoms = Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [-0.7, 0.0, 0.0]],
    )
    calculator = GXTB(
        command=str(executable),
        directory=tmp_path,
        properties=("gibbs_free_energy",),
    )
    atoms.calc = calculator

    assert calculator.get_gibbs_free_energy(atoms) == pytest.approx(
        -76.427574590877 * units.Hartree
    )


def test_gxtb_returns_vibrations_data(tmp_path: Path):
    executable = make_fake_xtb(tmp_path / "fake-xtb")
    atoms = Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [-0.7, 0.0, 0.0]],
    )
    calculator = GXTB(command=str(executable), directory=tmp_path)

    data = calculator.get_vibrations_data(atoms)

    assert data.get_hessian_2d().shape == (9, 9)
    np.testing.assert_allclose(
        data.get_hessian_2d(),
        np.arange(81).reshape(9, 9) * units.Hartree / units.Bohr**2,
    )
