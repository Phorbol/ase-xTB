from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, units

from xtb_ase import GXTB, XTB
from xtb_ase.gxtb import GXTBExecutionError


def make_fake_xtb(path: Path, *, fail: bool = False) -> Path:
    script = f"""#!/usr/bin/env python3
from pathlib import Path
import sys

args = sys.argv[1:]
Path('args.txt').write_text('\\n'.join(args) + '\\n')
input_path = Path(args[0])
Path('seen.xyz').write_text(input_path.read_text())
if {fail!r}:
    print('fake SCF did not converge', file=sys.stderr)
    raise SystemExit(7)

if '--hess' in args:
    Path('hessian').write_text('$hessian\\n' + ' '.join(str(float(i)) for i in range(81)) + '\\n')
    Path('vibspectrum').write_text('$vibrational spectrum\\n 1 -0.00 0.0 -\\n$end\\n')
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
