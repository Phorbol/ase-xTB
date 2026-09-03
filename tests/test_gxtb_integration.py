import os
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np
import pytest
from ase import Atoms, units

from xtb_ase import GXTB, XTB
from xtb_ase._parsers import parse_engrad


def gxtb_command() -> list[str]:
    value = os.environ.get(
        "GXTB_COMMAND",
        "/tmp/gxtb-v2-research.rW5RzX/extracted/xtb-6.7.1/bin/xtb",
    )
    command = shlex.split(value)
    if not command or not Path(command[0]).is_file():
        pytest.skip("set GXTB_COMMAND to run the g-xTB integration tests")
    return command


@pytest.mark.integration
def test_gxtb_wrapper_matches_direct_energy_and_gradient(tmp_path: Path):
    command = gxtb_command()
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.7586, 0.0, 0.5043], [-0.7586, 0.0, 0.5043]],
    )
    wrapped = GXTB(
        command=command,
        directory=tmp_path / "wrapped",
        keep_files=True,
        threads=1,
    )
    atoms.calc = wrapped
    wrapped_energy = atoms.get_potential_energy()
    wrapped_forces = atoms.get_forces()

    direct_dir = tmp_path / "direct"
    direct_dir.mkdir()
    direct_input = direct_dir / "structure.xyz"
    direct_input.write_text(
        "3\ndirect\n"
        "O 0.0 0.0 0.0\n"
        "H 0.7586 0.0 0.5043\n"
        "H -0.7586 0.0 0.5043\n"
    )
    direct_command = command + [
        "structure.xyz",
        "--gxtb",
        "--no-restart",
        "--chrg",
        "0",
        "--acc",
        "1",
        "--etemp",
        "0",
        "--parallel",
        "1",
        "--grad",
    ]
    completed = subprocess.run(
        direct_command,
        cwd=direct_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    direct = parse_engrad(direct_dir / "structure.engrad", natoms=3)

    np.testing.assert_allclose(wrapped_energy, direct.energy_hartree * 27.211386024367243)
    np.testing.assert_allclose(
        wrapped_forces,
        -direct.gradient_hartree_per_bohr * 27.211386024367243 / 0.5291772105638411,
        atol=1e-10,
    )


@pytest.mark.integration
def test_generic_xtb_wrapper_matches_standard_gfn2_cli(tmp_path: Path):
    command = gxtb_command()
    atoms = Atoms(
        "OH2",
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [0.7586, 0.0, 0.5043], [-0.7586, 0.0, 0.5043]]
        ),
    )
    wrapped = XTB(
        command=command,
        method="gfn2",
        directory=tmp_path / "wrapped",
        keep_files=True,
        threads=1,
    )
    atoms.calc = wrapped
    wrapped_energy = atoms.get_potential_energy()
    wrapped_forces = atoms.get_forces()

    direct_dir = tmp_path / "direct"
    direct_dir.mkdir()
    (direct_dir / "structure.xyz").write_text(
        "3\ndirect\n"
        "O 0.0 0.0 0.0\n"
        "H 0.7586 0.0 0.5043\n"
        "H -0.7586 0.0 0.5043\n"
    )
    completed = subprocess.run(
        command
        + [
            "structure.xyz",
            "--gfn",
            "2",
            "--no-restart",
            "--chrg",
            "0",
            "--acc",
            "1",
            "--parallel",
            "1",
            "--grad",
        ],
        cwd=direct_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    direct = parse_engrad(direct_dir / "structure.engrad", natoms=3)

    np.testing.assert_allclose(wrapped_energy, direct.energy_hartree * 27.211386024367243)
    np.testing.assert_allclose(
        wrapped_forces,
        -direct.gradient_hartree_per_bohr * 27.211386024367243 / 0.5291772105638411,
        atol=1e-10,
    )


@pytest.mark.integration
def test_gxtb_python_threads_setting_reaches_the_backend(tmp_path: Path):
    command = gxtb_command()
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.7586, 0.0, 0.5043], [-0.7586, 0.0, 0.5043]],
    )
    calculator = GXTB(
        command=command,
        directory=tmp_path,
        keep_files=True,
        threads=2,
        env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    atoms.calc = calculator

    atoms.get_potential_energy()

    raw = calculator.get_raw_output()
    assert raw["command"][raw["command"].index("--parallel") + 1] == "2"
    assert re.search(r"omp threads\s*:\s*2\b", raw["stdout"])


@pytest.mark.integration
def test_standard_xtb_exposes_common_electronic_properties(tmp_path: Path):
    command = gxtb_command()
    atoms = Atoms(
        "OH2",
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [0.7586, 0.0, 0.5043], [-0.7586, 0.0, 0.5043]]
        ),
    )
    calculator = XTB(
        command=command,
        method="gfn2",
        directory=tmp_path,
        keep_files=True,
        properties=(
            "charges",
            "dipole",
            "quadrupole",
            "bond_orders",
            "orbital_energies",
            "homo_lumo_gap",
        ),
        threads=1,
    )
    atoms.calc = calculator

    atoms.get_potential_energy()

    assert atoms.get_charges().shape == (3,)
    assert atoms.get_dipole_moment()[2] == pytest.approx(0.1699, abs=1e-3)
    assert calculator.get_quadrupole().shape == (3, 3)
    assert calculator.get_bond_orders()[0, 1] > 0.8
    assert calculator.get_orbital_energies().shape == (6,)
    assert calculator.get_homo_lumo_gap() == pytest.approx(16.3266, abs=1e-3)


@pytest.mark.integration
def test_gxtb_native_hessian_is_ase_vibrations_data(tmp_path: Path):
    command = gxtb_command()
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.7586, 0.0, 0.5043], [-0.7586, 0.0, 0.5043]],
    )
    calculator = GXTB(command=command, directory=tmp_path, threads=1)

    data = calculator.get_vibrations_data(atoms)

    assert data.get_hessian_2d().shape == (9, 9)
    assert data.get_frequencies().shape == (9,)
    assert np.isfinite(data.get_frequencies()).all()
    assert np.isfinite(data.get_zero_point_energy())
    assert np.isfinite(calculator.get_hessian(atoms) / (units.Hartree / units.Bohr**2)).all()
