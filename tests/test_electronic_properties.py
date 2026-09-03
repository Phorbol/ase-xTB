import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from xtb_ase import GXTB


def gxtb_path() -> Path:
    path = Path(
        os.environ.get(
            "GXTB_COMMAND",
            "/tmp/gxtb-v2-research.rW5RzX/extracted/xtb-6.7.1/bin/xtb",
        )
    )
    if not path.is_file():
        pytest.skip("set GXTB_COMMAND to run the g-xTB electronic-property tests")
    return path


def water() -> Atoms:
    return Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.7586, 0.0, 0.5043], [-0.7586, 0.0, 0.5043]],
    )


@pytest.mark.integration
def test_gxtb_exposes_common_electronic_properties_in_one_run(tmp_path: Path):
    atoms = water()
    calc = GXTB(
        command=str(gxtb_path()),
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
    atoms.calc = calc

    atoms.get_potential_energy()
    np.testing.assert_allclose(atoms.get_charges().sum(), 0.0, atol=1e-6)
    np.testing.assert_allclose(atoms.get_dipole_moment()[2], 0.4282, atol=1e-3)
    quadrupole = calc.get_quadrupole()
    assert quadrupole.shape == (3, 3)
    np.testing.assert_allclose(quadrupole, quadrupole.T)
    assert quadrupole[0, 0] > 0.3
    wbo = calc.get_bond_orders()
    assert wbo.shape == (3, 3)
    np.testing.assert_allclose(wbo, wbo.T)
    assert wbo[0, 1] > 0.8
    assert calc.get_orbital_energies().shape == (6,)
    assert calc.get_orbital_occupations().shape == (6,)
    assert calc.get_homo_lumo_gap() > 20.0
    assert calc.get_homo_lumo_gap(unit="hartree") == pytest.approx(
        calc.get_homo_lumo_gap() / 27.211386024367243
    )
    assert calc.get_molden_path().is_file()


@pytest.mark.integration
def test_gxtb_exposes_hessian_and_vibrational_frequencies(tmp_path: Path):
    atoms = water()
    calc = GXTB(
        command=str(gxtb_path()),
        directory=tmp_path,
        keep_files=True,
        properties=("hessian", "vibrational_frequencies"),
        threads=1,
    )
    atoms.calc = calc

    hessian = calc.get_hessian(atoms)
    frequencies = calc.get_vibrational_frequencies()

    assert hessian.shape == (9, 9)
    assert np.isfinite(hessian).all()
    assert frequencies.shape == (9,)
    assert frequencies[-1] > 4000.0


def test_gxtb_does_not_claim_unverified_stress_or_polarizability(tmp_path: Path):
    calc = GXTB(
        command="/does/not/exist",
        directory=tmp_path,
    )

    with pytest.raises(PropertyNotImplementedError):
        calc.get_property("stress")
    assert "polarizability" not in calc.implemented_properties
