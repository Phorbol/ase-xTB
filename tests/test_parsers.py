from pathlib import Path

import numpy as np
import pytest

from xtb_ase._parsers import (
    parse_charges,
    parse_engrad,
    parse_hessian,
    parse_molden,
    parse_stdout_properties,
    parse_vibspectrum,
    parse_wbo,
)


ENGRAD = """#
# Number of atoms
#
           3
#
# The current total energy in Eh
#
      -76.432502146434
#
# The current gradient in Eh/bohr
#
       0.1000000000  -0.2000000000   0.3000000000
      -0.4000000000   0.5000000000  -0.6000000000
       0.7000000000  -0.8000000000   0.9000000000
#
# The atomic numbers and current coordinates in Bohr
#
"""


STDOUT = """
         :: total energy             -76.432502146434 Eh    ::
         :: HOMO-LUMO gap             0.7906389 Eh           21.5144 eV    ::

Atomic dipole moments (in atomic units):
---------------------------------------------------------
     #    Z                 x            y            z
---------------------------------------------------------
           total     0.000000    -0.000000     0.809228
---------------------------------------------------------
         |total|     0.809228     2.056851 Debye

Atomic quadrupole moments (in atomic units):
-----------------------------------------------------------------------------------
           total     1.4552     0.0000    -1.3474    -0.0000     0.0000    -0.1078
"""


MOLDEN = """[Molden Format]
[MO]
Sym=     1a
Ene=   -1.08497831311066
Spin= Alpha
Occup=     2.00000000
Sym=     2a
Ene=    0.358473770376968
Spin= Alpha
Occup=     0.00000000
"""


def test_parse_engrad_returns_energy_and_gradient(tmp_path: Path):
    path = tmp_path / "water.engrad"
    path.write_text(ENGRAD)

    parsed = parse_engrad(path, natoms=3)

    assert parsed.energy_hartree == pytest.approx(-76.432502146434)
    np.testing.assert_allclose(
        parsed.gradient_hartree_per_bohr,
        np.arange(1, 10, dtype=float).reshape(3, 3) / 10
        * np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]]),
    )


def test_parse_charges_requires_one_value_per_atom(tmp_path: Path):
    path = tmp_path / "charges"
    path.write_text(" -0.65530845\n 0.32765423\n 0.32765423\n")

    np.testing.assert_allclose(
        parse_charges(path, natoms=3),
        [-0.65530845, 0.32765423, 0.32765423],
    )


def test_parse_stdout_properties_returns_energy_gap_and_dipole():
    parsed = parse_stdout_properties(STDOUT, natoms=3)

    assert parsed.total_energy_hartree == pytest.approx(-76.432502146434)
    assert parsed.homo_lumo_gap_ev == pytest.approx(21.5144)
    np.testing.assert_allclose(parsed.dipole_au, [0.0, 0.0, 0.809228])
    assert parsed.dipole_debye == pytest.approx(2.056851)
    np.testing.assert_allclose(
        parsed.quadrupole_au,
        [1.4552, 0.0, -1.3474, 0.0, 0.0, -0.1078],
    )


def test_parse_stdout_properties_supports_standard_full_quadrupole():
    text = """
molecular quadrupole (traceless):
                xx          xy          yy          xz          yz          zz
  q only:        0.926       0.000      -0.857      -0.000       0.000      -0.069
  q+dip:         1.176       0.000      -1.072      -0.000       0.000      -0.104
   full:         1.408       0.000      -1.343      -0.000       0.000      -0.065
"""

    parsed = parse_stdout_properties(text, natoms=3)

    np.testing.assert_allclose(
        parsed.quadrupole_au,
        [1.408, 0.0, -1.343, 0.0, 0.0, -0.065],
    )


def test_parse_stdout_properties_supports_standard_full_molecular_dipole():
    text = """
molecular dipole:
                 x           y           z       tot (Debye)
 q only:       -0.000       0.000       0.551
   full:       -0.000       0.000       0.816       2.073
"""

    parsed = parse_stdout_properties(text, natoms=3)

    np.testing.assert_allclose(parsed.dipole_debye_vector, [0.0, 0.0, 0.816])
    assert parsed.dipole_debye == pytest.approx(2.073)


def test_parse_wbo_builds_symmetric_atom_pair_matrix(tmp_path: Path):
    path = tmp_path / "wbo"
    path.write_text(
        "           1           2  0.884698650757462\n"
        "           1           3  0.884698650757462\n"
    )

    result = parse_wbo(path, natoms=3)

    assert result.shape == (3, 3)
    np.testing.assert_allclose(result, result.T)
    assert result[0, 1] == pytest.approx(0.884698650757462)
    assert result[1, 2] == 0.0


def test_parse_molden_returns_orbital_energies_and_occupations(tmp_path: Path):
    path = tmp_path / "molden.input"
    path.write_text(MOLDEN)

    parsed = parse_molden(path)

    np.testing.assert_allclose(parsed.energies_hartree, [-1.08497831311066, 0.358473770376968])
    np.testing.assert_allclose(parsed.occupations, [2.0, 0.0])


def test_parse_hessian_and_vibrational_spectrum(tmp_path: Path):
    hessian_path = tmp_path / "hessian"
    hessian_path.write_text("$hessian\n" + " ".join(str(float(i)) for i in range(81)) + "\n")
    vib_path = tmp_path / "vibspectrum"
    vib_path.write_text(
        "$vibrational spectrum\n"
        "# mode symmetry wave number IR intensity selection rules\n"
        "     1                      -0.00         0.00000          -\n"
        "     7        a           1377.77       121.46117          YES\n"
        "$end\n"
    )

    hessian = parse_hessian(hessian_path, natoms=3)
    frequencies = parse_vibspectrum(vib_path)

    assert hessian.shape == (9, 9)
    assert hessian[0, 0] == 0.0
    assert hessian[-1, -1] == 80.0
    np.testing.assert_allclose(frequencies, [-0.0, 1377.77])
