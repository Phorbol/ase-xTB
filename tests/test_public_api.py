from xtb_ase import (
    GFNFF,
    GFNFFDependencyError,
    GXTB,
    GXTBExecutionError,
    XTB,
)


def test_public_calculator_exports_and_properties():
    assert GXTB.implemented_properties[:4] == [
        "energy",
        "forces",
        "charges",
        "dipole",
    ]
    assert {
        "bond_orders",
        "orbital_energies",
        "orbital_occupations",
        "homo_lumo_gap",
        "hessian",
        "vibrational_frequencies",
    }.issubset(GXTB.implemented_properties)
    assert GFNFF.implemented_properties == ["energy", "forces", "stress"]
    assert issubclass(GFNFFDependencyError, ImportError)
    assert issubclass(GXTBExecutionError, RuntimeError)


def test_calculators_can_be_constructed_without_running_backend():
    gxtb = GXTB(command="/missing/xtb")
    xtb = XTB(command="/missing/xtb", method="gfn2-xtb")
    gfnff = GFNFF()

    assert gxtb.parameters.charge == 0
    assert gxtb.parameters.uhf is None
    assert gxtb.parameters.method == "gxtb"
    assert xtb.parameters.method == "gfn2-xtb"
    assert gfnff.parameters.charge == 0


def test_search_namespace_is_importable_without_optional_backends():
    import xtb_ase.search as search

    assert search.__name__ == "xtb_ase.search"
