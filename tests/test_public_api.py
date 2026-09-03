from xtb_ase import (
    ASEVibrationalThermochemistry,
    GFNFF,
    GFNFFDependencyError,
    GXTB,
    GXTBExecutionError,
    XTB,
    ase_vibrational_thermochemistry,
    get_vibrations_data,
    hessian_to_vibrations_data,
    run_vibrations,
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
        "gibbs_free_energy",
        "enthalpy",
        "zero_point_energy",
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


def test_ase_vibration_helpers_are_public():
    assert ASEVibrationalThermochemistry.__name__ == "ASEVibrationalThermochemistry"
    assert callable(ase_vibrational_thermochemistry)
    assert callable(get_vibrations_data)
    assert callable(hessian_to_vibrations_data)
    assert callable(run_vibrations)


def test_search_namespace_is_importable_without_optional_backends():
    import xtb_ase.search as search

    assert search.__name__ == "xtb_ase.search"
