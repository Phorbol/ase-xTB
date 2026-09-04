from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
try:
    from ase.thermochemistry import IdealGasThermo, MSRRHOThermo
except ImportError:  # pragma: no cover - supported by the optional test guard
    from ase.thermochemistry import IdealGasThermo

    MSRRHOThermo = None

from xtb_ase.vibrations import (
    ASEQuasiRRHOThermochemistry,
    ASEVibrationalThermochemistry,
    ase_quasi_rrho_thermochemistry,
    ase_vibrational_thermochemistry,
    get_vibrations_data,
    hessian_to_vibrations_data,
    run_vibrations,
)


def water() -> Atoms:
    return Atoms(
        "OH2",
        positions=[
            [0.0, 0.0, 0.0],
            [0.7586, 0.0, 0.5043],
            [-0.7586, 0.0, 0.5043],
        ],
    )


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, reference: np.ndarray, spring_constant: float = 0.5):
        super().__init__()
        self.reference = np.asarray(reference, dtype=float).copy()
        self.spring_constant = float(spring_constant)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        displacement = self.atoms.positions - self.reference
        self.results["energy"] = 0.5 * self.spring_constant * np.sum(displacement**2)
        self.results["forces"] = -self.spring_constant * displacement


class HessianCalculator(HarmonicCalculator):
    def __init__(self, reference: np.ndarray, hessian: np.ndarray):
        super().__init__(reference)
        self.hessian = np.asarray(hessian, dtype=float).copy()
        self.hessian_calls = 0

    def get_hessian(self, atoms=None):
        self.hessian_calls += 1
        return self.hessian.copy()


def positive_hessian(natoms: int) -> np.ndarray:
    return np.eye(3 * natoms) * 0.5


def test_hessian_adapter_returns_vibrations_data_and_zpe():
    atoms = water()
    data = hessian_to_vibrations_data(atoms, positive_hessian(len(atoms)))

    assert data.get_hessian_2d().shape == (9, 9)
    assert np.isfinite(data.get_frequencies()).all()
    assert data.get_zero_point_energy() > 0.0
    assert data.get_atoms().calc is None


def test_hessian_adapter_slices_active_indices():
    atoms = water()
    hessian = positive_hessian(len(atoms))
    data = hessian_to_vibrations_data(atoms, hessian, indices=[1, 2])

    assert data.get_indices().tolist() == [1, 2]
    assert data.get_hessian_2d().shape == (6, 6)


def test_get_vibrations_data_prefers_analytic_calculator_hessian():
    atoms = water()
    calculator = HessianCalculator(atoms.positions, positive_hessian(len(atoms)))

    data = get_vibrations_data(atoms, calculator=calculator)

    assert calculator.hessian_calls == 1
    assert data.get_hessian_2d().shape == (9, 9)


def test_get_vibrations_data_falls_back_to_ase_force_finite_difference():
    atoms = water()
    spring_constant = 0.5
    calculator = HarmonicCalculator(atoms.positions, spring_constant=spring_constant)

    data = get_vibrations_data(atoms, calculator=calculator)

    np.testing.assert_allclose(
        data.get_hessian_2d(),
        np.eye(9) * spring_constant,
        atol=1e-10,
    )


def test_run_vibrations_uses_ase_force_calculator_without_mutation():
    atoms = water()
    original_positions = atoms.positions.copy()
    calculator = HarmonicCalculator(atoms.positions)

    data = run_vibrations(atoms, calculator=calculator, delta=0.005, name=None)

    np.testing.assert_allclose(atoms.positions, original_positions)
    assert atoms.calc is None
    assert data.get_atoms().calc is None
    assert data.get_hessian_2d().shape == (9, 9)


def test_run_vibrations_rejects_periodic_molecular_input():
    atoms = water()
    atoms.cell = np.eye(3) * 10.0
    atoms.pbc = True
    with pytest.raises(NotImplementedError, match="periodic"):
        run_vibrations(atoms, calculator=HarmonicCalculator(atoms.positions))


def test_ase_thermochemistry_matches_direct_ideal_gas_thermo():
    atoms = water()
    data = hessian_to_vibrations_data(atoms, positive_hessian(len(atoms)))
    potential_energy = -5.0
    result = ase_vibrational_thermochemistry(
        atoms,
        data,
        temperature_K=298.15,
        pressure=units.bar,
        geometry="nonlinear",
        symmetrynumber=1,
        spin=0,
        potential_energy=potential_energy,
    )
    expected_thermo = IdealGasThermo(
        vib_energies=data.get_energies(),
        geometry="nonlinear",
        potentialenergy=potential_energy,
        atoms=atoms,
        symmetrynumber=1,
        spin=0,
    )

    assert isinstance(result, ASEVibrationalThermochemistry)
    assert result.zero_point_energy_eV == pytest.approx(
        expected_thermo.get_ZPE_correction()
    )
    assert result.enthalpy_eV == pytest.approx(
        expected_thermo.get_enthalpy(298.15, verbose=False)
    )
    assert result.gibbs_free_energy_eV == pytest.approx(
        expected_thermo.get_gibbs_energy(298.15, units.bar, verbose=False)
    )
    assert result.free_energy_eV == pytest.approx(result.gibbs_free_energy_eV)
    assert result.thermochemical_correction_eV == pytest.approx(
        result.gibbs_free_energy_eV - potential_energy
    )
    assert result.entropy_eV_per_K > 0.0


def test_ase_thermochemistry_requires_potential_energy_without_calculator():
    atoms = water()
    data = hessian_to_vibrations_data(atoms, positive_hessian(len(atoms)))
    with pytest.raises(ValueError, match="potential energy"):
        ase_vibrational_thermochemistry(
            atoms,
            data,
            geometry="nonlinear",
            symmetrynumber=1,
            spin=0,
        )


@pytest.mark.skipif(MSRRHOThermo is None, reason="ASE MSRRHOThermo is unavailable")
def test_ase_quasi_rrho_matches_ase_msr_rhho_with_xtb_conventions():
    atoms = water()
    data = hessian_to_vibrations_data(atoms, positive_hessian(len(atoms)))
    potential_energy = -5.0
    temperature = 400.0
    pressure = 101325.0 * units.Pascal
    frequency_scale = 0.92

    result = ase_quasi_rrho_thermochemistry(
        atoms,
        data,
        temperature_K=temperature,
        pressure=pressure,
        geometry="nonlinear",
        symmetrynumber=1,
        spin=0,
        potential_energy=potential_energy,
        rotor_cutoff_cm1=50.0,
        frequency_scale=frequency_scale,
    )

    energies = np.asarray(data.get_energies(), dtype=complex)
    selected = np.asarray(
        [float(np.real(energy))
         for energy in sorted(energies, key=lambda value: (value**2).real)[-3:]],
    )
    expected_thermo = MSRRHOThermo(
        vib_energies=selected,
        atoms=atoms,
        potentialenergy=potential_energy,
        tau=50.0,
        nu_scal=frequency_scale,
        treat_int_energy=False,
    )
    entropy_external, _ = expected_thermo.get_ideal_entropy(
        temperature,
        translation=True,
        vibration=False,
        rotation=True,
        geometry="nonlinear",
        electronic=False,
        pressure=pressure,
        symmetrynumber=1,
    )
    expected_entropy = expected_thermo.get_entropy(temperature, verbose=False)
    expected_entropy += entropy_external
    expected_internal_energy = expected_thermo.get_internal_energy(
        temperature,
        verbose=False,
    )
    expected_internal_energy += expected_thermo.get_ideal_translational_energy(temperature)
    expected_internal_energy += expected_thermo.get_ideal_rotational_energy(
        "nonlinear",
        temperature,
    )
    expected_enthalpy = expected_internal_energy + units.kB * temperature
    expected_gibbs = expected_enthalpy - temperature * expected_entropy

    assert isinstance(result, ASEQuasiRRHOThermochemistry)
    assert result.zero_point_energy_eV == pytest.approx(
        expected_thermo.get_ZPE_correction()
    )
    assert result.enthalpy_eV == pytest.approx(expected_enthalpy)
    assert result.entropy_eV_per_K == pytest.approx(expected_entropy)
    assert result.gibbs_free_energy_eV == pytest.approx(expected_gibbs)
    assert result.pressure_Pa == pytest.approx(101325.0)

    spin_result = ase_quasi_rrho_thermochemistry(
        atoms,
        data,
        temperature_K=temperature,
        pressure=pressure,
        geometry="nonlinear",
        symmetrynumber=1,
        spin=0.5,
        potential_energy=potential_energy,
        rotor_cutoff_cm1=50.0,
        frequency_scale=frequency_scale,
        include_electronic_entropy=True,
    )
    assert spin_result.entropy_eV_per_K - result.entropy_eV_per_K == pytest.approx(
        units.kB * np.log(2.0)
    )


@pytest.mark.skipif(MSRRHOThermo is None, reason="ASE MSRRHOThermo is unavailable")
def test_ase_quasi_rrho_uses_xtb_small_imaginary_mode_policy():
    atoms = water()

    class SyntheticVibrations:
        def get_energies(self):
            # Scaling happens before xTB applies the 20 cm^-1 cutoff:
            # 10/30 cm^-1 are retained at scale 0.5, while 60 cm^-1 is dropped.
            return np.array(
                [
                    10.0j * units.invcm,
                    30.0j * units.invcm,
                    60.0j * units.invcm,
                    100.0 * units.invcm,
                    200.0 * units.invcm,
                    300.0 * units.invcm,
                    400.0 * units.invcm,
                ],
                dtype=complex,
            )

        def get_atoms(self):
            return atoms

    result = ase_quasi_rrho_thermochemistry(
        atoms,
        SyntheticVibrations(),
        temperature_K=298.15,
        pressure=101325.0 * units.Pascal,
        geometry="nonlinear",
        symmetrynumber=1,
        spin=0,
        potential_energy=-5.0,
        vib_selection="all",
        frequency_scale=0.5,
    )

    expected = MSRRHOThermo(
        vib_energies=np.array(
            [
                10.0 * units.invcm,
                30.0 * units.invcm,
                100.0 * units.invcm,
                200.0 * units.invcm,
                300.0 * units.invcm,
                400.0 * units.invcm,
            ],
        ),
        atoms=atoms,
        potentialenergy=-5.0,
        tau=50.0,
        nu_scal=0.5,
        treat_int_energy=False,
    )
    assert result.zero_point_energy_eV == pytest.approx(expected.get_ZPE_correction())


def test_ase_quasi_rrho_rejects_invalid_model_parameters():
    atoms = water()
    data = hessian_to_vibrations_data(atoms, positive_hessian(len(atoms)))

    with pytest.raises(ValueError, match="rotor_cutoff_cm1"):
        ase_quasi_rrho_thermochemistry(
            atoms,
            data,
            geometry="nonlinear",
            symmetrynumber=1,
            rotor_cutoff_cm1=-1.0,
        )
    with pytest.raises(ValueError, match="frequency_scale"):
        ase_quasi_rrho_thermochemistry(
            atoms,
            data,
            geometry="nonlinear",
            symmetrynumber=1,
            frequency_scale=0.0,
        )


def test_ase_quasi_rrho_supports_monatomic_ideal_gas_limit():
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]])

    class EmptyVibrations:
        def get_energies(self):
            return np.zeros(3)

        def get_atoms(self):
            return atoms

    result = ase_quasi_rrho_thermochemistry(
        atoms,
        EmptyVibrations(),
        geometry="monatomic",
        symmetrynumber=1,
        potential_energy=-1.0,
    )
    expected = IdealGasThermo(
        vib_energies=np.empty(0),
        geometry="monatomic",
        potentialenergy=-1.0,
        atoms=atoms,
        symmetrynumber=1,
        spin=0,
        vib_selection="all",
    )
    assert result.zero_point_energy_eV == pytest.approx(
        expected.get_ZPE_correction()
    )
    assert result.enthalpy_eV == pytest.approx(
        expected.get_enthalpy(298.15, verbose=False)
    )
