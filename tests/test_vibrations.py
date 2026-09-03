from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.thermochemistry import IdealGasThermo

from xtb_ase.vibrations import (
    ASEVibrationalThermochemistry,
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
