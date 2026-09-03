import numpy as np
import pytest
from ase import Atoms

from xtb_ase.search.pamssw import PAMSSWComparisonConfig, run_pamssw


def test_pamssw_config_rejects_nonpositive_uphill_height():
    with pytest.raises(ValueError, match="target_uphill_energy"):
        PAMSSWComparisonConfig(target_uphill_energy_eV=0.0)


def test_pamssw_adapter_maps_state_and_archive(monkeypatch):
    import xtb_ase.search.pamssw as module

    seen = {}

    class FakeState:
        def __init__(self, **kwargs):
            seen.setdefault("states", []).append(kwargs)
            self.numbers = np.asarray(kwargs["numbers"], dtype=int)
            self.positions = np.asarray(kwargs["positions"], dtype=float)
            self.cell = kwargs.get("cell")
            self.pbc = tuple(kwargs.get("pbc", (False, False, False)))
            self.metadata = dict(kwargs.get("metadata", {}))

    class FakeConfig:
        def __init__(self, **kwargs):
            seen["config"] = kwargs

    class FakeASECalculator:
        def __init__(self, calculator):
            seen["wrapped_calculator"] = calculator

    class FakeEntry:
        def __init__(self, state, energy):
            self.state = state
            self.energy = energy

    class FakeResult:
        def __init__(self, best_state):
            self.best_state = best_state
            self.best_energy = -1.0
            self.archive = type(
                "Archive",
                (),
                {"entries": [FakeEntry(best_state, -1.0)]},
            )()

    best_state = FakeState(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))

    def fake_run_ssw(state, calculator, config):
        seen["run"] = (state, calculator, config)
        return FakeResult(best_state)

    monkeypatch.setattr(
        module,
        "_load_pamssw",
        lambda: (FakeState, FakeConfig, fake_run_ssw, FakeASECalculator),
    )
    calculator = object()
    result = run_pamssw(Atoms("H", positions=[[0, 0, 0]]), calculator)

    assert seen["config"]["target_uphill_energy"] == pytest.approx(0.05)
    assert seen["wrapped_calculator"] is calculator
    assert result.best_energy_eV == pytest.approx(-1.0)
    assert len(result.minima) == 1
    assert result.minima[0].calc is None


def test_pamssw_adapter_reports_missing_optional_dependency(monkeypatch):
    import xtb_ase.search.pamssw as module

    def missing():
        raise ImportError("pamssw is unavailable")

    monkeypatch.setattr(module, "_load_pamssw", missing)
    with pytest.raises(ImportError, match="pamssw"):
        run_pamssw(Atoms("H", positions=[[0, 0, 0]]), object())


@pytest.mark.integration
def test_pamssw_adapter_smoke_with_installed_backend():
    pytest.importorskip("pamssw")
    from ase.calculators.lj import LennardJones

    result = run_pamssw(
        Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]]),
        LennardJones(),
        PAMSSWComparisonConfig(
            target_uphill_energy_eV=0.05,
            max_trials=1,
            max_steps_per_walk=1,
            max_force_evals=120,
            rng_seed=7,
        ),
    )

    assert np.isfinite(result.best_energy_eV)
    assert result.minima
    assert result.archive_energies_eV.ndim == 1
    assert all(atoms.calc is None for atoms in result.minima)
