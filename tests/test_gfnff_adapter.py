import importlib.util
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np
import pytest
from ase import Atoms

from xtb_ase import GFNFF
from xtb_ase._parsers import parse_engrad
from xtb_ase.gfnff import GFNFFDependencyError


def standalone_gfnff_available() -> bool:
    return importlib.util.find_spec("gfnff") is not None


@pytest.mark.integration
def test_gfnff_adapter_forwards_to_standalone_ase_calculator(tmp_path: Path):
    if not standalone_gfnff_available():
        pytest.skip("install gfnff[ase] to run the GFN-FF integration test")

    atoms = Atoms(
        "OH2",
        positions=np.asarray(
            [[0, 0, 0], [0.7586, 0, 0.5043], [-0.7586, 0, 0.5043]]
        ),
    )
    calc = GFNFF(charge=0, solvent="", printlevel=0)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    assert np.isfinite(energy)
    assert forces.shape == (3, 3)
    assert np.isfinite(forces).all()
    assert stress.shape == (6,)


@pytest.mark.integration
def test_gfnff_adapter_matches_official_xtb_cli(tmp_path: Path):
    if not standalone_gfnff_available():
        pytest.skip("install gfnff[ase] to run the GFN-FF integration test")
    command_value = os.environ.get(
        "GXTB_COMMAND",
        "/tmp/gxtb-v2-research.rW5RzX/extracted/xtb-6.7.1/bin/xtb",
    )
    command = shlex.split(command_value)
    if not command or not Path(command[0]).is_file():
        pytest.skip("set GXTB_COMMAND to run the official xTB comparison")

    atoms = Atoms(
        "OH2",
        positions=np.asarray(
            [[0, 0, 0], [0.7586, 0, 0.5043], [-0.7586, 0, 0.5043]]
        ),
    )
    calc = GFNFF(threads=1)
    atoms.calc = calc
    adapter_energy = atoms.get_potential_energy()
    adapter_forces = atoms.get_forces()

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
            "--gfnff",
            "--grad",
            "--no-restart",
            "--chrg",
            "0",
            "--parallel",
            "1",
        ],
        cwd=direct_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    direct = parse_engrad(direct_dir / "structure.engrad", natoms=3)

    np.testing.assert_allclose(adapter_energy, direct.energy_hartree * 27.211386024367243)
    np.testing.assert_allclose(
        adapter_forces,
        -direct.gradient_hartree_per_bohr
        * 27.211386024367243
        / 0.5291772105638411,
        atol=1e-10,
    )


@pytest.mark.integration
def test_gfnff_adapter_supports_periodic_stress(tmp_path: Path):
    if not standalone_gfnff_available():
        pytest.skip("install gfnff[ase] to run the GFN-FF integration test")

    atoms = Atoms(
        "H2",
        positions=np.asarray([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]]),
        cell=np.eye(3) * 5.0,
        pbc=True,
    )
    atoms.calc = GFNFF(threads=1)

    assert np.isfinite(atoms.get_potential_energy())
    assert atoms.get_forces().shape == (2, 3)
    assert atoms.get_stress().shape == (6,)
    assert np.isfinite(atoms.get_stress()).all()


@pytest.mark.integration
def test_gfnff_adapter_invalidates_cache_when_atoms_info_charge_changes():
    if not standalone_gfnff_available():
        pytest.skip("install gfnff[ase] to run the GFN-FF integration test")

    atoms = Atoms(
        "OH2",
        positions=np.asarray(
            [[0, 0, 0], [0.7586, 0, 0.5043], [-0.7586, 0, 0.5043]]
        ),
    )
    atoms.info["charge"] = 0
    atoms.calc = GFNFF(threads=1)
    neutral = atoms.get_potential_energy()

    atoms.info["charge"] = 1
    charged = atoms.get_potential_energy()

    assert charged != pytest.approx(neutral)


def test_gfnff_adapter_has_actionable_missing_dependency_error(monkeypatch):
    from xtb_ase import gfnff

    monkeypatch.setattr(gfnff, "_load_standalone_gfnff", lambda: (_ for _ in ()).throw(
        GFNFFDependencyError("install xtb-ase[gfnff]")
    ))
    calc = GFNFF()
    atoms = Atoms("H", positions=np.asarray([[0, 0, 0]]))

    with pytest.raises(GFNFFDependencyError, match=r"install xtb-ase\[gfnff\]"):
        calc.calculate(atoms)


def test_gfnff_rejects_electronic_structure_only_parameters():
    for name in ("uhf", "spin", "unpaired_electrons", "etemp", "electronic_temperature"):
        with pytest.raises(TypeError, match="does not support"):
            GFNFF(**{name: 1})

    with pytest.raises(TypeError, match="CalculatorPool"):
        GFNFF(processes=2)


def test_gfnff_validates_python_thread_setting():
    with pytest.raises(ValueError, match="threads"):
        GFNFF(threads=0)


def test_gfnff_configures_threads_before_backend_creation(monkeypatch):
    from xtb_ase import gfnff

    events = []

    def set_threads(value):
        events.append(("threads", value))

    class FakeBackend:
        def __init__(self, **kwargs):
            events.append(("backend", kwargs))
            self.results = {}

        def calculate(self, atoms, properties, system_changes):
            self.results = {
                "energy": 0.0,
                "forces": np.zeros((len(atoms), 3)),
                "stress": np.zeros(6),
            }

    monkeypatch.setattr(gfnff, "_set_native_threads", set_threads)
    monkeypatch.setattr(gfnff, "_load_standalone_gfnff", lambda: FakeBackend)

    atoms = Atoms("H", positions=np.asarray([[0.0, 0.0, 0.0]]))
    GFNFF(threads=4).calculate(atoms)

    assert events[0] == ("threads", 4)
    assert events[1][0] == "backend"


def test_gfnff_environment_overlay_is_restored(monkeypatch):
    from xtb_ase import gfnff

    seen = []

    class FakeBackend:
        def __init__(self, **kwargs):
            seen.append(os.environ.get("XTBASE_GFNFF_ENV"))
            self.results = {}

        def calculate(self, atoms, properties, system_changes):
            self.results = {
                "energy": 0.0,
                "forces": np.zeros((len(atoms), 3)),
                "stress": np.zeros(6),
            }

    monkeypatch.delenv("XTBASE_GFNFF_ENV", raising=False)
    monkeypatch.setattr(gfnff, "_load_standalone_gfnff", lambda: FakeBackend)

    atoms = Atoms("H", positions=np.asarray([[0.0, 0.0, 0.0]]))
    GFNFF(env={"XTBASE_GFNFF_ENV": "worker-local"}).calculate(atoms)

    assert seen == ["worker-local"]
    assert "XTBASE_GFNFF_ENV" not in os.environ
