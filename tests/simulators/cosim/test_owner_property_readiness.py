"""Read motion from the owning simulator, including the first admission tick."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scenic.core.vectors import Orientation, Vector
from scenic.simulators.carla import actor_utils
from scenic.simulators.cosim.simulator import CosimSimulation, carla
from scenic.simulators.metsr import bridge_utils as native
from scenic.simulators.metsr.bridge_utils import METSRCosimState
from tests.simulators.cosim.helpers import make_admission_simulation


class Vehicle(SimpleNamespace):
    __hash__ = object.__hash__


@pytest.mark.parametrize("guard", [0, 1, 2])
def test_published_carla_owner_ignores_spawn_guard(guard, monkeypatch):
    simulation = object.__new__(CosimSimulation)
    simulation.metsr_state = METSRCosimState()
    simulation._carla_spawn_property_seeds = {}
    obj = Vehicle(
        name="npc",
        carla_actor_flag=True,
        spawn_guard=guard,
        carlaActor=object(),
        parentOrientation=Orientation.fromEuler(0, 0, 0),
    )
    result = {"source": "CARLA actual motion"}
    monkeypatch.setattr(actor_utils, "actor_properties", Mock(return_value=result))
    monkeypatch.setattr(
        native,
        "metsr_properties",
        Mock(side_effect=AssertionError("native cannot supply owned motion")),
    )
    assert simulation.getProperties(obj, {}) is result
    actor_utils.actor_properties.assert_called_once_with(
        obj.carlaActor, obj.parentOrientation
    )


@pytest.mark.parametrize("name", ["npc", "ego"])
def test_native_owner_reads_native_state(name, monkeypatch):
    simulation = object.__new__(CosimSimulation)
    simulation.metsr_state = METSRCosimState()
    simulation._carla_spawn_property_seeds = {}
    obj = Vehicle(
        name=name,
        carla_actor_flag=False,
        spawn_guard=0,
        parentOrientation=Orientation.fromEuler(0, 0, 0),
    )
    result = {"source": "native actual motion"}
    simulation.obj_data_cache = {obj: {}}
    monkeypatch.setattr(native, "metsr_properties", Mock(return_value=result))
    monkeypatch.setattr(
        actor_utils,
        "actor_properties",
        Mock(side_effect=AssertionError("native has no CARLA owner")),
    )
    assert simulation.getProperties(obj, {}) is result


def spawn_fixture(monkeypatch):
    simulation, obj, state, events = make_admission_simulation(snap=True)
    assert simulation.createObjectInCarla(obj, update_velocity=True, native_state=state)
    simulation._commit_promotion(obj, state)
    obj.carlaActor.id = 100
    snapshot = SimpleNamespace(find=Mock(return_value=None))
    simulation.carla_world.get_snapshot = Mock(return_value=snapshot)
    simulation.carla_world.tick = Mock()
    simulation.sim_ticks_per_carla = 1
    monkeypatch.setattr(
        native,
        "metsr_properties",
        Mock(side_effect=AssertionError("native cannot supply owned motion")),
    )
    return simulation, obj, state, snapshot


def test_unpublished_actor_uses_exact_spawn_seed_without_native_or_actor_queries(
    monkeypatch,
):
    simulation, obj, state, snapshot = spawn_fixture(monkeypatch)
    monkeypatch.setattr(
        actor_utils,
        "actor_properties",
        Mock(side_effect=AssertionError("actor proxy not published yet")),
    )
    # Both Scenic's old pose and any later native state are unrelated to the
    # fresh admission snapshot. Neither may replace its grounded spawn pose.
    assert tuple(obj.position) == (-999, 999, 0)
    state.update(x=999, y=888, z=777, bearing=0, speed=99)
    result = simulation.getProperties(obj, {})
    assert tuple(result["position"]) == pytest.approx((12.5, -23.5, 10.5))
    assert result["elevation"] == pytest.approx(10.5)
    assert tuple(result["velocity"]) == pytest.approx((-8, 0, 0), abs=1e-6)
    assert result["speed"] == 8
    assert result["angularSpeed"] == 0
    snapshot.find.assert_called_once_with(100)
    result["position"] = Vector(1, 2, 3)
    assert tuple(simulation.getProperties(obj, {})["position"]) == pytest.approx(
        (12.5, -23.5, 10.5)
    )


def configure_published_actor(obj):
    actor = obj.carlaActor
    actor.get_transform = Mock(
        return_value=carla.Transform(
            carla.Location(0, 0, 0), carla.Rotation(yaw=31, pitch=6, roll=4)
        )
    )
    actor.get_velocity = Mock(return_value=carla.Vector3D(3, 4, 2))
    actor.get_angular_velocity = Mock(return_value=carla.Vector3D(1, 2, 3))


@pytest.mark.parametrize("publication", ["normal_tick", "setup_snapshot"])
def test_first_published_tick_uses_actual_vectors_and_orientation_even_at_zero_pose(
    publication, monkeypatch
):
    simulation, obj, state, snapshot = spawn_fixture(monkeypatch)
    configure_published_actor(obj)
    if publication == "normal_tick":
        simulation.tick_carla()
        # The normal tick establishes readiness without needing another snapshot.
        simulation.carla_world.get_snapshot = Mock(
            side_effect=AssertionError("already published")
        )
    else:
        # Setup's direct world ticks are visible in the cached snapshot too.
        snapshot.find.return_value = object()
    result = simulation.getProperties(obj, {})
    assert tuple(result["position"]) == (0, 0, 0)
    assert tuple(result["velocity"]) == (3, -4, 2)
    assert result["speed"] == pytest.approx(29**0.5)
    assert abs(result["pitch"]) > 0.01
    assert abs(result["roll"]) > 0.01
    assert abs(result["angularSpeed"]) > 0.01
    assert obj not in simulation._carla_spawn_property_seeds
    native.metsr_properties.assert_not_called()


def test_failed_carla_tick_does_not_publish_spawn_seed(monkeypatch):
    simulation, obj, state, snapshot = spawn_fixture(monkeypatch)
    simulation.carla_world.tick.side_effect = RuntimeError("tick failed")
    with pytest.raises(RuntimeError, match="tick failed"):
        simulation.tick_carla()
    assert obj in simulation._carla_spawn_property_seeds
    result = simulation.getProperties(obj, {})
    assert tuple(result["position"]) == pytest.approx((12.5, -23.5, 10.5))
