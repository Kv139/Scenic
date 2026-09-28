"""Shared actor mechanics need no running CARLA server."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

carla = pytest.importorskip("carla")

from scenic.core.simulators import SimulationCreationError
from scenic.core.vectors import Orientation
from scenic.simulators.carla import actor_utils as actors
from scenic.simulators.carla.simulator import CarlaSimulation


class Blueprint:
    def __init__(self, attributes=("color", "is_invincible")):
        self.supported = set(attributes)
        self.attributes = {}

    def has_attribute(self, name):
        return name in self.supported

    def set_attribute(self, name, value):
        self.attributes[name] = value


def test_legacy_blueprint_preserves_attributes_and_updates_model():
    blueprint = Blueprint()
    library = SimpleNamespace(find=Mock(side_effect=[IndexError("missing"), blueprint]))
    obj = SimpleNamespace(
        blueprint="vehicle.lincoln.mkz_2017",
        rolename="hero",
        color=SimpleNamespace(r=1, g=0.5, b=0),
    )

    with pytest.warns(UserWarning, match="using older version vehicle.lincoln.mkz2017"):
        assert actors.prepare_actor_blueprint(obj, library) is blueprint

    assert obj.blueprint == "vehicle.lincoln.mkz2017"
    assert [call.args[0] for call in library.find.call_args_list] == [
        "vehicle.lincoln.mkz_2017",
        "vehicle.lincoln.mkz2017",
    ]
    assert blueprint.attributes == {
        "role_name": "hero",
        "is_invincible": "False",
        "color": "255,127,0",
    }


def test_missing_blueprint_retains_original_lookup_failure():
    failure = IndexError("missing current model")
    library = SimpleNamespace(find=Mock(side_effect=[failure, IndexError("missing old")]))
    obj = SimpleNamespace(blueprint="vehicle.lincoln.mkz_2017")

    with pytest.raises(
        SimulationCreationError, match="Unable to find blueprint"
    ) as error:
        actors.prepare_actor_blueprint(obj, library)

    assert error.value.__cause__ is failure
    assert obj.blueprint == "vehicle.lincoln.mkz_2017"


def test_unsupported_blueprint_attributes_are_not_written():
    blueprint = Blueprint(attributes=())
    obj = SimpleNamespace(
        blueprint="walker.test", rolename=None, color=SimpleNamespace(r=1, g=1, b=1)
    )
    assert (
        actors.prepare_actor_blueprint(obj, SimpleNamespace(find=lambda _: blueprint))
        is blueprint
    )
    assert blueprint.attributes == {}


def test_actor_properties_convert_world_motion_and_parent_orientation():
    actor = SimpleNamespace(
        get_transform=Mock(
            return_value=carla.Transform(
                carla.Location(10, -20, 3), carla.Rotation(yaw=-90)
            )
        ),
        get_velocity=Mock(return_value=carla.Vector3D(3, -4, 12)),
        get_angular_velocity=Mock(return_value=carla.Vector3D(0, 30, 30)),
    )

    properties = actors.actor_properties(actor, Orientation.fromEuler(math.pi / 2, 0, 0))

    assert tuple(properties["position"]) == (10, 20, 3)
    assert tuple(properties["velocity"]) == (3, 4, 12)
    assert properties["speed"] == 13
    assert properties["elevation"] == 3
    assert properties["yaw"] == pytest.approx(-math.pi / 2)
    assert properties["pitch"] == pytest.approx(0)
    assert properties["roll"] == pytest.approx(0)
    assert tuple(properties["angularVelocity"]) == pytest.approx(
        (0, -math.pi / 6, math.pi / 6)
    )
    assert properties["angularSpeed"] == pytest.approx(math.sqrt(2) * math.pi / 6)
    actor.get_transform.assert_called_once_with()
    actor.get_velocity.assert_called_once_with()
    actor.get_angular_velocity.assert_called_once_with()


@pytest.fixture
def vehicle(monkeypatch):
    class Vehicle:
        def __init__(self):
            self.bounding_box = SimpleNamespace(extent=carla.Vector3D(2, 0, -1))
            self.apply_control = Mock()
            self.set_autopilot = Mock()
            self.set_simulate_physics = Mock()

    monkeypatch.setattr(carla, "Vehicle", Vehicle)
    return Vehicle()


def test_vehicle_dimensions_preserve_invalid_extents_and_initialize_gear(vehicle):
    obj = SimpleNamespace(carlaActor=vehicle, width=1.8, length=3, height=1.5)

    actors.configure_actor(obj, None, None)

    assert (obj.width, obj.length, obj.height) == (1.8, 4, 1.5)
    control = vehicle.apply_control.call_args.args[0]
    assert control.manual_gear_shift
    assert control.gear == 1
    vehicle.set_autopilot.assert_not_called()


@pytest.mark.parametrize("behavior", [None, object()])
def test_standalone_carla_keeps_its_autopilot_policy(vehicle, behavior):
    obj = SimpleNamespace(
        carlaActor=None,
        blueprint="vehicle.test",
        rolename=None,
        color=None,
        position=SimpleNamespace(x=0, y=0, z=0),
        orientation=Orientation.fromEuler(0, 0, 0),
        snapToGround=False,
        physics=True,
        width=1.8,
        length=3,
        height=1.5,
        behavior=behavior,
        sensors={},
    )
    simulation = object.__new__(CarlaSimulation)
    simulation.world = SimpleNamespace(try_spawn_actor=lambda *args: vehicle)
    simulation.blueprintLib = SimpleNamespace(find=lambda _: Blueprint())
    simulation.tm = SimpleNamespace(get_port=lambda: 8000)
    simulation.total_spawned_actors = 0

    assert simulation.createObjectInSimulator(obj) is vehicle

    assert obj.carlaActor is vehicle
    assert simulation.total_spawned_actors == 1
    if behavior is None:
        vehicle.set_autopilot.assert_called_once_with(True, 8000)
    else:
        vehicle.set_autopilot.assert_not_called()


@pytest.mark.parametrize("spawn_succeeds", [False, True])
def test_walker_controller_creation(monkeypatch, spawn_succeeds):
    class Walker:
        apply_control = Mock()

    monkeypatch.setattr(carla, "Walker", Walker)
    obj = SimpleNamespace(carlaActor=Walker())
    controller = object() if spawn_succeeds else None
    world = SimpleNamespace(try_spawn_actor=Mock(return_value=controller))
    library = SimpleNamespace(find=Mock(return_value="controller blueprint"))

    if spawn_succeeds:
        actors.configure_actor(obj, world, library)
        assert obj.carlaController is controller
    else:
        with pytest.raises(
            SimulationCreationError, match="Unable to spawn carla controller"
        ):
            actors.configure_actor(obj, world, library)
        assert not hasattr(obj, "carlaController")
    library.find.assert_called_once_with("controller.ai.walker")
    assert world.try_spawn_actor.call_args.args[2] is obj.carlaActor


@pytest.mark.parametrize("registered", [None, False, True])
def test_controller_is_stopped_before_conditional_destruction(registered):
    events = []
    controller = SimpleNamespace(
        stop=lambda: events.append("stop"), destroy=lambda: events.append("destroy")
    )

    def check_registered(candidate):
        assert candidate is controller
        events.append("check")
        return registered

    actors.destroy_walker_controller(
        controller, is_registered=check_registered if registered is not None else None
    )

    expected = ["stop"]
    if registered is not None:
        expected.append("check")
    if registered is not False:
        expected.append("destroy")
    assert events == expected
