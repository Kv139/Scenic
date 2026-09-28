"""Small simulation and protocol doubles shared by co-simulation tests."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

from scenic.core.vectors import Orientation, Vector
from scenic.simulators.cosim.simulator import CosimSimulation
from scenic.simulators.metsr.bridge_utils import METSRCosimState
from scenic.simulators.metsr.network_utils import NetworkMetadata


class Vehicle(SimpleNamespace):
    __hash__ = object.__hash__


def make_simulation():
    simulation = object.__new__(CosimSimulation)
    simulation.metsr_state = METSRCosimState(metadata=NetworkMetadata())
    simulation.metsr_client = SimpleNamespace()
    simulation.ego = None
    simulation.map = None
    simulation.count = 0
    simulation.carla_actors = []
    simulation.metsr_actors = []
    simulation.boundary_actors = {}
    simulation.bubble_spawn_queue = set()
    simulation.completed_route = {}
    simulation._carla_destination_plans = {}
    simulation._carla_spawn_property_seeds = {}
    simulation._pre_actor_teardown_callbacks = []
    simulation.bubble_topology = SimpleNamespace(roads_by_id={})
    simulation.tm = SimpleNamespace(set_path=Mock())
    simulation.carla_world = SimpleNamespace(
        get_actors=lambda: SimpleNamespace(filter=lambda pattern: []),
        get_actor=lambda actor_id: SimpleNamespace(is_alive=True),
    )
    return simulation


def metsr_response(message_type, data=(), status="ok", **fields):
    """Build one current native METS-R response envelope for test doubles."""
    response = {
        "messageType": message_type,
        "status": status,
        "data": list(data),
    }
    response.update(fields)
    if message_type == "teleportCoSimVeh":
        for record in response["data"]:
            if isinstance(record, dict):
                record.setdefault("controlMode", "cosim")
                record.setdefault("releasedFromCoSim", record["controlMode"] == "native")
    return response


def make_admission_simulation(*, snap=False):
    obj = type("Vehicle", (), {})()
    obj.name = "fresh_admission"
    obj.carla_actor_flag = False
    obj.blueprint = "vehicle.test"
    obj.rolename = None
    obj.color = None
    obj.physics = True
    obj.snapToGround = snap
    obj.position = Vector(-999, 999, 0)
    obj.parentOrientation = Orientation.fromEuler(math.pi / 2, 0, 0)
    obj.yaw = math.pi / 2
    obj.orientation = obj.parentOrientation * Orientation.fromEuler(obj.yaw, 0, 0)
    obj.velocity = Vector(99, 99, 0)
    actor = SimpleNamespace(
        set_simulate_physics=lambda enabled: None,
        set_target_velocity=lambda velocity: events.append(("velocity", velocity)),
    )
    events = []

    def spawn(blueprint, transform):
        events.append(("spawn", transform))
        return actor

    def waypoint(location):
        events.append(("ground_query", (location.x, location.y, location.z)))
        return SimpleNamespace(
            transform=SimpleNamespace(
                location=SimpleNamespace(z=10.0 if location.z > 5.0 else 0.0)
            )
        )

    simulation = make_simulation()
    simulation.blueprintLib = SimpleNamespace(
        find=lambda name: SimpleNamespace(has_attribute=lambda name: False)
    )
    simulation.carla_world = SimpleNamespace(
        try_spawn_actor=spawn,
        get_map=lambda: SimpleNamespace(get_waypoint=waypoint),
    )
    simulation.bubble_spawn_queue = set()
    simulation._carla_spawn_property_seeds = {}
    state = {"x": 12.5, "y": -23.5, "z": 10.0, "bearing": 270.0, "speed": 8.0}
    return simulation, obj, state, events
