"""Boundary occupancy is visible in CARLA while METS-R retains motion control."""
from types import SimpleNamespace

import pytest

from scenic.core.simulators import SimulationCreationError
from scenic.simulators.cosim.simulator import CosimSimulation, METSRControlError, carla


def response(message_type, data):
    return {"messageType": message_type, "status": "ok", "data": data}


class Blueprint:
    def __init__(self, model):
        self.id = model
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def has_attribute(self, key):
        return key in ("role_name", "color")


class Actor:
    def __init__(self, actor_id, blueprint, transform):
        self.id = actor_id
        self.type_id = blueprint.id
        self.attributes = dict(blueprint.attributes)
        self.transform = transform
        self.bounding_box = carla.BoundingBox(
            carla.Location(0, 0, 0.2), carla.Vector3D(2.4, 1.0, 0.7)
        )
        self.is_alive = True
        self.physics = True
        self.autopilot = []
        self.destroy_calls = 0

    def set_autopilot(self, enabled, port):
        self.autopilot.append((enabled, port))

    def set_simulate_physics(self, enabled):
        self.physics = enabled

    def set_transform(self, transform):
        self.transform = transform

    def destroy(self):
        assert self.is_alive
        self.is_alive = False
        self.destroy_calls += 1
        return True


class World:
    def __init__(self):
        self.actors = []
        self.reject_spawn = False
        self.tick_poses = []

    def try_spawn_actor(self, blueprint, transform):
        if self.reject_spawn:
            return None
        actor = Actor(len(self.actors) + 1, blueprint, transform)
        self.actors.append(actor)
        return actor

    def get_actor(self, actor_id):
        return next((a for a in self.actors if a.id == actor_id and a.is_alive), None)

    def tick(self):
        self.tick_poses.append([
            (a.id, a.transform.location.x, a.physics)
            for a in self.actors if a.is_alive
        ])


class ScenicObject(SimpleNamespace):
    __hash__ = object.__hash__


@pytest.fixture
def boundary_simulation(monkeypatch):
    simulation = object.__new__(CosimSimulation)
    simulation.boundary_actors = {}
    simulation.pv_id_map = {}
    simulation.objects = []
    simulation.carla_actors = []
    simulation.metsr_actors = []
    simulation.carla_control_roads = {}
    simulation.map = SimpleNamespace(get_waypoint=lambda location, **kwargs: SimpleNamespace(
        transform=carla.Transform(carla.Location(location.x, location.y, 3.0))
    ))
    simulation.carla_world = World()
    simulation.tm = SimpleNamespace(get_port=lambda: 8000)
    available = {"vehicle.test.car", "vehicle.test.bus", "vehicle.test.scenic"}

    def find(model):
        if model not in available:
            raise IndexError(model)
        return Blueprint(model)

    simulation.blueprintLib = SimpleNamespace(find=find)
    monkeypatch.setattr("scenic.simulators.cosim.simulator.carModels", ["vehicle.test.car"])
    monkeypatch.setattr("scenic.simulators.cosim.simulator.busModels", ["vehicle.test.bus"])
    records = [{
        "vehicleId": 42, "isPrivate": True,
        "segmentId": "exit", "segmentType": "road", "controlMode": "native",
        "onRoad": True, "onConnector": False, "laneIndex": 0,
        # These are upcoming route vertices, deliberately far from the vehicle.
        "coordinateTrail": [[900.0, 800.0], [901.0, 801.0]],
    }]
    states = {(True, "42"): {
        "vehicleId": 42, "vehicleClass": 3,
        "x": 10.0, "y": 20.0, "z": None, "bearing": 90.0, "speed": 8.0,
    }}
    queries = []

    def query_vehicle(ids, private_veh, transform_coords):
        queries.append((ids, private_veh, transform_coords))
        return response("vehicle", [states[(private, str(vid))]
                                    for vid, private in zip(ids, private_veh)])

    simulation.metsr_client = SimpleNamespace(
        query_boundary_vehicle=lambda: response("boundaryVehicle", records),
        query_vehicle=query_vehicle,
    )
    return simulation, records, states, queries


def test_current_pose_is_visible_as_static_vehicle_across_carla_substeps(boundary_simulation):
    simulation, _, states, queries = boundary_simulation
    simulation._synchronize_boundary_vehicles()
    actor = simulation.boundary_actors[(True, "42")]
    assert actor.type_id.startswith("vehicle.")
    assert actor.attributes["role_name"] == "metsr_boundary_private_42"
    assert actor.physics is False
    assert actor.autopilot == [(False, 8000)]
    assert actor.transform.location.x == pytest.approx(10.0)
    assert actor.transform.location.y == pytest.approx(-20.0)
    # Place the bottom on the road; physics is disabled so gravity cannot settle it.
    assert actor.transform.location.z == pytest.approx(3.55)
    assert actor.transform.rotation.yaw == pytest.approx(0.0)
    assert queries == [([42], [True], True)]
    assert simulation.objects == simulation.carla_actors == simulation.metsr_actors == []
    assert simulation.pv_id_map == simulation.carla_control_roads == {}

    states[(True, "42")].update(x=12.0, bearing=180.0)
    simulation.sim_ticks_per_carla = 3
    simulation.tick_carla()
    assert simulation.carla_world.tick_poses == [[(actor.id, 10.0, False)]] * 3
    simulation._synchronize_boundary_vehicles()
    assert simulation.boundary_actors[(True, "42")] is actor
    assert len(simulation.carla_world.actors) == 1
    assert actor.transform.location.x == pytest.approx(12.0)
    assert actor.transform.rotation.yaw == pytest.approx(90.0)
    assert states[(True, "42")]["speed"] == 8.0


def test_vehicle_leaving_boundary_is_removed_without_an_extra_pose_query(boundary_simulation):
    simulation, records, _, queries = boundary_simulation
    simulation._synchronize_boundary_vehicles()
    actor = simulation.boundary_actors[(True, "42")]
    records.clear()
    simulation._synchronize_boundary_vehicles()
    simulation._synchronize_boundary_vehicles()
    assert simulation.boundary_actors == {}
    assert actor.destroy_calls == 1
    assert len(queries) == 1


def test_road_takeover_removes_obstacle_before_dynamic_promotion(boundary_simulation):
    simulation, _, _, queries = boundary_simulation
    simulation._synchronize_boundary_vehicles()
    actor = simulation.boundary_actors[(True, "42")]
    # Defensively handle even an old boundary snapshot during road takeover.
    simulation.carla_control_roads = {"exit": True}
    simulation._synchronize_boundary_vehicles()
    assert simulation.boundary_actors == {}
    assert actor.destroy_calls == 1
    assert len(queries) == 1


def test_existing_dynamic_vehicle_is_never_duplicated(boundary_simulation):
    simulation, _, _, queries = boundary_simulation
    dynamic_actor = object()
    obj = ScenicObject(carlaActor=dynamic_actor)
    simulation.pv_id_map[obj] = 42
    simulation.carla_actors = [obj]
    simulation._synchronize_boundary_vehicles()
    assert simulation.boundary_actors == {}
    assert queries == []
    assert obj.carlaActor is dynamic_actor


def test_private_and_public_ids_and_native_buses_are_distinct(boundary_simulation):
    simulation, records, states, queries = boundary_simulation
    records.append(dict(records[0], isPrivate=False))
    states[(False, "42")] = dict(states[(True, "42")], vehicleClass=2, x=25.0)
    simulation._synchronize_boundary_vehicles()
    assert len(simulation.boundary_actors) == 2
    private = simulation.boundary_actors[(True, "42")]
    public = simulation.boundary_actors[(False, "42")]
    assert private.type_id == "vehicle.test.car"
    assert public.type_id == "vehicle.test.bus"
    assert public.attributes["role_name"] == "metsr_boundary_public_42"
    assert public.transform.location.x == pytest.approx(25.0)
    assert queries == [([42, 42], [True, False], True)]


def test_known_scenic_model_and_elevation_are_preserved(boundary_simulation):
    simulation, _, states, _ = boundary_simulation
    obj = ScenicObject(
        carlaActor=None, blueprint="vehicle.test.scenic", snapToGround=False,
        color=SimpleNamespace(r=1.0, g=0.0, b=0.0), carla_actor_flag=False,
    )
    simulation.pv_id_map[obj] = 42
    states[(True, "42")]["z"] = 7.0
    simulation._synchronize_boundary_vehicles()
    actor = simulation.boundary_actors[(True, "42")]
    assert actor.type_id == obj.blueprint
    assert actor.transform.location.z == pytest.approx(7.0)
    assert actor.attributes["color"] == "255,0,0"
    assert obj.carlaActor is None
    assert obj.carla_actor_flag is False


def test_duplicate_membership_creates_one_obstacle(boundary_simulation):
    simulation, records, _, queries = boundary_simulation
    records.append(dict(records[0]))
    simulation._synchronize_boundary_vehicles()
    assert len(simulation.boundary_actors) == 1
    assert queries == [([42], [True], True)]


@pytest.mark.parametrize("bad_response", [
    None,
    {"messageType": "boundaryVehicle", "status": "error", "data": []},
    {"messageType": "boundaryVehicle", "status": "partial", "data": []},
    {"messageType": "boundaryVehicle", "status": "ok"},
])
def test_query_failure_does_not_clear_existing_blockers(boundary_simulation, bad_response):
    simulation, _, _, _ = boundary_simulation
    simulation._synchronize_boundary_vehicles()
    actor = simulation.boundary_actors[(True, "42")]
    simulation.metsr_client.query_boundary_vehicle = lambda: bad_response
    with pytest.raises(METSRControlError):
        simulation._synchronize_boundary_vehicles()
    assert simulation.boundary_actors[(True, "42")] is actor
    assert actor.is_alive


@pytest.mark.parametrize("changes", [
    {"x": float("nan")}, {"bearing": None}, {"vehicleId": 99}, {"vehicleClass": 2},
])
def test_invalid_pose_does_not_move_or_remove_an_obstacle(boundary_simulation, changes):
    simulation, _, states, _ = boundary_simulation
    simulation._synchronize_boundary_vehicles()
    actor = simulation.boundary_actors[(True, "42")]
    states[(True, "42")].update(changes)
    with pytest.raises(METSRControlError):
        simulation._synchronize_boundary_vehicles()
    assert actor.is_alive
    assert actor.transform.location.x == pytest.approx(10.0)


def test_rejected_spawn_reports_missing_obstacle(boundary_simulation):
    simulation, _, _, _ = boundary_simulation
    simulation.carla_world.reject_spawn = True
    with pytest.raises(SimulationCreationError, match="could not spawn METS-R boundary"):
        simulation._synchronize_boundary_vehicles()
    assert simulation.boundary_actors == {}
    assert simulation.carla_world.tick_poses == []


def test_externally_destroyed_obstacle_is_recreated(boundary_simulation):
    simulation, _, _, _ = boundary_simulation
    simulation._synchronize_boundary_vehicles()
    actor = simulation.boundary_actors[(True, "42")]
    actor.is_alive = False
    simulation._synchronize_boundary_vehicles()
    assert simulation.boundary_actors[(True, "42")] is not actor
    assert actor.destroy_calls == 0


def test_failed_obstacle_configuration_remains_tracked_for_cleanup(boundary_simulation, monkeypatch):
    simulation, _, _, _ = boundary_simulation

    def reject_physics(self, enabled):
        raise RuntimeError("CARLA configuration failed")

    monkeypatch.setattr(Actor, "set_simulate_physics", reject_physics)
    with pytest.raises(RuntimeError, match="CARLA configuration failed"):
        simulation._synchronize_boundary_vehicles()
    assert len(simulation.boundary_actors) == 1
    simulation._remove_boundary_actor((True, "42"))
    assert simulation.carla_world.actors[0].destroy_calls == 1
