"""CARLA publishes poses without required hints; native handback is acknowledged."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import pytest

from scenic.core.simulators import SimulationCreationError
from scenic.simulators.carla.utils import actors as actor_utils, routing as carla_routing
from scenic.simulators.cosim import cosim_utils
from scenic.simulators.cosim.simulator import carla
from scenic.simulators.cosim.utils.global_route_planner import GlobalRoutePlanner
from scenic.simulators.metsr import (
    cosim_routing_utils as routing,
    cosim_utils as native,
    network as metsr_network,
)
from scenic.simulators.metsr.cosim_utils import METSRCosimState
from tests.simulators.cosim.helpers import make_simulation


class Vehicle(SimpleNamespace):
    __hash__ = object.__hash__


@pytest.fixture(scope="module")
def town05(getAssetPath):
    xml = getAssetPath("maps/CARLA/Town05.net.xml")
    network = ET.parse(xml).getroot()
    metadata = metsr_network.load_network_metadata(xml)
    world_map = carla.Map(
        "Town05", Path(getAssetPath("maps/CARLA/Town05.xodr")).read_text()
    )
    return (
        network,
        world_map,
        metadata.lane_mapping,
        metadata.lane_indices,
    )


@pytest.fixture
def connector_route(town05):
    network, world_map, mapping, lane_indices = town05

    def build(source="50", target="-2"):
        simulation = make_simulation()
        simulation.metsr_state = METSRCosimState()
        simulation.map = world_map
        simulation.grp = GlobalRoutePlanner(world_map, sampling_resolution=2.0)
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, lane_mapping=mapping
        )
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, lane_indices=lane_indices
        )
        simulation._carla_waypoints_by_key = None
        simulation._carla_waypoints_by_key = carla_routing.index_carla_waypoints(
            simulation.map, simulation.grp
        )
        offset = [
            float(value) for value in network.find("location").get("netOffset").split(",")
        ]
        simulation.metsr_state.road_cache = {}
        for edge in network.findall("edge"):
            for lane in edge.findall("lane"):
                lane_id = lane.get("id")
                road, index = metsr_network.mapped_lane_query(
                    simulation.metsr_state, lane_id
                )
                if index is None:
                    continue
                points = []
                for raw in lane.get("shape").split():
                    point = [float(value) for value in raw.split(",")]
                    points.append(
                        (
                            point[0] - offset[0],
                            point[1] - offset[1],
                            point[2] if len(point) > 2 else 0.0,
                        )
                    )
                simulation.metsr_state.road_cache[road, index] = points
        connections = [
            item
            for item in network.findall("connection")
            if item.get("from") == source and item.get("to") == target
        ]
        connections = [
            item
            for item in connections
            if any(
                key in simulation._carla_waypoints_by_key
                for key in simulation.metsr_state.metadata.carla_keys_by_lane.get(
                    item.get("via"), ()
                )
            )
        ]
        assert connections
        # Opaque ID deliberately does not encode the road pair.
        connector = "occupied-maneuver"
        paths = [
            {
                "connectorId": connector,
                "connectorPathId": index,
                "sourceRoadId": source,
                "targetRoadId": target,
                "sourceLaneId": f"{source}_{item.get('fromLane')}",
                "targetLaneId": f"{target}_{item.get('toLane')}",
                "viaLaneIds": [item.get("via")],
            }
            for index, item in enumerate(connections)
        ]
        simulation.metsr_state.controlled_roads = {target: True}
        simulation.metsr_state.connector_records = {
            connector: {
                "connectorId": connector,
                "sourceRoadId": source,
                "targetRoadId": target,
                "paths": paths,
            }
        }
        simulation.metsr_state.road_connectors = {target: {connector}}
        simulation.metsr_state.connector_path_cache = {connector: paths}
        simulation.metsr_client = SimpleNamespace(
            query_centerline=Mock(side_effect=AssertionError("Use cached map geometry")),
            query_connector_path=Mock(
                side_effect=AssertionError("Use canonical path records")
            ),
        )
        key = sorted(
            simulation.metsr_state.metadata.carla_keys_by_lane[paths[0]["viaLaneIds"][0]]
        )[0]
        points = simulation._carla_waypoints_by_key[key]
        start = points[len(points) // 2].transform.location
        obj = Vehicle(
            name="car_53",
            route=["NATIVE_ROUTE_MUST_NOT_CONSTRAIN_CARLA"],
            carlaActor=SimpleNamespace(get_location=lambda: start),
        )
        simulation.ego = object()
        simulation.metsr_state.authoritative_segments = {obj: connector}
        state = {
            "onConnector": True,
            "segmentType": "connector",
            "segmentId": connector,
            "connectorId": connector,
            "connectorPathId": 0,
            "destinationRoadId": "-4",
            "x": start.x,
            "y": -start.y,
            "z": start.z,
        }
        return simulation, obj, state, paths

    return build


def mapped_lanes(simulation, location):
    point = simulation.map.get_waypoint(location)
    return simulation.metsr_state.metadata.lane_mapping.get(
        f"{point.road_id}_{point.lane_id}", ()
    )


@pytest.mark.parametrize(
    "xyz",
    [
        (-183.93374633789062, -99.63856506347656, 0.021136073395609856),  # car90
        (-124.84190368652344, 1.7761878967285156, 0.06740444153547287),  # car53
        (-187.240966796875, -7.059037208557129, 0.5407504439353943),  # car9
        (-182.840424, 95.930557, 0.051330),  # crossing observation
        (-1000, -1000, 20),  # no usable native road at all
        (0, 0, 0),  # a zero coordinate is still an authoritative pose
    ],
)
@pytest.mark.parametrize("pcla_ego", [False, True])
def test_scenic_publishes_actual_pose_without_segment_selectors(
    connector_route, xyz, pcla_ego
):
    simulation, obj, state, _ = connector_route()
    location = carla.Location(*xyz)
    obj.carlaActor = SimpleNamespace(
        is_alive=True,
        get_location=lambda: location,
        get_transform=lambda: carla.Transform(location, carla.Rotation(yaw=37)),
        get_velocity=lambda: carla.Vector3D(3, 4, 0),
        apply_control=Mock(
            side_effect=AssertionError("native shadow cannot control CARLA")
        ),
        set_transform=Mock(side_effect=AssertionError("native shadow cannot move CARLA")),
    )
    simulation.count = 1
    simulation.carla_actors = [obj]
    simulation.completed_route = {}
    simulation.metsr_state.controlled_roads = {
        road: True for road, _ in simulation.metsr_state.road_cache
    }
    simulation.getMetsrPrivateVehId = lambda target: 91
    if pcla_ego:
        simulation.ego = obj
    simulation.remove_bubble_object = Mock(
        side_effect=AssertionError("pose update must retain owner")
    )
    calls = []

    def publish(vehicle_id, x, y, **kwargs):
        calls.append((vehicle_id, x, y, kwargs))
        assert "pose_only" not in kwargs
        assert not {"segment_id", "lane_index", "connector_path_id"} & kwargs.keys()
        return {
            "messageType": "teleportCoSimVeh",
            "status": "ok",
            "data": [
                {
                    "vehicleId": vehicle_id,
                    "status": "ok",
                    "controlMode": "cosim",
                    "releasedFromCoSim": False,
                    "segmentId": "50",
                    "segmentType": "road",
                }
            ],
        }

    simulation.metsr_client.query_cosim_vehicle = lambda: {
        "messageType": "coSimVehicle",
        "status": "ok",
        "data": [],
    }
    simulation.metsr_client.teleport_cosim_vehicle = publish
    simulation.metsr_client.reach_dest = Mock(
        side_effect=AssertionError("no native completion")
    )
    # A deliberately contradictory native route and connector cannot veto pose.
    state.update(
        segmentId="native-unrelated",
        connectorId="unknown-connector",
        connectorPathId=999,
        destinationRoadId="not-here",
    )
    simulation.synchronize_clients(vehicle_data={obj: state})
    assert len(calls) == 1
    _, x, y, kwargs = calls[0]
    assert (x, y, kwargs["z"]) == pytest.approx((location.x, -location.y, location.z))
    assert kwargs["speed"] == 5
    assert obj.route == ["NATIVE_ROUTE_MUST_NOT_CONSTRAIN_CARLA"]
    assert simulation.carla_actors == [obj]
    obj.carlaActor.apply_control.assert_not_called()
    obj.carlaActor.set_transform.assert_not_called()


def test_destination_routing_starts_at_actual_pose_and_ignores_native_turn(
    connector_route,
):
    simulation, obj, state, _ = connector_route()
    start = simulation.map.get_waypoint_xodr(994, 1, 23.8).transform.location
    trace = Mock(wraps=simulation.grp.trace_route)
    simulation.grp.trace_route = trace
    state.update(connectorId="missing", segmentId="missing", connectorPathId=999)
    simulation.metsr_state.authoritative_segments[obj] = "nonexistent-native-connector"
    path = routing.generate_carla_destination_trajectory(
        simulation, "-4", obj, target_start=start
    )
    assert path
    assert all(call.args[0].distance(start) < 1e-6 for call in trace.call_args_list)
    # CARLA's own planner may choose the straight sibling; it is no longer
    # rewritten to make it obey the native left-turn connector.
    assert "49_4" in mapped_lanes(simulation, path[0])
    assert obj.route == ["NATIVE_ROUTE_MUST_NOT_CONSTRAIN_CARLA"]


def publication_fixture(*, target=None):
    simulation = make_simulation()
    simulation.map = None
    simulation.metsr_state = METSRCosimState()
    obj = Vehicle(name="npc", carla_actor_flag=True, spawn_guard=0)
    obj.carlaActor = SimpleNamespace(apply_control=Mock(), set_transform=Mock())
    simulation.ego = object()
    simulation.objects = [simulation.ego, obj]
    simulation.count = 10
    simulation.carla_actors = [obj]
    simulation.metsr_state.controlled_roads = {"controlled": True}
    simulation.completed_route = {}
    simulation.metsr_state.private_vehicle_ids = {obj: 91}
    simulation.metsr_state.authoritative_segments = {obj: "controlled"}
    update = {
        "object": obj,
        "vehicle_id": 91,
        "handoff": target,
        "x": -183.9,
        "y": 99.6,
        "z": 0.02,
        "bearing": 37,
        "speed": 8,
        "private_veh": True,
        "transform_coords": True,
        "vehicle_state": {"destinationRoadId": "Z"},
        "cosim_record": {},
    }
    simulation.remove_bubble_object = Mock(
        side_effect=lambda target: simulation.carla_actors.remove(target)
    )
    return simulation, obj, update


def ack(vehicle_id=91, **fields):
    # Fields emitted by ControlMessageHandler.teleportCoSimVeh (20d3a12).
    return {
        "messageType": "teleportCoSimVeh",
        "status": "ok",
        "data": [
            {
                "vehicleId": vehicle_id,
                "status": "ok",
                "controlMode": "cosim",
                "releasedFromCoSim": False,
                "segmentId": "controlled",
                "segmentType": "road",
                "laneIndex": 1,
                **fields,
            }
        ],
    }


def test_ordinary_update_accepts_native_reply_without_new_protocol_fields():
    simulation, obj, update = publication_fixture()
    simulation.metsr_client = SimpleNamespace(
        teleport_cosim_vehicle=Mock(return_value=ack())
    )
    simulation._publish_carla_vehicle_updates([update])
    simulation.metsr_client.teleport_cosim_vehicle.assert_called_once_with(
        91,
        -183.9,
        99.6,
        z=0.02,
        bearing=37,
        speed=8,
        private_veh=True,
        transform_coords=True,
    )
    simulation.remove_bubble_object.assert_not_called()
    assert simulation.metsr_state.authoritative_segments[obj] == "controlled"


@pytest.mark.parametrize("target", [("outside", 1, None), ("connector-outside", None, 2)])
def test_handoff_is_one_request_and_removal_follows_native_ack(target):
    simulation, obj, update = publication_fixture(target=target)

    def teleport(*args, **kwargs):
        simulation.remove_bubble_object.assert_not_called()
        assert kwargs["segment_id"] == target[0]
        if target[1] is not None:
            assert kwargs["lane_index"] == target[1]
            assert "connector_path_id" not in kwargs
        else:
            assert kwargs["connector_path_id"] == target[2]
            assert "lane_index" not in kwargs
        return ack(
            controlMode="native",
            releasedFromCoSim=True,
            segmentId=target[0],
            laneIndex=target[1],
            connectorPathId=target[2],
        )

    simulation.metsr_client = SimpleNamespace(
        teleport_cosim_vehicle=Mock(side_effect=teleport)
    )
    simulation._publish_carla_vehicle_updates([update])
    assert simulation.metsr_client.teleport_cosim_vehicle.call_count == 1
    simulation.remove_bubble_object.assert_called_once_with(obj)


@pytest.mark.parametrize(
    "code,target",
    [
        ("NO_MAP_MATCH", None),
        ("NATIVE_HANDOFF_FAILED", ("outside", 1, None)),
    ],
)
def test_native_rejection_aborts_without_changing_carla_actor(code, target):
    from scenic.simulators.metsr.cosim_utils import METSRControlError

    simulation, obj, update = publication_fixture(target=target)
    failure = {
        "messageType": "teleportCoSimVeh",
        "status": "error",
        "data": [{"vehicleId": 91, "status": "error", "errorCode": code}],
    }
    simulation.metsr_client = SimpleNamespace(
        teleport_cosim_vehicle=Mock(return_value=failure)
    )
    with pytest.raises(METSRControlError):
        simulation._publish_carla_vehicle_updates([update])
    assert simulation.carla_actors == [obj]
    simulation.remove_bubble_object.assert_not_called()
    obj.carlaActor.apply_control.assert_not_called()
    obj.carlaActor.set_transform.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"controlMode": "cosim"},
        {"releasedFromCoSim": False},
        {"segmentId": "wrong"},
        {"laneIndex": 2},
    ],
)
def test_incomplete_handoff_ack_keeps_actor(changes):
    from scenic.simulators.metsr.cosim_utils import METSRControlError

    simulation, obj, update = publication_fixture(target=("outside", 1, None))
    fields = dict(
        controlMode="native", releasedFromCoSim=True, segmentId="outside", laneIndex=1
    )
    fields.update(changes)
    simulation.metsr_client = SimpleNamespace(
        teleport_cosim_vehicle=Mock(return_value=ack(**fields))
    )
    with pytest.raises(METSRControlError, match="native handback"):
        simulation._publish_carla_vehicle_updates([update])
    simulation.remove_bubble_object.assert_not_called()


def test_mixed_batch_matches_reordered_native_acknowledgments():
    simulation, obj, ordinary = publication_fixture()
    other = Vehicle(name="outgoing", carla_actor_flag=True, spawn_guard=1)
    outgoing = dict(ordinary, object=other, vehicle_id=92, handoff=("outside", 1, None))
    simulation.carla_actors.append(other)
    response = ack()
    response["data"].insert(
        0,
        ack(92, controlMode="native", releasedFromCoSim=True, segmentId="outside")[
            "data"
        ][0],
    )
    simulation.metsr_client = SimpleNamespace(
        teleport_cosim_vehicle=Mock(return_value=response)
    )
    simulation._publish_carla_vehicle_updates([ordinary, outgoing])
    call = simulation.metsr_client.teleport_cosim_vehicle.call_args
    assert call.kwargs["segment_id"] == [None, "outside"]
    assert call.kwargs["lane_index"] == [None, 1]
    assert simulation.carla_actors == [obj]
    simulation.remove_bubble_object.assert_called_once_with(other)


def test_ego_never_requests_native_handback(monkeypatch):
    simulation, obj, _ = publication_fixture()
    simulation.ego = obj
    monkeypatch.setattr(
        carla_routing,
        "projected_carla_driving_waypoint",
        Mock(side_effect=AssertionError("ego needs no mapping")),
    )
    assert routing.native_handoff_target(simulation, obj, object()) is None


def test_inside_region_needs_no_lane_geometry_resolution(monkeypatch):
    simulation, obj, _ = publication_fixture()
    monkeypatch.setattr(
        carla_routing,
        "projected_carla_driving_waypoint",
        lambda simulation, location: (SimpleNamespace(is_junction=False), 0),
    )
    monkeypatch.setattr(
        routing,
        "mapped_lanes_for_carla_waypoint",
        lambda simulation, waypoint: ("controlled_0", "controlled_1"),
    )
    monkeypatch.setattr(
        metsr_network,
        "mapped_lane_query",
        lambda simulation, lane: ("controlled", int(lane[-1])),
    )
    monkeypatch.setattr(
        routing,
        "identify_nearest_road",
        Mock(side_effect=AssertionError("native does association")),
    )
    assert routing.native_handoff_target(simulation, obj, object()) is None


def test_outgoing_target_uses_actual_waypoint_not_previous_native_route(monkeypatch):
    simulation, obj, _ = publication_fixture()
    monkeypatch.setattr(
        carla_routing,
        "projected_carla_driving_waypoint",
        lambda simulation, location: (SimpleNamespace(is_junction=False), 0.1),
    )
    monkeypatch.setattr(
        routing,
        "mapped_lanes_for_carla_waypoint",
        lambda simulation, waypoint: ("outside_1",),
    )
    monkeypatch.setattr(
        metsr_network, "mapped_lane_query", lambda simulation, lane: ("outside", 1)
    )
    assert routing.native_handoff_target(simulation, obj, object()) == (
        "outside",
        1,
        None,
    )
    assert simulation.metsr_state.authoritative_segments[obj] == "controlled"


def cleanup_fixture(monkeypatch):
    simulation, obj, _ = publication_fixture()
    events = []
    simulation.run_name = "test"
    simulation.render = False
    simulation.metsr_client = SimpleNamespace(
        reset=Mock(side_effect=lambda: events.append("reset")),
        verbose=False,
        reach_dest=Mock(side_effect=AssertionError("cleanup is not arrival")),
    )
    monkeypatch.setattr(
        cosim_utils, "log_trip_times", lambda simulation: events.append("log")
    )
    simulation.carla_client = SimpleNamespace(stop_recorder=Mock())
    simulation.release_roads = Mock(side_effect=lambda roads: events.append("release"))
    monkeypatch.setattr(
        cosim_utils,
        "destroy_carla_object_during_teardown",
        Mock(side_effect=lambda simulation, obj: events.append("destroy")),
    )
    return simulation, obj, events


@pytest.mark.parametrize("partially_promoted", [False, True])
def test_normal_cleanup_logs_releases_and_destroys_once(partially_promoted, monkeypatch):
    simulation, obj, events = cleanup_fixture(monkeypatch)
    if partially_promoted:
        simulation.carla_actors = []
    simulation.destroy()
    simulation.destroy()
    assert events == ["log", "release", "destroy"]
    simulation.metsr_client.reset.assert_not_called()
    cosim_utils.destroy_carla_object_during_teardown.assert_called_once_with(
        simulation, obj
    )


def test_blocked_teardown_release_resets_after_logging(monkeypatch):
    simulation, obj, events = cleanup_fixture(monkeypatch)
    simulation.release_roads.side_effect = RuntimeError("RELEASE_BLOCKED")
    simulation.destroy()
    assert events == ["log", "reset", "destroy"]


@pytest.mark.parametrize("reset_fails", [False, True])
def test_aborted_run_resets_and_preserves_original_error(reset_fails, monkeypatch):
    simulation, obj, events = cleanup_fixture(monkeypatch)
    if reset_fails:
        simulation.metsr_client.reset.side_effect = RuntimeError("reset failed")
    original = RuntimeError("sensor failure")
    with pytest.raises(RuntimeError) as caught:
        try:
            raise original
        finally:
            simulation.destroy()
    assert caught.value is original
    simulation.metsr_client.reset.assert_called_once()
    simulation.release_roads.assert_not_called()
    assert events[0] == "log" and events[-1] == "destroy"


@pytest.mark.parametrize("reported_dead", [False, True])
def test_lost_or_unreadable_actor_aborts_without_cancellation_or_arrival(
    reported_dead, monkeypatch
):
    simulation, obj, _ = cleanup_fixture(monkeypatch)
    obj.carlaActor = SimpleNamespace(
        is_alive=not reported_dead,
        get_location=Mock(side_effect=RuntimeError("actor gone")),
    )
    simulation.metsr_client.query_cosim_vehicle = lambda: {
        "messageType": "coSimVehicle",
        "status": "ok",
        "data": [],
    }
    with pytest.raises(
        SimulationCreationError, match="Cannot read CARLA actor"
    ) as caught:
        simulation.synchronize_clients(vehicle_data={obj: {}})
    assert isinstance(caught.value.__cause__, RuntimeError)
    simulation.remove_bubble_object.assert_not_called()
    simulation.metsr_client.reach_dest.assert_not_called()
    assert obj not in simulation.completed_route


def test_partial_batch_never_commits_local_handback():
    from scenic.simulators.metsr.cosim_utils import METSRControlError

    simulation, obj, update = publication_fixture(target=("outside", 1, None))
    other = Vehicle(name="other", carla_actor_flag=True)
    ordinary = dict(update, object=other, vehicle_id=92, handoff=None)
    simulation.carla_actors.append(other)
    response = ack(controlMode="native", releasedFromCoSim=True, segmentId="outside")
    response["status"] = "partial"
    response["data"].append(
        {"vehicleId": 92, "status": "error", "errorCode": "NO_MAP_MATCH"}
    )
    simulation.metsr_client = SimpleNamespace(
        teleport_cosim_vehicle=Mock(return_value=response)
    )
    with pytest.raises(METSRControlError):
        simulation._publish_carla_vehicle_updates([update, ordinary])
    assert simulation.carla_actors == [obj, other]
    simulation.remove_bubble_object.assert_not_called()


def test_retryable_road_release_keeps_ownership_until_later_success():
    simulation, obj, _ = publication_fixture()
    simulation.metsr_state.valid_roads = ["controlled"]
    blocked = {
        "messageType": "releaseCoSimRoad",
        "status": "ok",
        "data": [
            {
                "roadId": "controlled",
                "status": "error",
                "errorCode": "RELEASE_BLOCKED",
                "retryable": True,
            }
        ],
    }
    released = {
        "messageType": "releaseCoSimRoad",
        "status": "ok",
        "data": [
            {
                "roadId": "controlled",
                "status": "ok",
                "releasedConnectorIds": [],
                "releasedConnectors": [],
                "connectorIds": [],
                "connectors": [],
            }
        ],
    }
    simulation.metsr_client = SimpleNamespace(
        release_cosim_road=Mock(side_effect=[blocked, released])
    )
    simulation.release_roads(["controlled"], allow_defer=True)
    assert simulation.metsr_state.controlled_roads == {"controlled": True}
    assert simulation.metsr_client.release_cosim_road.call_count == 1
    simulation.release_roads(["controlled"], allow_defer=True)
    assert not simulation.metsr_state.controlled_roads
    assert simulation.metsr_client.release_cosim_road.call_count == 2


def promotion_fixture():
    from collections import defaultdict

    simulation, obj, _ = publication_fixture()
    obj.carla_actor_flag = False
    obj.carlaActor = None
    simulation.carla_actors = []
    simulation.metsr_actors = [obj]
    simulation.bubble_roads = []
    simulation.sim_ticks_per_carla = 1
    simulation.road_pop_density = defaultdict(int)
    state = {
        "segmentId": "controlled",
        "segmentType": "road",
        "onRoad": True,
        "x": 1,
        "y": 2,
        "z": 0.5,
        "speed": 4,
        "bearing": 90,
    }
    return simulation, obj, state


def test_promotion_commits_only_after_controller_initialization():
    simulation, obj, state = promotion_fixture()
    events = []

    def spawn(target, *, update_velocity, native_state):
        assert native_state is state and update_velocity
        target.carlaActor = object()
        events.append("spawn with native velocity")
        return True

    def initialize(target, native_state):
        assert simulation.carla_actors == [] and simulation.metsr_actors == [obj]
        assert native_state is state
        events.append("controller initialized")

    simulation.createObjectInCarla = spawn
    simulation._initialize_promoted_carla_autopilot = initialize
    simulation.update_bubble_objects(vehicle_data={obj: state})
    assert events == ["spawn with native velocity", "controller initialized"]
    assert simulation.carla_actors == [obj] and simulation.metsr_actors == []


def test_failed_controller_initialization_does_not_commit_promotion():
    simulation, obj, state = promotion_fixture()

    def spawn(*args, **kwargs):
        obj.carlaActor = object()
        return True

    simulation.createObjectInCarla = spawn
    simulation._initialize_promoted_carla_autopilot = Mock(
        side_effect=RuntimeError("controller failed")
    )
    with pytest.raises(RuntimeError, match="controller failed"):
        simulation.update_bubble_objects(vehicle_data={obj: state})
    assert simulation.carla_actors == [] and simulation.metsr_actors == [obj]
    assert obj.carla_actor_flag is False
    # The partially initialized actor remains discoverable for teardown.
    assert obj in simulation.objects and obj.carlaActor is not None


def test_existing_native_vehicle_spawn_failure_has_bounded_retry():
    from scenic.simulators.metsr.cosim_utils import COSIM_ADMISSION_SPAWN_MAX_TICKS

    simulation, obj, state = promotion_fixture()
    simulation.createObjectInCarla = Mock(return_value=False)
    simulation.update_bubble_objects(vehicle_data={obj: state})
    started = simulation.count
    simulation.count += 1
    simulation.update_bubble_objects(vehicle_data={obj: state})
    assert simulation.metsr_state.admitted_vehicles == {
        91: {"roadId": "controlled", "admittedAt": started}
    }
    assert simulation.carla_actors == [] and simulation.metsr_actors == [obj]
    assert (
        native.releasable_old_roads(
            simulation.metsr_state, ["controlled"], simulation.carla_actors
        )
        == []
    )
    simulation.count = started + COSIM_ADMISSION_SPAWN_MAX_TICKS + 1
    with pytest.raises(SimulationCreationError, match="did not spawn"):
        native.check_queue_admission_spawn_deadlines(
            simulation.metsr_state, simulation.count, set()
        )


@pytest.mark.parametrize(
    "observed,expected", [("upstream", None), ("other-exit", ("other-exit", 1, None))]
)
def test_incoming_connector_overlap_is_not_a_handback_to_its_source(
    observed, expected, monkeypatch
):
    simulation, obj, _ = publication_fixture()
    simulation.metsr_state.authoritative_segments[obj] = "opaque-incoming-connector"
    simulation.metsr_state.connector_records = {
        "opaque-incoming-connector": {
            "sourceRoadId": "upstream",
            "targetRoadId": "controlled",
        }
    }
    monkeypatch.setattr(
        carla_routing,
        "projected_carla_driving_waypoint",
        lambda simulation, location: (SimpleNamespace(is_junction=False), 0.1),
    )
    monkeypatch.setattr(
        routing,
        "mapped_lanes_for_carla_waypoint",
        lambda simulation, waypoint: (observed + "_1",),
    )
    monkeypatch.setattr(
        metsr_network, "mapped_lane_query", lambda simulation, lane: (observed, 1)
    )
    # Native admitted this connector, but CARLA can still report the source
    # road at its entrance. A different actual exit remains a valid handback.
    assert routing.native_handoff_target(simulation, obj, object()) == expected


def test_initial_native_npc_has_properties_before_first_step(monkeypatch):
    from unittest.mock import patch

    from scenic.core.vectors import Orientation

    simulation = make_simulation()

    simulation.metsr_state = METSRCosimState()
    obj = Vehicle(
        name="initial-native",
        carla_actor_flag=False,
        carlaActor=None,
        parentOrientation=Orientation.fromEuler(0, 0, 0),
    )
    simulation.objects = [obj]
    simulation.scene = SimpleNamespace(objects=[obj], params={})
    simulation.metsr_actors = [obj]
    simulation.carla_actors = []
    simulation.workspace = object()
    simulation.metsr_state.metadata = replace(
        simulation.metsr_state.metadata, lane_mapping={}
    )
    simulation.map = SimpleNamespace(name="Town05")
    simulation.render = False
    simulation.record = ""
    simulation.metsr_client = SimpleNamespace(
        reset=Mock(), query_road=lambda: {"roadIds": ["A"]}
    )
    simulation.carla_world = SimpleNamespace(tick=Mock())
    simulation._synchronize_boundary_vehicles = Mock()
    state = {"x": 1, "y": 2, "z": 3, "speed": 4, "bearing": 90}
    monkeypatch.setattr(
        native, "collect_metsr_vehicle_data", Mock(return_value={obj: state})
    )
    monkeypatch.setattr(
        actor_utils,
        "actor_properties",
        Mock(side_effect=AssertionError("native NPC has no CARLA actor")),
    )
    with patch("scenic.simulators.cosim.simulator.DrivingSimulation.setup"), patch(
        "scenic.simulators.cosim.simulator.network_cache"
    ), patch("scenic.simulators.cosim.simulator.GlobalRoutePlanner"):
        simulation.setup()
    # Scenic's core requests these properties immediately after setup, before step().
    properties = simulation.getProperties(obj, {})
    assert tuple(properties["position"]) == (1, 2, 3)
    assert tuple(properties["velocity"]) == pytest.approx((4, 0, 0), abs=1e-6)
    native.collect_metsr_vehicle_data.assert_called_once_with(
        simulation.metsr_client, {obj: simulation.metsr_state.private_vehicle_ids[obj]}
    )
