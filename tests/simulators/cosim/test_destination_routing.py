"""CARLA chooses its route; the ordinary METS-R vehicle query supplies the goal."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import pytest

from scenic.core.simulators import SimulationCreationError
from scenic.simulators.carla import routing_utils as carla_routing
from scenic.simulators.carla.global_route_planner import GlobalRoutePlanner
from scenic.simulators.cosim import runtime_utils as runtime
from scenic.simulators.cosim.simulator import CosimSimulation, carla
from scenic.simulators.metsr import (
    network_utils as metsr_network,
    routing_utils as routing,
)
from scenic.simulators.metsr.bridge_utils import METSRCosimState


class Vehicle(SimpleNamespace):
    __hash__ = object.__hash__


def waypoint(road, lane, x, y=0):
    return SimpleNamespace(
        road_id=road,
        lane_id=lane,
        section_id=0,
        s=float(x),
        is_junction=False,
        transform=carla.Transform(carla.Location(x=float(x), y=float(y))),
    )


@pytest.fixture
def destination_route(monkeypatch):
    simulation = object.__new__(CosimSimulation)
    simulation.metsr_client = None
    simulation.metsr_state = METSRCosimState()
    simulation._carla_destination_plans = {}
    simulation.ego = None
    simulation.metsr_state.authoritative_segments = {}
    origin = waypoint(1, -1, 0)
    other = waypoint(2, -1, 5)
    far = waypoint(3, -1, 30)
    near = waypoint(3, -2, 20, 3)
    simulation.metsr_state.metadata = metsr_network.NetworkMetadata(
        lane_mapping={
            "1_-1": ["A_0"],
            "2_-1": ["CARLA_CHOICE_0"],
            "3_-1": ["Z_2"],
            "3_-2": ["Z_3"],
            "parking": ["Z_0"],
        },
        lane_indices={
            "A_0": 0,
            "CARLA_CHOICE_0": 0,
            "Z_2": 0,
            "Z_3": 1,
        },
    )
    simulation.map = SimpleNamespace(get_waypoint=lambda *args, **kwargs: origin)
    monkeypatch.setattr(
        routing,
        "lane_anchor_waypoint",
        lambda simulation, lane, final=False: {"Z_2": far, "Z_3": near}[lane],
    )
    traces = {
        30.0: [(origin, None), (other, None), (far, None)],
        20.0: [(origin, None), (other, None), (near, None)],
    }
    calls = []

    def trace_route(start, end):
        calls.append((start, end))
        result = traces[end.x]
        if isinstance(result, Exception):
            raise result
        return result

    simulation.grp = SimpleNamespace(trace_route=trace_route)
    obj = Vehicle(
        name="car_4",
        route=["METS_NEXT", "METS_OTHER", "Z"],
        carlaActor=SimpleNamespace(get_location=lambda: origin.transform.location),
    )
    return simulation, obj, traces, calls


def test_carla_selects_shortest_reachable_destination_lane_and_its_own_roads(
    destination_route,
):
    simulation, obj, _, calls = destination_route
    path = routing.generate_carla_destination_trajectory(simulation, "Z", obj)
    assert [location.x for location in path] == [0.0, 5.0, 20.0]
    assert len(calls) == 2  # Driving lanes only, not the parking lane.
    assert obj.route == ["METS_NEXT", "METS_OTHER", "Z"]


@pytest.mark.parametrize("failed_trace", [[], nx.NetworkXNoPath("unreachable")])
def test_unreachable_destination_lane_does_not_hide_other_choices(
    destination_route, failed_trace
):
    simulation, obj, traces, _ = destination_route
    traces[20.0] = failed_trace
    path = routing.generate_carla_destination_trajectory(simulation, "Z", obj)
    assert path[-1].x == 30.0


def test_carla_path_must_reach_target_road(destination_route):
    simulation, obj, traces, _ = destination_route
    traces[20.0] = traces[20.0][:-1]
    traces[30.0] = traces[30.0][:-1]
    with pytest.raises(SimulationCreationError, match="destination road Z"):
        routing.generate_carla_destination_trajectory(simulation, "Z", obj)


def test_already_on_target_does_not_plan_a_loop(destination_route):
    simulation, obj, traces, calls = destination_route
    target = traces[20.0][-1][0]
    simulation.map.get_waypoint = lambda *args, **kwargs: target
    obj.carlaActor.get_location = lambda: target.transform.location
    path = routing.generate_carla_destination_trajectory(simulation, "Z", obj)
    assert path == [target.transform.location]
    assert calls == []


def test_unmapped_target_is_reported_as_destination_problem(destination_route):
    simulation, obj, _, _ = destination_route
    with pytest.raises(SimulationCreationError, match="destination road missing"):
        routing.generate_carla_destination_trajectory(simulation, "missing", obj)


def test_destination_change_refreshes_default_path_once(destination_route, monkeypatch):
    simulation, obj, _, _ = destination_route
    old_path, new_path = [object()], [object()]
    obj.trajectory = old_path
    obj.active_autopilot = True
    simulation._carla_destination_plans = {obj: ("OLD", old_path)}
    generated, installed = [], []
    monkeypatch.setattr(
        routing,
        "generate_carla_destination_trajectory",
        lambda simulation, destination, target, **kwargs: generated.append(
            (destination, target)
        )
        or new_path,
    )
    simulation.tm = SimpleNamespace(set_path=lambda actor, path: installed.append(path))
    simulation._refresh_carla_destination_paths({obj: {"destinationRoadId": "Z"}})
    simulation._refresh_carla_destination_paths({obj: {"destinationRoadId": "Z"}})
    assert generated == [("Z", obj)]
    assert installed == [new_path]
    assert obj.trajectory is new_path


def test_explicit_scenic_path_is_preserved(destination_route):
    simulation, obj, _, _ = destination_route
    previous_default, custom = [object()], [object()]
    obj.trajectory = custom
    obj.active_autopilot = True
    simulation._carla_destination_plans = {obj: ("OLD", previous_default)}
    simulation._refresh_carla_destination_paths({obj: {"destinationRoadId": "Z"}})
    assert obj.trajectory is custom
    assert simulation._carla_destination_plans == {}


def test_default_path_is_cleared_on_demotion(destination_route):
    simulation, obj, _, _ = destination_route
    path = [object()]
    obj.trajectory = path
    simulation._carla_destination_plans = {obj: ("Z", path)}
    runtime.clear_carla_route_state(simulation, obj)
    assert obj.trajectory is None
    assert simulation._carla_destination_plans == {}


def test_autopilot_without_trajectory_still_gets_a_destination_path(
    destination_route, monkeypatch
):
    simulation, obj, _, _ = destination_route
    obj.carla_actor_flag = True
    obj.autopilot_action = True
    obj.active_autopilot = False
    simulation.agents = [obj]
    path, installed = [object()], []
    simulation.metsr_client = SimpleNamespace(
        query_vehicle=lambda *args, **kwargs: {
            "messageType": "vehicle",
            "status": "ok",
            "data": [{"vehicleId": 4, "destinationRoadId": "Z"}],
        }
    )
    simulation.getMetsrPrivateVehId = lambda target: 4
    monkeypatch.setattr(
        routing, "generate_carla_destination_trajectory", lambda *args, **kwargs: path
    )
    simulation.tm = SimpleNamespace(set_path=lambda actor, value: installed.append(value))
    simulation._enable_carla_autopilot = lambda target: True
    with patch("scenic.simulators.cosim.simulator.DrivingSimulation.executeActions"):
        simulation.executeActions({obj: ()})
    assert installed == [path]
    assert obj.trajectory is path


@pytest.mark.parametrize("start_s", [1.0, 5.0])
def test_default_town05_route_reaches_minus_48_without_metsr_path_constraints(
    getAssetPath, start_s
):
    xml = getAssetPath("maps/CARLA/Town05.net.xml")
    simulation = object.__new__(CosimSimulation)
    simulation.metsr_client = None
    simulation.metsr_state = METSRCosimState()
    simulation._carla_destination_plans = {}
    metadata = metsr_network.load_network_metadata(xml)
    simulation.metsr_state = METSRCosimState(metadata=metadata)
    simulation.map = carla.Map(
        "Town05", Path(getAssetPath("maps/CARLA/Town05.xodr")).read_text()
    )
    simulation.grp = GlobalRoutePlanner(simulation.map, sampling_resolution=2.0)
    simulation._carla_waypoints_by_key = None
    simulation._carla_waypoints_by_key = carla_routing.index_carla_waypoints(
        simulation.map, simulation.grp
    )
    simulation.metsr_state.metadata = replace(
        metadata,
        lane_mapping={
            key: tuple(
                lane
                for lane in lanes
                if lane.rsplit("_", 1)[0] != "-48" or (key == "48_-1" and lane == "-48_4")
            )
            for key, lanes in metadata.lane_mapping.items()
        },
    )
    simulation.metsr_state.road_cache = {
        ("-48", 0): [
            (wp.transform.location.x, -wp.transform.location.y, wp.transform.location.z)
            for wp in sorted(
                simulation._carla_waypoints_by_key["48_-1"], key=lambda wp: wp.s
            )
        ]
    }
    start = simulation.map.get_waypoint_xodr(816, -1, start_s)
    obj = Vehicle(
        name="car_4",
        route=["-48"],
        carlaActor=SimpleNamespace(get_location=lambda: start.transform.location),
    )
    simulation.getMetsrPrivateVehId = lambda target: 4
    simulation.metsr_client = SimpleNamespace(
        query_vehicle=lambda *args, **kwargs: {
            "messageType": "vehicle",
            "status": "ok",
            "data": [{"vehicleId": 4, "destinationRoadId": "-48"}],
        }
    )
    simulation.tm = SimpleNamespace(set_path=lambda actor, path: None)
    path = simulation._install_carla_destination_path(obj)
    assert len(path) > 2
    terminal = simulation.map.get_waypoint(path[-1])
    assert (
        "-48_4"
        in simulation.metsr_state.metadata.lane_mapping[
            f"{terminal.road_id}_{terminal.lane_id}"
        ]
    )


@pytest.mark.parametrize("wrong_terminal", [False, True])
def test_shared_carla_lane_mapping_uses_physical_destination_fragment(
    destination_route, wrong_terminal
):
    simulation, obj, traces, _ = destination_route
    simulation.metsr_state.metadata = replace(
        simulation.metsr_state.metadata,
        lane_mapping={
            **simulation.metsr_state.metadata.lane_mapping,
            "1_-1": ["A_0", "Z_2"],
        },
    )
    simulation.metsr_state.road_cache = {
        ("A", 0): [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        ("Z", 0): [(15.0, 0.0, 0.0), (30.0, 0.0, 0.0)],
    }
    if wrong_terminal:
        for key in traces:
            traces[key] = [(waypoint(1, -1, 5), None)]
        with pytest.raises(SimulationCreationError, match="destination road Z"):
            routing.generate_carla_destination_trajectory(simulation, "Z", obj)
    else:
        path = routing.generate_carla_destination_trajectory(simulation, "Z", obj)
        assert [location.x for location in path] == [0.0, 5.0, 20.0]


def test_tm_path_keeps_curved_road_points_and_targets_junction_exits(destination_route):
    simulation, obj, traces, _ = destination_route
    traces[30.0] = []
    origin = traces[20.0][0][0]
    curve = waypoint(2, -1, 5, 8)
    inside = waypoint(200, -1, 10, 8)
    inside.is_junction = True
    target = traces[20.0][-1][0]
    traces[20.0] = [(wp, None) for wp in (origin, curve, inside, target)]
    path = routing.generate_carla_destination_trajectory(simulation, "Z", obj)
    assert [(point.x, point.y) for point in path] == [(0, 0), (5, 8), (20, 3)]


def test_route_length_still_includes_geometry_inside_junctions(destination_route):
    simulation, obj, traces, _ = destination_route
    origin, target = traces[20.0][0][0], traces[20.0][-1][0]
    long_turn = waypoint(200, -1, 10, 50)
    long_turn.is_junction = True
    traces[20.0] = [(wp, None) for wp in (origin, long_turn, target)]
    path = routing.generate_carla_destination_trajectory(simulation, "Z", obj)
    assert path[-1].x == 30.0  # The closer goal needs a longer junction traversal.


def test_default_route_requires_an_unambiguous_destination_anchor(destination_route):
    simulation, obj, traces, _ = destination_route
    simulation.metsr_state.metadata = replace(
        simulation.metsr_state.metadata,
        lane_mapping={
            key: lanes
            for key, lanes in simulation.metsr_state.metadata.lane_mapping.items()
            if key != "3_-1"
        },
    )
    traces[30.0] = []
    traces[20.0][-1][0].is_junction = True
    with pytest.raises(
        SimulationCreationError, match="no unambiguous destination road anchor"
    ):
        routing.generate_carla_destination_trajectory(simulation, "Z", obj)


def test_captured_car83_path_targets_straight_exit_instead_of_overlapping_entrance(
    getAssetPath,
):
    xml = getAssetPath("maps/CARLA/Town05.net.xml")
    simulation = object.__new__(CosimSimulation)
    simulation.metsr_client = None
    simulation.metsr_state = METSRCosimState()
    simulation._carla_destination_plans = {}
    metadata = metsr_network.load_network_metadata(xml)
    simulation.metsr_state = METSRCosimState(metadata=metadata)
    simulation.map = carla.Map(
        "Town05", Path(getAssetPath("maps/CARLA/Town05.xodr")).read_text()
    )
    simulation.grp = GlobalRoutePlanner(simulation.map, sampling_resolution=2.0)
    simulation._carla_waypoints_by_key = None
    simulation._carla_waypoints_by_key = carla_routing.index_carla_waypoints(
        simulation.map, simulation.grp
    )
    simulation.metsr_state.metadata = replace(
        metadata,
        lane_mapping={
            key: tuple(
                lane
                for lane in lanes
                if lane.rsplit("_", 1)[0] != "-43" or (key == "43_-2" and lane == "-43_3")
            )
            for key, lanes in metadata.lane_mapping.items()
        },
    )
    simulation.metsr_state.road_cache = {
        ("-43", 0): [
            (wp.transform.location.x, -wp.transform.location.y, wp.transform.location.z)
            for wp in sorted(
                simulation._carla_waypoints_by_key["43_-2"], key=lambda wp: wp.s
            )
        ]
    }
    start = carla.Location(-184.54019165039062, 105.16989135742188, 0.5407505035400391)
    obj = Vehicle(name="car_83", carlaActor=SimpleNamespace(get_location=lambda: start))
    anchor = routing.lane_anchor_waypoint(simulation, "-43_3", final=True)
    full_trace = simulation.grp.trace_route(start, anchor.transform.location)
    old_anchor = full_trace[0][0]
    assert old_anchor.is_junction
    # TM's ImportPath compares junction exits to the next imported coordinate.
    # The internal entrance coordinate is closer to the wrong right-turn exit.
    distances = {}
    for branch in simulation.map.get_waypoint(start).next(2.0):
        exit_point = branch
        for _ in range(200):
            if not exit_point.is_junction:
                break
            exit_point = exit_point.next(0.5)[0]
        distances[exit_point.road_id] = exit_point.transform.location.distance(
            old_anchor.transform.location
        )
    assert distances[23] < distances[43]
    path = routing.generate_carla_destination_trajectory(simulation, "-43", obj)
    first = simulation.map.get_waypoint(path[0])
    assert not first.is_junction
    assert first.road_id == 43
    assert first.lane_id == -2
    assert all(not simulation.map.get_waypoint(point).is_junction for point in path)


def test_promoted_default_autopilot_starts_before_first_motion_from_fresh_state(
    destination_route, monkeypatch
):
    simulation, obj, _, _ = destination_route
    obj.autopilot_action = True
    obj.active_autopilot = False
    obj.carla_actor_flag = True
    obj.trajectory = None
    obj._control = object()
    obj.carlaActor.get_location = lambda: pytest.fail(
        "New CARLA actor snapshot is not published yet"
    )
    simulation.metsr_client = SimpleNamespace(
        query_vehicle=lambda *a, **k: pytest.fail("Use admission snapshot")
    )
    path = [carla.Location(-184.5, 73, 0.06)]
    calls = []

    def generate(simulation, destination, target, *, target_start):
        calls.append(("plan", destination, target_start))
        return path

    monkeypatch.setattr(routing, "generate_carla_destination_trajectory", generate)
    simulation.tm = SimpleNamespace(
        set_path=lambda actor, value: calls.append(("path", value)),
        auto_lane_change=lambda actor, enabled: calls.append(("lanes", enabled)),
        get_port=lambda: 8000,
    )
    obj.carlaActor.set_autopilot = lambda enabled, port: calls.append(
        ("autopilot", enabled, port)
    )
    state = {"destinationRoadId": "-43", "x": -184.54, "y": -105.76, "z": 0.06}
    assert simulation._initialize_promoted_carla_autopilot(obj, state)
    assert [entry[0] for entry in calls] == ["plan", "path", "autopilot", "lanes"]
    assert calls[0][1] == "-43"
    assert (calls[0][2].x, calls[0][2].y, calls[0][2].z) == pytest.approx(
        (-184.54, 105.76, 0.06)
    )
    assert obj.active_autopilot is True
    assert obj._control is None
    assert simulation._carla_destination_plans[obj] == ("-43", path)
    # The next action frame and the same-step destination refresh must retain it.
    simulation.agents = [obj]
    with patch("scenic.simulators.cosim.simulator.DrivingSimulation.executeActions"):
        simulation.executeActions({obj: ()})
    simulation._refresh_carla_destination_paths({obj: state})
    assert len(calls) == 4


@pytest.mark.parametrize(
    "kind", ["manual", "explicit", "pcla", "pcla_npc", "already_active"]
)
def test_early_promotion_does_not_override_other_control_modes(
    destination_route, kind, monkeypatch
):
    simulation, obj, _, _ = destination_route
    obj.autopilot_action = kind != "manual"
    obj.active_autopilot = kind == "already_active"
    explicit = [object()]
    obj.trajectory = explicit if kind == "explicit" else None
    if kind in ("pcla", "pcla_npc"):
        obj.pcla = object()
        if kind == "pcla":
            simulation.ego = obj
    monkeypatch.setattr(
        routing,
        "generate_carla_destination_trajectory",
        lambda *a, **k: pytest.fail("Control mode must be preserved"),
    )
    assert not simulation._initialize_promoted_carla_autopilot(obj, {})
    if kind == "explicit":
        assert obj.trajectory is explicit


def test_failed_destination_path_install_preserves_previous_plan(
    destination_route, monkeypatch
):
    simulation, obj, _, _ = destination_route
    old_path, new_path = [object()], [object()]
    obj.trajectory = old_path
    obj.active_autopilot = True
    simulation._carla_destination_plans[obj] = ("OLD", old_path)
    monkeypatch.setattr(
        routing, "generate_carla_destination_trajectory", lambda *args, **kwargs: new_path
    )

    def reject_path(actor, path):
        raise RuntimeError("Traffic Manager rejected path")

    simulation.tm = SimpleNamespace(set_path=reject_path)
    with pytest.raises(RuntimeError, match="rejected path"):
        simulation._refresh_carla_destination_paths({obj: {"destinationRoadId": "Z"}})
    assert obj.trajectory is old_path
    assert simulation._carla_destination_plans[obj] == ("OLD", old_path)
