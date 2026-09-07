"""CARLA chooses its route; the ordinary METS-R vehicle query supplies the goal."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import pytest

from scenic.core.simulators import SimulationCreationError
from scenic.simulators.cosim.simulator import CosimSimulation, carla
from scenic.simulators.cosim.utils import utils
from scenic.simulators.cosim.utils.global_route_planner import GlobalRoutePlanner


class Vehicle(SimpleNamespace):
    __hash__ = object.__hash__


def waypoint(road, lane, x, y=0):
    return SimpleNamespace(
        road_id=road, lane_id=lane, section_id=0, s=float(x), is_junction=False,
        transform=carla.Transform(carla.Location(x=float(x), y=float(y))),
    )


@pytest.fixture
def destination_route():
    simulation = object.__new__(CosimSimulation)
    origin = waypoint(1, -1, 0)
    other = waypoint(2, -1, 5)
    far = waypoint(3, -1, 30)
    near = waypoint(3, -2, 20, 3)
    simulation.scenic_to_metsr_map = {
        "1_-1": ["A_0"], "2_-1": ["CARLA_CHOICE_0"],
        "3_-1": ["Z_2"], "3_-2": ["Z_3"],
    }
    simulation.metsr_lane_indices = {"A_0": 0, "CARLA_CHOICE_0": 0, "Z_2": 0, "Z_3": 1}
    simulation._sumo_lane_to_carla_keys = {
        "Z_2": {"3_-1"}, "Z_3": {"3_-2"}, "Z_0": {"parking"},
    }
    simulation.map = SimpleNamespace(get_waypoint=lambda *args, **kwargs: origin)
    simulation._lane_anchor_waypoint = lambda lane, final=False: {"Z_2": far, "Z_3": near}[lane]
    traces = {30.0: [(origin, None), (other, None), (far, None)],
              20.0: [(origin, None), (other, None), (near, None)]}
    calls = []

    def trace_route(start, end):
        calls.append((start, end))
        result = traces[end.x]
        if isinstance(result, Exception):
            raise result
        return result

    simulation.grp = SimpleNamespace(trace_route=trace_route)
    obj = Vehicle(
        name="car_4", route=["METS_NEXT", "METS_OTHER", "Z"],
        carlaActor=SimpleNamespace(get_location=lambda: origin.transform.location),
    )
    return simulation, obj, traces, calls


def test_carla_selects_shortest_reachable_destination_lane_and_its_own_roads(destination_route):
    simulation, obj, _, calls = destination_route
    simulation.generate_carla_trajectory = lambda *args, **kwargs: pytest.fail("No strict macro-route matching")
    path = simulation.generate_carla_destination_trajectory("Z", obj)
    assert [location.x for location in path] == [0.0, 5.0, 20.0]
    assert len(calls) == 2  # Driving lanes only, not the parking lane.
    assert obj.route == ["METS_NEXT", "METS_OTHER", "Z"]


@pytest.mark.parametrize("failed_trace", [[], nx.NetworkXNoPath("unreachable")])
def test_unreachable_destination_lane_does_not_hide_other_choices(destination_route, failed_trace):
    simulation, obj, traces, _ = destination_route
    traces[20.0] = failed_trace
    path = simulation.generate_carla_destination_trajectory("Z", obj)
    assert path[-1].x == 30.0


def test_carla_path_must_reach_target_road(destination_route):
    simulation, obj, traces, _ = destination_route
    traces[20.0] = traces[20.0][:-1]
    traces[30.0] = traces[30.0][:-1]
    with pytest.raises(SimulationCreationError, match="destination road Z"):
        simulation.generate_carla_destination_trajectory("Z", obj)


def test_already_on_target_does_not_plan_a_loop(destination_route):
    simulation, obj, traces, calls = destination_route
    target = traces[20.0][-1][0]
    simulation.map.get_waypoint = lambda *args, **kwargs: target
    obj.carlaActor.get_location = lambda: target.transform.location
    path = simulation.generate_carla_destination_trajectory("Z", obj)
    assert path == [target.transform.location]
    assert calls == []


def test_unmapped_target_is_reported_as_destination_problem(destination_route):
    simulation, obj, _, _ = destination_route
    with pytest.raises(SimulationCreationError, match="destination road missing"):
        simulation.generate_carla_destination_trajectory("missing", obj)


def test_destination_change_refreshes_default_path_once(destination_route):
    simulation, obj, _, _ = destination_route
    old_path, new_path = [object()], [object()]
    obj.trajectory = old_path
    obj.active_autopilot = True
    simulation._carla_destination_plans = {obj: ("OLD", old_path)}
    generated, installed = [], []
    simulation.generate_carla_destination_trajectory = (
        lambda destination, target: generated.append((destination, target)) or new_path
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
    simulation._clear_carla_authoritative_segment(obj)
    assert obj.trajectory is None
    assert simulation._carla_destination_plans == {}


def test_autopilot_without_trajectory_still_gets_a_destination_path(destination_route):
    simulation, obj, _, _ = destination_route
    obj.carla_actor_flag = True
    obj.autopilot_action = True
    obj.active_autopilot = False
    simulation.agents = [obj]
    path, installed = [object()], []
    simulation.metsr_trajectory_to_carla = lambda target: path
    simulation.tm = SimpleNamespace(set_path=lambda actor, value: installed.append(value))
    simulation.initiate_autopilot = lambda target: True
    with patch("scenic.simulators.cosim.simulator.DrivingSimulation.executeActions"):
        simulation.executeActions({obj: ()})
    assert installed == [path]
    assert obj.trajectory is path


def test_cached_route_does_not_authorize_disconnected_physical_jump():
    simulation = object.__new__(CosimSimulation)
    simulation.metsr_lane_connections = {}
    obj = Vehicle(route=["A", "B"])
    assert not simulation._carla_segment_transition_is_compatible(obj, "A", "B")
    simulation.metsr_lane_connections = {("A", "C"): [("A_0", "C_0")]}
    assert simulation._carla_segment_transition_is_compatible(obj, "A", "C")


@pytest.mark.parametrize("start_s", [1.0, 5.0])
def test_default_town05_route_reaches_minus_48_without_metsr_path_constraints(getAssetPath, start_s):
    xml = getAssetPath("maps/CARLA/Town05.net.xml")
    simulation = object.__new__(CosimSimulation)
    simulation.scenic_to_metsr_map = utils.generate_map(xml)
    simulation.metsr_lane_indices = utils.generate_metsr_lane_index_map(xml)
    simulation.map = carla.Map("Town05", Path(getAssetPath("maps/CARLA/Town05.xodr")).read_text())
    simulation.grp = GlobalRoutePlanner(simulation.map, sampling_resolution=2.0)
    simulation._carla_waypoints_by_key = None
    simulation._index_carla_waypoints()
    simulation._sumo_lane_to_carla_keys = {"-48_4": {"48_-1"}}
    simulation.metsr_road_cache = {("-48", 0): [
        (wp.transform.location.x, -wp.transform.location.y, wp.transform.location.z)
        for wp in sorted(simulation._carla_waypoints_by_key["48_-1"], key=lambda wp: wp.s)
    ]}
    start = simulation.map.get_waypoint_xodr(816, -1, start_s)
    obj = Vehicle(name="car_4", route=["-48"], carlaActor=SimpleNamespace(
        get_location=lambda: start.transform.location
    ))
    simulation.getMetsrPrivateVehId = lambda target: 4
    simulation.metsr_client = SimpleNamespace(query_vehicle=lambda *args, **kwargs: {
        "messageType": "vehicle", "status": "ok",
        "data": [{"vehicleId": 4, "destinationRoadId": "-48"}],
    })
    path = simulation.metsr_trajectory_to_carla(obj)
    assert len(path) > 2
    terminal = simulation.map.get_waypoint(path[-1])
    assert "-48_4" in simulation.scenic_to_metsr_map[
        f"{terminal.road_id}_{terminal.lane_id}"
    ]


@pytest.mark.parametrize("wrong_terminal", [False, True])
def test_shared_carla_lane_mapping_uses_physical_destination_fragment(destination_route, wrong_terminal):
    simulation, obj, traces, _ = destination_route
    simulation.scenic_to_metsr_map["1_-1"] = ["A_0", "Z_2"]
    simulation.metsr_road_cache = {
        ("A", 0): [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        ("Z", 0): [(15.0, 0.0, 0.0), (30.0, 0.0, 0.0)],
    }
    if wrong_terminal:
        for key in traces:
            traces[key] = [(waypoint(1, -1, 5), None)]
        with pytest.raises(SimulationCreationError, match="destination road Z"):
            simulation.generate_carla_destination_trajectory("Z", obj)
    else:
        path = simulation.generate_carla_destination_trajectory("Z", obj)
        assert [location.x for location in path] == [0.0, 5.0, 20.0]
