"""A final macro road can begin ahead of CARLA's current connector/road."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scenic.core.simulators import SimulationCreationError
from scenic.simulators.cosim.simulator import CosimSimulation, carla
from scenic.simulators.cosim.utils import utils
from scenic.simulators.cosim.utils.global_route_planner import GlobalRoutePlanner


def waypoint(road, lane, x, junction=False):
    return SimpleNamespace(
        road_id=road, lane_id=lane, section_id=0, s=float(x),
        is_junction=junction,
        transform=carla.Transform(carla.Location(x=float(x))),
    )


@pytest.fixture
def terminal_route():
    simulation = object.__new__(CosimSimulation)
    source = waypoint(31, -1, 0)
    target = waypoint(48, -1, 20)
    simulation.scenic_to_metsr_map = {
        "31_-1": ["incoming_4"], "48_-1": ["-48_4"],
    }
    simulation.metsr_lane_indices = {
        "incoming_4": 0, "incoming_5": 1, "-48_4": 0, "-48_5": 1,
    }
    simulation.metsr_lane_connections = {
        ("incoming", "-48"): [("incoming_4", "-48_4"), ("incoming_5", "-48_5")],
    }
    simulation.metsr_internal_lane_connections = {
        ":junction_0_0": [("incoming_4", "-48_4")],
    }
    simulation._sumo_lane_to_carla_keys = {"-48_4": {"48_-1"}}
    simulation._carla_waypoints_by_key = {"48_-1": [target]}
    simulation.metsr_road_cache = {
        ("-48", 0): [(10.0, 0.0, 0.0), (30.0, 0.0, 0.0)],
    }
    simulation.map = SimpleNamespace(get_waypoint=lambda location, **kwargs: min(
        (source, target), key=lambda wp: abs(wp.transform.location.x - location.x)
    ))
    simulation.grp = SimpleNamespace(
        trace_route=lambda start, end: [(source, None), (target, None)],
    )
    obj = SimpleNamespace(
        name="car_4", carlaActor=SimpleNamespace(get_location=lambda: source.transform.location),
    )
    return simulation, obj, source, target


@pytest.mark.parametrize("on_connector", [False, True])
def test_single_road_route_stitches_from_exact_incoming_lane(terminal_route, on_connector):
    simulation, obj, source, _ = terminal_route
    if on_connector:
        source.is_junction = True
        simulation.scenic_to_metsr_map["31_-1"] = [":junction_0_0"]

    assert simulation._candidate_metsr_lane_chains(
        ["-48"], source.transform.location
    ) == [["-48_4"]]
    locations = simulation.generate_carla_trajectory(["-48"], obj)
    assert [location.x for location in locations] == [0.0, 20.0]


def test_single_road_route_on_target_keeps_current_lane(terminal_route):
    simulation, _, _, target = terminal_route
    assert simulation._candidate_metsr_lane_chains(
        ["-48"], target.transform.location
    ) == [["-48_4"]]


def test_single_road_route_rejects_wrong_predecessor_lane(terminal_route):
    simulation, obj, source, _ = terminal_route
    simulation.scenic_to_metsr_map["31_-1"] = ["incoming_6"]
    simulation.metsr_lane_indices["incoming_6"] = 2
    assert simulation._candidate_metsr_lane_chains(
        ["-48"], source.transform.location
    ) == []
    with pytest.raises(SimulationCreationError):
        simulation.generate_carla_trajectory(["-48"], obj)


def test_single_road_route_rejects_connector_for_another_exit(terminal_route):
    simulation, obj, source, _ = terminal_route
    source.is_junction = True
    simulation.scenic_to_metsr_map["31_-1"] = [":other_0_0"]
    simulation.metsr_internal_lane_connections[":other_0_0"] = [("incoming_4", "other_0")]
    with pytest.raises(SimulationCreationError, match="No direct SUMO"):
        simulation.generate_carla_trajectory(["-48"], obj)


def test_single_road_route_does_not_use_a_filtered_target_lane(terminal_route):
    simulation, _, source, _ = terminal_route
    simulation.metsr_lane_indices.pop("-48_4")
    assert simulation._candidate_metsr_lane_chains(
        ["-48"], source.transform.location
    ) == []


def test_single_road_failure_reports_the_observed_lane(terminal_route):
    simulation, obj, source, _ = terminal_route
    source.is_junction = True
    simulation.scenic_to_metsr_map["31_-1"] = [":unmapped_0_0"]
    with pytest.raises(SimulationCreationError) as error:
        simulation.generate_carla_trajectory(["-48"], obj)
    message = str(error.value)
    assert "carla_lane='31_-1'" in message
    assert "is_junction=True" in message
    assert "mapped_sumo_lanes=[':unmapped_0_0']" in message


def test_town05_connector_to_road_minus_48_builds_single_road_route(getAssetPath):
    xml = getAssetPath("maps/CARLA/Town05.net.xml")
    xodr = getAssetPath("maps/CARLA/Town05.xodr")
    simulation = object.__new__(CosimSimulation)
    simulation.scenic_to_metsr_map = utils.generate_map(xml)
    simulation.metsr_lane_indices = utils.generate_metsr_lane_index_map(xml)
    simulation.metsr_lane_connections = utils.generate_metsr_lane_connection_map(xml)
    simulation.metsr_internal_lane_connections = utils.generate_metsr_internal_lane_connection_map(xml)
    simulation.map = carla.Map("Town05", Path(xodr).read_text())
    key = next(key for key, lanes in simulation.scenic_to_metsr_map.items()
               if ":720_9_0" in lanes)
    road, lane = map(int, key.split("_"))
    # The first meter overlaps another turn geometrically. Use a point on the
    # distinct incoming movement so CARLA identifies the intended connector.
    start = simulation.map.get_waypoint_xodr(road, lane, 5.0)
    assert start is not None and start.is_junction
    route = simulation._align_carla_route_start(["-48"], start.transform.location)
    assert simulation._candidate_metsr_lane_chains(
        route, start.transform.location
    ) == [["-48_4"]]

    simulation.grp = GlobalRoutePlanner(simulation.map, sampling_resolution=2.0)
    simulation._carla_waypoints_by_key = None
    simulation._index_carla_waypoints()
    simulation._sumo_lane_to_carla_keys = {"-48_4": {"48_-1"}}
    # Supply an aligned target centerline locally; no METS-R server is needed.
    target_waypoints = sorted(
        simulation._carla_waypoints_by_key["48_-1"], key=lambda wp: wp.s
    )
    simulation.metsr_road_cache = {("-48", 0): [
        (wp.transform.location.x, -wp.transform.location.y, wp.transform.location.z)
        for wp in target_waypoints
    ]}
    obj = SimpleNamespace(carlaActor=SimpleNamespace(
        get_location=lambda: start.transform.location
    ))
    locations = simulation.generate_carla_trajectory(["-48"], obj)
    assert len(locations) > 2
    terminal = simulation.map.get_waypoint(locations[-1])
    assert "-48_4" in simulation.scenic_to_metsr_map[
        f"{terminal.road_id}_{terminal.lane_id}"
    ]
