"""Destination routing works from a road or connector before the final road."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scenic.core.simulators import SimulationCreationError
from scenic.simulators.carla import routing_utils as carla_routing
from scenic.simulators.carla.global_route_planner import GlobalRoutePlanner
from scenic.simulators.cosim.simulator import CosimSimulation, carla
from scenic.simulators.metsr import (
    network_utils as metsr_network,
    routing_utils as routing,
)
from scenic.simulators.metsr.bridge_utils import METSRCosimState


def waypoint(road, lane, x, junction=False):
    return SimpleNamespace(
        road_id=road,
        lane_id=lane,
        section_id=0,
        s=float(x),
        is_junction=junction,
        transform=carla.Transform(carla.Location(x=float(x))),
    )


@pytest.fixture
def terminal_route():
    simulation = object.__new__(CosimSimulation)
    simulation.metsr_client = None
    source = waypoint(31, -1, 0)
    target = waypoint(48, -1, 20)
    simulation.metsr_state = METSRCosimState(
        metadata=metsr_network.NetworkMetadata(
            lane_mapping={
                "31_-1": ["incoming_4"],
                "48_-1": ["-48_4"],
            },
            lane_indices={
                "incoming_4": 0,
                "incoming_5": 1,
                "-48_4": 0,
                "-48_5": 1,
            },
            internal_lane_connections={":junction_0_0": [("incoming_4", "-48_4")]},
        )
    )
    simulation._carla_waypoints_by_key = {"48_-1": [target]}
    simulation.metsr_state.road_cache = {
        ("-48", 0): [(10.0, 0.0, 0.0), (30.0, 0.0, 0.0)],
    }
    simulation.map = SimpleNamespace(
        get_waypoint=lambda location, **kwargs: min(
            (source, target), key=lambda wp: abs(wp.transform.location.x - location.x)
        )
    )
    simulation.grp = SimpleNamespace(
        trace_route=lambda start, end: [(source, None), (target, None)],
    )
    obj = SimpleNamespace(
        name="car_4",
        carlaActor=SimpleNamespace(get_location=lambda: source.transform.location),
    )
    return simulation, obj, source, target


@pytest.mark.parametrize("on_connector", [False, True])
def test_destination_route_from_incoming_lane_or_connector(terminal_route, on_connector):
    simulation, obj, source, _ = terminal_route
    if on_connector:
        source.is_junction = True
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata,
            lane_mapping={
                **simulation.metsr_state.metadata.lane_mapping,
                "31_-1": [":junction_0_0"],
            },
        )

    locations = routing.generate_carla_destination_trajectory(simulation, "-48", obj)
    assert [location.x for location in locations] == (
        [20.0] if on_connector else [0.0, 20.0]
    )


def test_vehicle_on_destination_keeps_current_position(terminal_route):
    simulation, obj, _, target = terminal_route
    obj.carlaActor.get_location = lambda: target.transform.location
    locations = routing.generate_carla_destination_trajectory(simulation, "-48", obj)
    assert locations == [target.transform.location]


def test_carla_can_route_from_another_incoming_lane(terminal_route):
    simulation, obj, _, _ = terminal_route
    simulation.metsr_state.metadata = replace(
        simulation.metsr_state.metadata,
        lane_mapping={
            **simulation.metsr_state.metadata.lane_mapping,
            "31_-1": ["incoming_6"],
        },
        lane_indices={**simulation.metsr_state.metadata.lane_indices, "incoming_6": 2},
    )
    locations = routing.generate_carla_destination_trajectory(simulation, "-48", obj)
    assert [location.x for location in locations] == [0.0, 20.0]


def test_connector_mapping_does_not_constrain_carla_route(terminal_route):
    simulation, obj, source, _ = terminal_route
    source.is_junction = True
    simulation.metsr_state.metadata = replace(
        simulation.metsr_state.metadata,
        lane_mapping={
            **simulation.metsr_state.metadata.lane_mapping,
            "31_-1": [":other_0_0"],
        },
        internal_lane_connections={
            **simulation.metsr_state.metadata.internal_lane_connections,
            ":other_0_0": [("incoming_4", "other_0")],
        },
    )
    locations = routing.generate_carla_destination_trajectory(simulation, "-48", obj)
    assert [location.x for location in locations] == [20.0]


def test_destination_route_does_not_use_a_filtered_target_lane(terminal_route):
    simulation, obj, _, _ = terminal_route
    simulation.metsr_state.metadata = replace(
        simulation.metsr_state.metadata,
        lane_indices={
            lane: index
            for lane, index in simulation.metsr_state.metadata.lane_indices.items()
            if lane != "-48_4"
        },
    )
    with pytest.raises(SimulationCreationError, match="no mapped destination lane"):
        routing.generate_carla_destination_trajectory(simulation, "-48", obj)


def test_unmapped_connector_entrance_can_reach_mapped_destination(terminal_route):
    simulation, obj, source, _ = terminal_route
    source.is_junction = True
    simulation.metsr_state.metadata = replace(
        simulation.metsr_state.metadata,
        lane_mapping={
            **simulation.metsr_state.metadata.lane_mapping,
            "31_-1": [":unmapped_0_0"],
        },
    )
    locations = routing.generate_carla_destination_trajectory(simulation, "-48", obj)
    assert [location.x for location in locations] == [20.0]


def test_town05_connector_to_road_minus_48_builds_single_road_route(getAssetPath):
    xml = getAssetPath("maps/CARLA/Town05.net.xml")
    xodr = getAssetPath("maps/CARLA/Town05.xodr")
    simulation = object.__new__(CosimSimulation)
    simulation.metsr_client = None
    metadata = metsr_network.load_network_metadata(xml)
    simulation.metsr_state = METSRCosimState(metadata=metadata)
    simulation.map = carla.Map("Town05", Path(xodr).read_text())
    key = next(
        key
        for key, lanes in simulation.metsr_state.metadata.lane_mapping.items()
        if ":720_9_0" in lanes
    )
    road, lane = map(int, key.split("_"))
    # The first meter overlaps another turn geometrically. Use a point on the
    # distinct incoming movement so CARLA identifies the intended connector.
    start = simulation.map.get_waypoint_xodr(road, lane, 5.0)
    assert start is not None and start.is_junction
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
    # Supply an aligned target centerline locally; no METS-R server is needed.
    target_waypoints = sorted(
        simulation._carla_waypoints_by_key["48_-1"], key=lambda wp: wp.s
    )
    simulation.metsr_state.road_cache = {
        ("-48", 0): [
            (wp.transform.location.x, -wp.transform.location.y, wp.transform.location.z)
            for wp in target_waypoints
        ]
    }
    obj = SimpleNamespace(
        carlaActor=SimpleNamespace(get_location=lambda: start.transform.location)
    )
    locations = routing.generate_carla_destination_trajectory(simulation, "-48", obj)
    assert len(locations) > 2
    terminal = simulation.map.get_waypoint(locations[-1])
    assert (
        "-48_4"
        in simulation.metsr_state.metadata.lane_mapping[
            f"{terminal.road_id}_{terminal.lane_id}"
        ]
    )
