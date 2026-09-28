"""Scenic road topology and selection of the CARLA co-simulation bubble."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import carla

from scenic.core.regions import CircularRegion
from scenic.core.simulators import SimulationCreationError
from scenic.domains.driving.roads import Intersection, Road
from scenic.simulators.carla import coordinate_utils


@dataclass(frozen=True)
class BubbleTopology:
    """Static Scenic road indexes reused across bubble selections."""

    roads: tuple[Road, ...]
    intersections: tuple[Intersection, ...]
    roads_by_id: Mapping[str, Road]
    adjacency: Mapping[str, frozenset[str]]

    @classmethod
    def from_network(cls, network):
        roads = tuple(network.allRoads)
        roads_by_id = {str(road.id): road for road in roads}
        adjacency = {road_id: set() for road_id in roads_by_id}
        for road in roads:
            road_id = str(road.id)
            for lane in road.lanes:
                for neighbor in (lane._predecessor, lane._successor):
                    neighbor_road = getattr(neighbor, "road", None)
                    if neighbor_road is None:
                        continue
                    neighbor_id = str(neighbor_road.id)
                    if neighbor_id in adjacency:
                        adjacency[road_id].add(neighbor_id)
                        adjacency[neighbor_id].add(road_id)
        return cls(
            roads=roads,
            intersections=tuple(network.intersections),
            roads_by_id=MappingProxyType(roads_by_id),
            adjacency=MappingProxyType(
                {
                    road_id: frozenset(neighbors)
                    for road_id, neighbors in adjacency.items()
                }
            ),
        )


def connected_bubble_roads(topology, bubble_region, anchor_road_id=None):
    """Select the ego's local component, retaining ramps and incoming traffic.

    XY overlap alone includes disconnected overpasses. Undirected lane links
    retain roads which can interact with the ego without an elevation cutoff.
    """
    candidates = [road for road in topology.roads if road.intersects(bubble_region)]
    if anchor_road_id is None:
        return candidates
    candidate_ids = {str(road.id) for road in candidates}
    selected = set()
    pending = [str(anchor_road_id)]
    while pending:
        road_id = pending.pop()
        if road_id in selected or road_id not in candidate_ids:
            continue
        selected.add(road_id)
        pending.extend(topology.adjacency[road_id] - selected)
    return [road for road in candidates if str(road.id) in selected]


def intersections_for_roads(topology, roads):
    """Include every intersection attached to the selected road component."""
    road_ids = {str(road.id) for road in roads}
    intersections = []
    for intersection in topology.intersections:
        attached_roads = list(intersection.roads)
        attached_roads.extend(
            maneuver.connectingLane.road
            for maneuver in intersection.maneuvers
            if maneuver.connectingLane is not None
        )
        if any(str(road.id) in road_ids for road in attached_roads):
            intersections.append(intersection)
    return intersections


def get_bubble_roads(simulation, bubble_region: CircularRegion | None = None):
    """Select nearby roads connected to the ego's current CARLA road layer."""
    if bubble_region is None:
        bubble_region = simulation.ego.bubble
    actor = getattr(simulation.ego, "carlaActor", None)
    location = (
        actor.get_location()
        if actor is not None
        else coordinate_utils.scenicToCarlaLocation(simulation.ego.position)
    )
    waypoint = simulation.map.get_waypoint(
        location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if waypoint is None:
        raise SimulationCreationError(
            "Cannot determine the ego's CARLA road for bubble selection"
        )
    roads = connected_bubble_roads(
        simulation.bubble_topology, bubble_region, anchor_road_id=waypoint.road_id
    )
    if not roads:
        raise SimulationCreationError(
            "The ego's CARLA road is absent from the Scenic bubble: "
            f"road={waypoint.road_id}, location={location}"
        )
    return roads
