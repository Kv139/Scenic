"""Select the roads and intersections in the co-simulation bubble."""

import carla

from scenic.core.regions import CircularRegion
from scenic.core.simulators import SimulationCreationError
from scenic.domains.driving.roads import Intersection, Road
from scenic.simulators.carla.utils import utils


def get_bubble_intersections(simulation, bubble_roads: list[Road]) -> list[Intersection]:
    """Include intersections attached to the selected road component.

    Every incident intersection is needed for METS-R ownership consistency.
    XY overlap alone is insufficient because it can select another level.
    """
    road_ids = {str(road.id) for road in bubble_roads}
    bubble_intersections = []
    for intersection in simulation.network_helper.network_intersections:
        attached_roads = list(intersection.roads)
        attached_roads.extend(
            maneuver.connectingLane.road
            for maneuver in intersection.maneuvers
            if maneuver.connectingLane is not None
        )
        if any(str(road.id) in road_ids for road in attached_roads):
            bubble_intersections.append(intersection)
    return bubble_intersections


def get_bubble_roads(
    simulation, bubble_region: CircularRegion | None = None
) -> list[Road]:
    """Select nearby roads connected to the ego's current CARLA road layer."""
    if bubble_region is None:
        bubble_region = simulation.ego.bubble
    actor = getattr(simulation.ego, "carlaActor", None)
    location = (
        actor.get_location()
        if actor is not None
        else utils.scenicToCarlaLocation(simulation.ego.position)
    )
    waypoint = simulation.map.get_waypoint(
        location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if waypoint is None:
        raise SimulationCreationError(
            "Cannot determine the ego's CARLA road for bubble selection"
        )
    roads = simulation.network_helper._get_bubble_roads(
        bubble_region, anchor_road_id=waypoint.road_id
    )
    if not roads:
        raise SimulationCreationError(
            "The ego's CARLA road is absent from the Scenic bubble: "
            f"road={waypoint.road_id}, location={location}"
        )
    return roads
