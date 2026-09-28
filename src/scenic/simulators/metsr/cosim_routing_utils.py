"""Map native METS-R roads and connectors to CARLA routes and placements."""

from dataclasses import dataclass
import math
import random

import carla
import networkx as nx

from scenic.core.object_types import Object
from scenic.core.simulators import SimulationCreationError
from scenic.domains.driving.roads import Road
from scenic.simulators.carla.utils import routing as carla_routing, utils

from . import cosim_utils, network

CARLA_OWNED_PROJECTION_TOLERANCE = 4.25


def mapped_lanes_for_carla_waypoint(state, waypoint):
    """Return raw SUMO lanes mapped from one projected CARLA waypoint."""
    if waypoint is None:
        return ()
    key = f"{waypoint.road_id}_{waypoint.lane_id}"
    lane_mapping = state.metadata.lane_mapping
    return tuple(str(lane) for lane in lane_mapping.get(key, ()))


@dataclass(frozen=True)
class LaneProjection:
    road_id: str | None = None
    lane_index: int | None = None
    distance: float = math.inf
    at_terminal: bool = False


def _nearest_lane_projection(state, client, position, candidates):
    """Project onto resolved native lanes, preserving first-candidate tie handling."""
    best = LaneProjection()
    for road, lane_index in candidates:
        distance, at_terminal = carla_routing.point_to_polyline_projection(
            position[:2], query_mapped_centerline(state, client, road, lane_index)
        )
        if distance < best.distance:
            best = LaneProjection(road, lane_index, distance, at_terminal)
    return best


def identify_nearest_road(state, client, obj: Object, roads: list[str]) -> str | None:
    """Resolve an ambiguous lane mapping by distance to each full lane."""
    candidates = []
    unlocated_internal_segments = set()
    for road_lane in roads:
        road, lane_index = network.mapped_lane_query(state, road_lane)
        if lane_index is None:
            if cosim_utils.is_controlled_internal_segment(state, road):
                unlocated_internal_segments.add(str(road))
        elif road in state.valid_roads or road in state.controlled_segments:
            candidates.append((road, lane_index))

    nearest = _nearest_lane_projection(
        state, client, getattr(obj, "position", obj), candidates
    )
    if nearest.road_id is None and unlocated_internal_segments:
        owners = {
            state.internal_edge_to_connector[segment]
            for segment in unlocated_internal_segments
        }
        # SUMO may split one CARLA junction lane into consecutive edges.
        # They are unambiguous for ownership when every edge has one owner;
        # the exact connector path is resolved separately from the pose.
        if len(owners) == 1:
            return min(unlocated_internal_segments)
    return nearest.road_id


def query_mapped_centerline(state, client, road, lane_index):
    """Cache the lane geometry returned by METS-R's centerline query."""
    cache_key = (road, lane_index)
    if cache_key not in state.road_cache:
        response = client.query_centerline(
            road, lane_index=lane_index, transform_coords=True
        )
        records = response["data"]
        state.road_cache[cache_key] = records[0].get("centerline", []) if records else []
    return state.road_cache[cache_key]


def normalized_connector_path_id(value):
    """Return a non-negative connector-local path ID, or ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if numeric < 0 or str(numeric) != str(value):
        return None
    return numeric


def connector_path_records(state, client, connector_id):
    """Return cached path records for one server-owned connector ID."""
    connector_id = str(connector_id)
    cache = state.connector_path_cache
    if connector_id in cache:
        return cache[connector_id]

    connector = state.connector_records.get(connector_id, {})
    paths = connector.get("paths")
    if paths is None:
        response = client.query_connector_path(connector_id)
        paths = []
        for record in response["data"]:
            if str(record.get("connectorId")) == connector_id:
                paths = record.get("paths", [record])
                break

    normalized = []
    for record in paths:
        path_id = normalized_connector_path_id(record.get("connectorPathId"))
        if path_id is None:
            continue
        item = dict(record, connectorId=connector_id, connectorPathId=path_id)
        for field in ("sourceRoadId", "targetRoadId"):
            if connector.get(field) is not None:
                item.setdefault(field, str(connector[field]))
        normalized.append(item)
    cache[connector_id] = tuple(normalized)
    return cache[connector_id]


def connector_path_for_mapped_lane(state, client, mapped_lane, connector_id=None):
    """Resolve a SUMO internal via-lane to its canonical connector path."""
    mapped_lane = str(mapped_lane)
    lane_pairs = state.metadata.internal_lane_connections.get(mapped_lane)
    if not lane_pairs:
        return None

    expected_pairs = {(str(source), str(target)) for source, target in lane_pairs}
    ownership = state.internal_edge_to_connector
    owner_ids = {str(owner) for owner in ownership.values()}
    if connector_id is not None:
        requested_owner = str(connector_id)
        if requested_owner not in owner_ids:
            return None
        owner_ids = {requested_owner}

    matches = {}
    for owner in owner_ids:
        owned_internal_edges = {
            str(edge)
            for edge, candidate_owner in ownership.items()
            if str(candidate_owner) == owner
        }
        for record in connector_path_records(state, client, owner):
            internal_edges = {str(edge) for edge in record.get("internalEdgeIds", ())}
            if internal_edges and not internal_edges & owned_internal_edges:
                continue
            via_lanes = {str(lane) for lane in record.get("viaLaneIds", ())}
            lane_pair = (
                str(record.get("sourceLaneId")),
                str(record.get("targetLaneId")),
            )
            if mapped_lane not in via_lanes and lane_pair not in expected_pairs:
                continue
            key = (owner, record["connectorPathId"])
            matches[key] = record
    if len(matches) != 1:
        return None
    (owner, path_id), record = next(iter(matches.items()))
    return owner, path_id, record


def query_connector_path_centerline(state, client, connector_id, connector_path_id):
    """Cache a connector movement's geometry by connector-local path ID."""
    connector_id = str(connector_id)
    connector_path_id = normalized_connector_path_id(connector_path_id)
    if connector_path_id is None:
        return []
    cache = state.connector_centerline_cache
    cache_key = (connector_id, connector_path_id)
    if cache_key not in cache:
        response = client.query_centerline(
            connector_id, lane_index=-1, transform_coords=True
        )
        record = next(
            (
                record
                for record in response["data"]
                if str(record.get("segmentId")) == connector_id
            ),
            {},
        )
        centerlines = record.get("centerlines", ())
        cache[cache_key] = (
            list(centerlines[connector_path_id])
            if connector_path_id < len(centerlines)
            else []
        )
    return cache[cache_key]


def nearest_mapped_connector_path_projection(
    state, client, position, mapped_lanes, connector_id=None
):
    """Return the closest mapped connector path and projection metadata."""
    best = (None, None, math.inf, False, None)
    seen = set()
    for mapped_lane in mapped_lanes:
        resolved = connector_path_for_mapped_lane(
            state, client, mapped_lane, connector_id=connector_id
        )
        if resolved is None:
            continue
        candidate_connector, path_id, record = resolved
        key = (candidate_connector, path_id)
        if key in seen:
            continue
        seen.add(key)
        distance, at_terminal = carla_routing.point_to_polyline_projection(
            position[:2],
            query_connector_path_centerline(state, client, candidate_connector, path_id),
        )
        if distance < best[2]:
            best = (
                candidate_connector,
                path_id,
                distance,
                at_terminal,
                record,
            )
    return best


def distance_to_mapped_road(state, client, position, mapped_lanes, target_road):
    """Measure a point against mapped lanes belonging to one METS-R road."""
    return nearest_mapped_lane(state, client, position, mapped_lanes, target_road)[1]


def nearest_mapped_lane(state, client, position, mapped_lanes, target_road):
    """Return the closest mapped METS-R lane index and its distance."""
    lane_index, distance, _ = nearest_mapped_lane_projection(
        state, client, position, mapped_lanes, target_road
    )
    return lane_index, distance


def nearest_mapped_lane_projection(state, client, position, mapped_lanes, target_road):
    """Return the closest mapped lane, distance, and terminal status."""
    target_road = str(target_road)
    candidates = (
        (road, lane_index)
        for road, lane_index in (
            network.mapped_lane_query(state, lane) for lane in mapped_lanes
        )
        if road == target_road and lane_index is not None
    )
    nearest = _nearest_lane_projection(state, client, position, candidates)
    return nearest.lane_index, nearest.distance, nearest.at_terminal


def generate_carla_destination_trajectory(
    simulation, destination_road_id, obj, *, target_start=None
):
    """Choose a CARLA route to any reachable driving lane on the target road."""
    destination = str(destination_road_id)
    if target_start is None:
        target_start = obj.carlaActor.get_location()

    def on_destination(waypoint, location=None):
        if waypoint is None:
            return False
        key = f"{waypoint.road_id}_{waypoint.lane_id}"
        lanes = [
            lane
            for lane in simulation.metsr_state.metadata.lane_mapping.get(key, ())
            if network.mapped_lane_query(simulation.metsr_state, lane)[1] is not None
        ]
        roads = {
            network.mapped_lane_query(simulation.metsr_state, lane)[0] for lane in lanes
        }
        if destination not in roads:
            return False
        if len(roads) == 1:
            return True
        # A single OpenDRIVE lane can cover several SUMO road fragments.
        # Its mapping alone must not imply arrival on a distant fragment.
        location = waypoint.transform.location if location is None else location
        point = (location.x, -location.y)
        distances = {
            road: distance_to_mapped_road(
                simulation.metsr_state, simulation.metsr_client, point, lanes, road
            )
            for road in roads
        }
        return distances[destination] + 1e-6 < min(
            distance for road, distance in distances.items() if road != destination
        )

    start_waypoint = simulation.map.get_waypoint(
        target_start, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if on_destination(start_waypoint, target_start):
        # Arrival is recorded by normal pose synchronization. Do not plan a
        # loop to an anchor behind a car which already reached its target.
        return [target_start]

    target_lanes = sorted(
        lane
        for lane in simulation.metsr_state.metadata.carla_keys_by_lane
        if network.mapped_lane_query(simulation.metsr_state, lane)[0] == destination
        and network.mapped_lane_query(simulation.metsr_state, lane)[1] is not None
    )
    best_path, best_length = None, math.inf
    failures = []
    for lane in target_lanes:
        anchor = lane_anchor_waypoint(simulation, lane, final=True)
        if anchor is None:
            continue
        try:
            trace = list(
                simulation.grp.trace_route(target_start, anchor.transform.location)
            )
        except (nx.NetworkXException, KeyError, IndexError) as error:
            failures.append(f"{lane}: {error}")
            continue
        if not trace or not on_destination(trace[-1][0]):
            failures.append(f"{lane}: CARLA did not reach the destination road")
            continue
        # Only the destination is constrained. CARLA selects all intermediate
        # roads; no SUMO macro-route matching or lane-chain stitching runs.
        locations = [waypoint.transform.location for waypoint, _ in trace]
        previous = target_start
        length = 0.0
        for location in locations:
            length += math.dist(
                (previous.x, previous.y, previous.z),
                (location.x, location.y, location.z),
            )
            previous = location
        if length < best_length:
            # Traffic Manager's ImportPath selects junction branches by
            # their exit road relative to the next imported coordinate.
            # Dense points inside a junction instead make it choose the
            # exit nearest that interior point, which can be another turn.
            # Keep the CARLA-planned exit anchors and ordinary road curves;
            # use the complete trace above when comparing route lengths.
            anchors = []
            for waypoint, _ in trace:
                if waypoint.is_junction:
                    continue
                projected = simulation.map.get_waypoint(waypoint.transform.location)
                if projected is None or projected.is_junction:
                    # A road entry may share its exact coordinates with the
                    # junction endpoint. Use the next point inside the road.
                    continue
                anchors.append(waypoint)
            if not anchors or not on_destination(anchors[-1]):
                failures.append(f"{lane}: no unambiguous destination road anchor")
                continue
            best_path = [waypoint.transform.location for waypoint in anchors]
            best_length = length
    if best_path is not None:
        return best_path
    detail = failures[0] if failures else "no mapped destination lane has a CARLA anchor"
    raise SimulationCreationError(
        f"CARLA cannot route {obj} from {target_start} to METS-R destination "
        f"road {destination}: {detail}"
    )


def lane_anchor_waypoint(simulation, sumo_lane, final=False):
    if simulation._carla_waypoints_by_key is None:
        simulation._carla_waypoints_by_key = carla_routing.index_carla_waypoints(
            simulation.map, simulation.grp
        )
    road, lane_index = network.mapped_lane_query(simulation.metsr_state, sumo_lane)
    if lane_index is None:
        return None
    centerline = query_mapped_centerline(
        simulation.metsr_state, simulation.metsr_client, road, lane_index
    )
    target = carla_routing.polyline_fraction(centerline, 0.9 if final else 0.7)
    if target is None:
        return None
    target = (target[0], -target[1], target[2])

    candidates = []
    for carla_key in simulation.metsr_state.metadata.carla_keys_by_lane.get(
        str(sumo_lane), ()
    ):
        candidates.extend(simulation._carla_waypoints_by_key.get(carla_key, ()))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda waypoint: math.dist(
            (
                waypoint.transform.location.x,
                waypoint.transform.location.y,
                waypoint.transform.location.z,
            ),
            target,
        ),
    )


def degenerate_connector_spawn_orientation(simulation, state, position):
    """Recover a zero-length connector's direction from its mapped lane.

    SUMO continuity connectors can contain only repeated points. Native
    displacement into such a connector then reflects lateral lane placement,
    not a driving tangent. Keep ordinary native headings; only recover this
    undefined direction when the exact path topology and 3D pose agree.
    """
    connector_id = state.get("connectorId")
    path_id = normalized_connector_path_id(state.get("connectorPathId"))
    if path_id is None or not cosim_utils.is_controlled_connector_id(
        simulation.metsr_state, connector_id
    ):
        return None
    connector_id = str(connector_id)
    record = simulation.metsr_state.connector_records[connector_id]
    source, target = record.get("sourceRoadId"), record.get("targetRoadId")
    if source is None or target is None:
        return None
    mapped_endpoints = set()
    connections = simulation.metsr_state.metadata.internal_lane_connections
    for via_lane in state.get("connectorPathViaLaneIds", ()):
        for source_lane, target_lane in connections.get(str(via_lane), ()):
            if network.mapped_lane_query(simulation.metsr_state, source_lane)[0] == str(
                source
            ) and network.mapped_lane_query(simulation.metsr_state, target_lane)[
                0
            ] == str(
                target
            ):
                mapped_endpoints.update((str(source_lane), str(target_lane)))
    if not mapped_endpoints:
        return None

    centerline = query_connector_path_centerline(
        simulation.metsr_state, simulation.metsr_client, connector_id, path_id
    )
    point = (position.x, position.y, position.z)
    if (
        carla_routing.point_to_polyline_distance_3d(point, centerline)
        > CARLA_OWNED_PROJECTION_TOLERANCE
    ):
        return None
    points = [tuple(map(float, vertex[:3])) for vertex in centerline]
    if any(math.dist(points[0], vertex) > 1e-6 for vertex in points[1:]):
        return None
    waypoint, error = carla_routing.projected_carla_driving_waypoint(
        simulation.map, carla.Location(position.x, -position.y, position.z)
    )
    if waypoint is None or error > CARLA_OWNED_PROJECTION_TOLERANCE:
        return None
    if not mapped_endpoints.intersection(
        mapped_lanes_for_carla_waypoint(simulation.metsr_state, waypoint)
    ):
        return None
    return utils.carlaToScenicOrientation(waypoint.transform.rotation)


def native_handoff_target(simulation, obj, location):
    """Resolve an outgoing NPC's native placement; ordinary poses need no hint.

    CARLA's current waypoint identifies a possible boundary crossing. Only
    an ordinary road outside the controlled region triggers handback;
    junctions and ambiguous/unmapped poses use ordinary native association.
    """
    if obj is getattr(simulation, "ego", None):
        return None
    waypoint, distance = carla_routing.projected_carla_driving_waypoint(
        simulation.map, location
    )
    if (
        waypoint is None
        or waypoint.is_junction
        or distance > CARLA_OWNED_PROJECTION_TOLERANCE
    ):
        return None
    mapped_lanes = mapped_lanes_for_carla_waypoint(simulation.metsr_state, waypoint)
    candidates = []
    for mapped_lane in mapped_lanes:
        road, index = network.mapped_lane_query(simulation.metsr_state, mapped_lane)
        if (
            road is not None
            and not str(road).startswith(":")
            and index is not None
            and index >= 0
        ):
            candidates.append((str(road), index, mapped_lane))
    if not candidates or all(
        road in simulation.metsr_state.controlled_roads for road, _, _ in candidates
    ):
        return None
    if len(candidates) == 1:
        road, lane, _ = candidates[0]
    else:
        position = (location.x, -location.y, location.z)
        road = identify_nearest_road(
            simulation.metsr_state,
            simulation.metsr_client,
            position,
            [entry[2] for entry in candidates],
        )
        if road is None or str(road) in simulation.metsr_state.controlled_roads:
            return None
        road = str(road)
        lane, _ = nearest_mapped_lane(
            simulation.metsr_state,
            simulation.metsr_client,
            position,
            mapped_lanes,
            road,
        )
        if lane is None:
            return None
    # CARLA and SUMO place the start of a junction at slightly different
    # positions. A newly admitted connector may still project onto its
    # upstream CARLA road; handing it back there causes immediate re-entry
    # and repeated promotion. This is entry, not an outgoing boundary.
    current = simulation.metsr_state.authoritative_segments.get(obj)
    connector = simulation.metsr_state.connector_records.get(current, {})
    if str(connector.get("sourceRoadId")) == road:
        return None
    return road, lane, None


@dataclass(frozen=True)
class BubbleRoadSelection:
    """Scenic and native bubble selection, before ownership changes are committed."""

    roads: tuple[Road, ...]
    native_road_ids: frozenset[str]
    entering: tuple[str, ...]
    leaving: tuple[str, ...]


def classify_bubble_roads(state, bubble_roads, scenic_roads_by_id):
    """Return repaired bubble mappings and the proposed ownership changes."""
    road_ids = [str(road.id) for road in bubble_roads]
    native_ids = tuple(
        dict.fromkeys(
            native_id
            for road in bubble_roads
            for native_id in network.map_scenic_to_metsr_road(state, road)
        )
    )
    repaired = network.repair_mapping(state, road_ids, native_ids, scenic_roads_by_id)
    roads_by_id = {str(road.id): road for road in (*bubble_roads, *repaired)}
    active = frozenset(native_ids)
    return BubbleRoadSelection(
        roads=tuple(roads_by_id.values()),
        native_road_ids=active,
        entering=tuple(road for road in native_ids if road not in state.controlled_roads),
        leaving=tuple(road for road in state.controlled_roads if road not in active),
    )


def vehicle_destination_road(obj, vehicle_state):
    destination = vehicle_state.get("destinationRoadId")
    if destination is None or isinstance(destination, bool) or str(destination) == "":
        raise SimulationCreationError(
            f"METS-R vehicle query omitted destinationRoadId for {obj}"
        )
    return str(destination)


def select_spawn_placement(simulation, obj, origin=None):
    """Map a Scenic spawn to a native road lane or connector movement."""
    lane = simulation.network_helper._nearest_lane(obj)
    if lane is None:
        raise SimulationCreationError(
            f"Unable to identify a Scenic origin lane for {obj}"
        )
    lane_key = f"{lane.road.id}_{lane.id}"
    mapped_lanes = simulation.metsr_state.metadata.lane_mapping.get(lane_key, ())
    if not mapped_lanes:
        raise SimulationCreationError(
            f"No METS-R lane mapping exists for Scenic lane {lane_key}"
        )
    road = (
        str(origin)
        if origin is not None
        else identify_nearest_road(
            simulation.metsr_state, simulation.metsr_client, obj, mapped_lanes
        )
    )
    lane_index, distance, at_terminal = nearest_mapped_lane_projection(
        simulation.metsr_state,
        simulation.metsr_client,
        obj.position,
        mapped_lanes,
        road,
    )
    connector, path_id, connector_distance, connector_terminal, path = (
        nearest_mapped_connector_path_projection(
            simulation.metsr_state, simulation.metsr_client, obj.position, mapped_lanes
        )
        if origin is None
        else (None, None, math.inf, False, None)
    )
    if (
        connector is not None
        and math.isfinite(connector_distance)
        and connector_distance <= distance
    ):
        segment = str(connector)
        lane_index = None
        distance, at_terminal = connector_distance, connector_terminal
        route_origin, route_query_origin = path.get("sourceRoadId"), path.get(
            "targetRoadId"
        )
        if route_origin is None or route_query_origin is None:
            raise SimulationCreationError(
                f"Connector {segment!r} path {path_id} omitted its source or target road"
            )
        route_origin, route_query_origin = str(route_origin), str(route_query_origin)
    elif road is not None and lane_index is not None and math.isfinite(distance):
        segment = route_origin = route_query_origin = str(road)
        path_id = None
    else:
        raise SimulationCreationError(
            f"No queryable METS-R road lane or connector path exists for {obj} "
            f"on Scenic lane {lane_key}; mapped lanes were {mapped_lanes}"
        )

    is_cosim = (
        segment in simulation.metsr_state.controlled_roads
        or cosim_utils.is_controlled_connector_id(simulation.metsr_state, segment)
    )
    # Native vehicles enter the origin road's pending queue, so their sampled
    # coordinates need not project onto that road. Co-simulation spawns do.
    if is_cosim:
        limit = carla_routing.source_projection_error_limit(
            simulation.map,
            obj.position,
            lane_key,
            max(3.0, float(obj.width) * 1.5),
            at_terminal,
        )
        if distance > limit:
            raise SimulationCreationError(
                f"Scenic spawn for {obj} is {distance:.2f} m from mapped METS-R "
                f"segment {segment}; refusing a remote projection; "
                f"Scenic lane={lane_key}, limit={limit:.2f} m"
            )
    return {
        "segment_id": segment,
        "lane_index": lane_index,
        "connector_path_id": path_id,
        "distance": distance,
        "lane_key": lane_key,
        "route_origin": route_origin,
        "route_query_origin": route_query_origin,
        "is_cosim": is_cosim,
    }


def choose_destination_route(simulation, obj, placement, attempts=100):
    """Sample a reachable destination using the mapped CARLA spawn points."""
    origin = placement["route_origin"]
    query_origin = placement["route_query_origin"]
    for _ in range(attempts):
        destination_spawn = random.choice(simulation.spawn_points)
        waypoint = simulation.map.get_waypoint(destination_spawn.location)
        if waypoint is None:
            continue
        lane_key = f"{waypoint.road_id}_{waypoint.lane_id}"
        mapped_lanes = simulation.metsr_state.metadata.lane_mapping.get(lane_key, ())
        destination = identify_nearest_road(
            simulation.metsr_state,
            simulation.metsr_client,
            utils.carlaToScenicPosition(destination_spawn.location),
            mapped_lanes,
        )
        if destination is None:
            continue
        candidate = simulation.metsr_client.query_route_between_roads(
            query_origin, destination
        )["data"][0]
        if not isinstance(candidate, dict):
            continue
        roads = [str(road) for road in candidate.get("roadIds", ())]
        if not roads or roads[0] != query_origin:
            continue
        if origin != query_origin:
            roads.insert(0, origin)
        if len(roads) > 1:
            return roads
    raise SimulationCreationError(
        f"Failed to generate a multi-road trajectory for {obj} at "
        f"{(obj.position.x, obj.position.y)} from mapped segment {placement['segment_id']}"
    )


def validate_spawn_placement(obj, requested, record, state):
    """Check native initialization kept the selected lane/path near the spawn."""
    if "x" not in state or "y" not in state:
        raise SimulationCreationError(
            f"METS-R did not return an on-network placement for {obj}"
        )
    segment = requested["segment_id"]
    if str(record.get("segmentId")) != segment:
        raise SimulationCreationError(
            f"METS-R placed {obj} on segment {record.get('segmentId')!r}, "
            f"but Scenic selected segment {segment!r}"
        )
    path_id, lane_index = requested["connector_path_id"], requested["lane_index"]
    if path_id is not None:
        if (
            record.get("laneIndex") not in (None, -1)
            or record.get("connectorPathId") != path_id
        ):
            raise SimulationCreationError(
                f"METS-R placed {obj} on connector path {record.get('connectorPathId')!r} "
                f"of {segment}, but Scenic lane {requested['lane_key']} maps to "
                f"connectorPathId {path_id}"
            )
    elif lane_index >= 0 and record.get("laneIndex") != lane_index:
        raise SimulationCreationError(
            f"METS-R placed {obj} on lane {record.get('laneIndex')} of road {segment}, "
            f"but Scenic lane {requested['lane_key']} maps to METS-R lane {lane_index}"
        )
    backward_shift = abs(float(record.get("backwardShift", 0.0)))
    displacement = math.hypot(state["x"] - obj.position.x, state["y"] - obj.position.y)
    limit = requested["distance"] + backward_shift + 0.25
    if displacement > limit:
        raise SimulationCreationError(
            f"METS-R projected {obj} {displacement:.2f} m from its Scenic spawn on "
            f"segment {segment} (reported backward shift {backward_shift:.2f} m); "
            f"allowed displacement={limit:.2f} m"
        )
