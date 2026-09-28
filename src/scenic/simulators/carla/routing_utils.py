"""Reusable CARLA waypoint projection, indexing, and route geometry."""

import math

import carla

from scenic.simulators.carla import coordinate_utils as utils


def carla_waypoint_distance(location, waypoint):
    """Measure an actor pose against its projected CARLA waypoint."""
    if waypoint is None:
        return math.inf
    projected = waypoint.transform.location
    return math.dist(
        (location.x, location.y, location.z),
        (projected.x, projected.y, projected.z),
    )


def projected_carla_driving_waypoint(carla_map, location):
    """Return CARLA's nearest driving waypoint and the projection error."""
    if carla_map is None:
        return None, math.inf
    waypoint = carla_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    return waypoint, carla_waypoint_distance(location, waypoint)


def source_projection_error_limit(carla_map, position, lane_key, base_limit, at_terminal):
    """Allow a bounded map-end mismatch only on the confirmed CARLA lane."""
    if not at_terminal:
        return base_limit
    waypoint = carla_map.get_waypoint(
        utils.scenicToCarlaLocation(position),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is not None and f"{waypoint.road_id}_{waypoint.lane_id}" == lane_key:
        return base_limit + float(waypoint.lane_width)
    return base_limit


def point_to_polyline_distance_3d(point, polyline):
    """Measure a pose against path geometry with an explicit elevation."""
    if not polyline or any(len(vertex) < 3 for vertex in polyline):
        return math.inf
    points = [tuple(map(float, vertex[:3])) for vertex in polyline]
    if not all(math.isfinite(value) for vertex in [point, *points] for value in vertex):
        return math.inf
    if len(points) == 1:
        return math.dist(point, points[0])
    best = math.inf
    for start, end in zip(points, points[1:]):
        delta = tuple(b - a for a, b in zip(start, end))
        length_squared = sum(value * value for value in delta)
        ratio = (
            sum((p - a) * d for p, a, d in zip(point, start, delta)) / length_squared
            if length_squared
            else 0.0
        )
        ratio = min(1.0, max(0.0, ratio))
        closest = tuple(a + ratio * d for a, d in zip(start, delta))
        best = min(best, math.dist(point, closest))
    return best


def point_to_polyline_distance(point, polyline):
    return point_to_polyline_projection(point, polyline)[0]


def point_to_polyline_projection(point, polyline):
    """Return distance to a polyline and whether its nearest point is an end."""
    if not polyline:
        return math.inf, False

    # Remove consecutive duplicate control points so terminal detection is
    # based on the first/last usable point rather than a zero-length edge.
    points = []
    for item in polyline:
        candidate = tuple(item[:2])
        if not points or candidate != points[-1]:
            points.append(candidate)
    if len(points) == 1:
        return math.dist(point, points[0]), True

    px, py = point
    best = math.inf
    closest = None
    for start, end in zip(points, points[1:]):
        sx, sy = start
        ex, ey = end
        dx, dy = ex - sx, ey - sy
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            distance = math.hypot(px - sx, py - sy)
        else:
            ratio = ((px - sx) * dx + (py - sy) * dy) / length_squared
            ratio = min(1.0, max(0.0, ratio))
            closest_x = sx + ratio * dx
            closest_y = sy + ratio * dy
            distance = math.hypot(px - closest_x, py - closest_y)
        if distance < best:
            best = distance
            closest = (closest_x, closest_y) if length_squared else start

    endpoint_epsilon = 1e-6
    at_terminal = closest is not None and (
        math.dist(closest, points[0]) <= endpoint_epsilon
        or math.dist(closest, points[-1]) <= endpoint_epsilon
    )
    return best, at_terminal


def polyline_fraction(polyline, fraction):
    points = [tuple(point[:3]) for point in polyline]
    if not points:
        return None
    if len(points) == 1:
        return points[0]
    lengths = [math.dist(start, end) for start, end in zip(points, points[1:])]
    target = sum(lengths) * fraction
    traversed = 0.0
    for start, end, length in zip(points, points[1:], lengths):
        if length == 0:
            continue
        if traversed + length >= target:
            ratio = (target - traversed) / length
            return tuple(
                start[axis] + ratio * (end[axis] - start[axis]) for axis in range(3)
            )
        traversed += length
    return points[-1]


def index_carla_waypoints(carla_map, route_planner):
    """Index map and route-planner waypoints by their CARLA road and lane IDs."""
    resolution = max(0.5, float(getattr(route_planner, "_sampling_resolution", 2.0)))
    waypoints = list(carla_map.generate_waypoints(resolution))
    for segment in getattr(route_planner, "_topology", ()):
        waypoints.extend((segment["entry"], *segment["path"], segment["exit"]))
    index = {}
    seen = set()
    for waypoint in waypoints:
        identity = (
            waypoint.road_id,
            waypoint.section_id,
            waypoint.lane_id,
            round(float(waypoint.s), 3),
        )
        if identity in seen:
            continue
        seen.add(identity)
        key = f"{waypoint.road_id}_{waypoint.lane_id}"
        index.setdefault(key, []).append(waypoint)
    return index
