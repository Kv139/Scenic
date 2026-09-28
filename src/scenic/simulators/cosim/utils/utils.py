import math

import numpy as np


def point_to_segment_distance(px, py, ax, ay, bx, by) -> float:
    """Perpendicular distance from (px,py) to the segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    seg_sq = dx * dx + dy * dy
    if seg_sq == 0.0:  # degenerate segment
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_sq
    t = max(0.0, min(1.0, t))  # clamp so we stay on the segment
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def polyline_distance(point, polyline) -> float:
    """Distance from a point to the nearest part of a centerline polyline."""
    px, py = point
    if not polyline:
        return math.inf
    if len(polyline) == 1:
        return math.hypot(px - polyline[0][0], py - polyline[0][1])
    return min(
        point_to_segment_distance(px, py, ax, ay, bx, by)
        for (ax, ay), (bx, by) in zip(polyline, polyline[1:])
    )


def circle_aabb_intersects(cx, cy, radius, bounds) -> bool:
    """Conservative circle/bounding-box overlap test.

    Returns True whenever the circle *could* touch the box, so it is safe to use as a
    cheap pre-filter in front of an exact polygon intersection: it never rejects a
    shape the exact test would have accepted.
    """
    xmin, ymin, xmax, ymax = bounds
    dx = max(xmin - cx, 0.0, cx - xmax)
    dy = max(ymin - cy, 0.0, cy - ymax)
    return dx * dx + dy * dy <= radius * radius


def within_threshold_to(object, cars, threshold=None, verbose=False) -> bool:
    """Return the nearest other actor, optionally restricted by threshold."""
    is_close = False
    object_pos = np.array(object.position)
    obj_distances = {}
    danger_veh = None
    min_dist = math.inf
    for car in cars:
        if car != object:
            dist = np.linalg.norm(np.array(car.position) - object_pos)
            if dist < min_dist:
                danger_veh = car
                min_dist = dist
            obj_distances[car.name] = dist
    if danger_veh is not None:
        is_close = threshold is None or min_dist <= threshold
    if verbose:
        if obj_distances:
            print(f"Min Distance for {object} was: {min_dist}")
        else:
            print(f"No other CARLA actors were available near {object}")
    return is_close, danger_veh


def get_metsr_rotation(carla_yaw):
    """
    Invert carla_yaw = (bearing - 90) % 360
    to recover the original METSR compass bearing.
    """
    # ensure 0 ≤ yaw < 360
    carla_yaw = carla_yaw % 360
    # invert the shift of -90°
    return (carla_yaw + 90) % 360


def get_carla_light_state(light) -> dict:
    light_state_dict = {
        "green_time": light.get_green_time(),
        "red_time": light.get_red_time(),
        "yellow_time": light.get_yellow_time(),
        "state": light.get_state(),
    }

    return light_state_dict


def disable_carla_autopilot(obj, tm) -> bool:
    if hasattr(obj, "carlaActor"):
        if obj.carlaActor != None:
            obj.carlaActor.set_autopilot(False, tm.get_port())
            return True
    else:
        return False
