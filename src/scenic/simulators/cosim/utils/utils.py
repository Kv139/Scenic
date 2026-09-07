import xml.etree.ElementTree as ET
import math
import os
import numpy as np
import carla


def point_to_segment_distance(px, py, ax, ay, bx, by) -> float:
    """Perpendicular distance from (px,py) to the segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    seg_sq = dx * dx + dy * dy
    if seg_sq == 0.0:                       # degenerate segment
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_sq
    t = max(0.0, min(1.0, t))               # clamp so we stay on the segment
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

def generate_map(map):
    try:
        tree = ET.parse(map)
    except FileNotFoundError:
        print(f"Could not find map: {map} from {os.getcwd()}")
        return {}
    
    root = tree.getroot()
    lane_mappings = {}
    edges = root.iterfind("edge")

    for edge in edges: 
    
        lanes = edge.findall('lane')
        for lane in lanes:
            metrs_lane = lane.attrib.get("id")
            params = lane.findall('param')

            for param in params:
                if param.get('key') == "origId":
                    orig_id = param.get('value')
                    orig_id = orig_id.split()
                    if isinstance(orig_id, list):
                        for id in orig_id:
                            if id in lane_mappings:
                                lane_mappings[id].append(metrs_lane)
                            else:
                                lane_mappings[id] = [metrs_lane]
                    else:
                        if orig_id in lane_mappings:
                            lane_mappings[orig_id].append(metrs_lane)
                        else:
                            lane_mappings[orig_id] = [metrs_lane]
            # else:
            #     print(f"Skipping lane: {lane.attrib.get('id')}")

    if lane_mappings == {}:
        print(f"An occured attempting to process map: {map}")

    return lane_mappings


def generate_metsr_lane_index_map(map):
    """Map SUMO lane IDs to the filtered lane indices used by METS-R.

    SUMO lane suffixes are not necessarily METS-R lane indices: METS-R drops
    non-driving lanes before sorting the remaining lanes by their SUMO suffix.
    For example, if ``road_0`` and ``road_1`` are shoulders, ``road_2`` is
    queried from METS-R with ``laneIndex=0``, not ``laneIndex=2``.
    """
    try:
        tree = ET.parse(map)
    except FileNotFoundError:
        print(f"Could not find map: {map} from {os.getcwd()}")
        return {}

    lane_indices = {}
    for edge in tree.getroot().iterfind("edge"):
        if (edge.attrib.get("function") or "").lower() == "internal":
            continue
        edge_type = edge.attrib.get("type")
        if edge_type is None or not any(
            marker in edge_type.lower() for marker in ("highway", "driving")
        ):
            continue

        driving_lanes = []
        for lane in edge.findall("lane"):
            lane_type = lane.attrib.get("type")
            if lane_type is not None and lane_type.lower() != "driving":
                continue
            lane_id = lane.attrib.get("id")
            if not lane_id:
                continue
            try:
                suffix = int(lane_id.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            driving_lanes.append((suffix, lane_id))

        for lane_index, (_, lane_id) in enumerate(sorted(driving_lanes)):
            lane_indices[lane_id] = lane_index

    return lane_indices


def generate_metsr_lane_connection_map(map):
    """Return direct driving-lane connections between METS-R macro roads.

    The SUMO ``connection`` records use each lane's raw ``index`` attribute,
    while METS-R exposes a compact index after filtering non-driving lanes.
    Keep full SUMO lane IDs in this map so callers cannot accidentally confuse
    the two index spaces.
    """
    try:
        tree = ET.parse(map)
    except FileNotFoundError:
        print(f"Could not find map: {map} from {os.getcwd()}")
        return {}

    root = tree.getroot()
    driving_lane_ids = set(generate_metsr_lane_index_map(map))
    lane_by_road_and_index = {}
    for edge in root.iterfind("edge"):
        road_id = edge.attrib.get("id")
        if not road_id or road_id.startswith(":"):
            continue
        for lane in edge.findall("lane"):
            lane_id = lane.attrib.get("id")
            if lane_id not in driving_lane_ids:
                continue
            raw_index = lane.attrib.get("index")
            if raw_index is None:
                try:
                    raw_index = lane_id.rsplit("_", 1)[1]
                except (AttributeError, IndexError):
                    continue
            lane_by_road_and_index[(road_id, str(raw_index))] = lane_id

    connections = {}
    for connection in root.iterfind("connection"):
        from_road = connection.attrib.get("from")
        to_road = connection.attrib.get("to")
        if (
            not from_road
            or not to_road
            or from_road.startswith(":")
            or to_road.startswith(":")
        ):
            continue
        from_lane = lane_by_road_and_index.get(
            (from_road, connection.attrib.get("fromLane"))
        )
        to_lane = lane_by_road_and_index.get(
            (to_road, connection.attrib.get("toLane"))
        )
        if from_lane is None or to_lane is None:
            continue
        edge = (from_road, to_road)
        lane_pair = (from_lane, to_lane)
        if lane_pair not in connections.setdefault(edge, []):
            connections[edge].append(lane_pair)

    return connections


def generate_metsr_internal_lane_connection_map(map):
    """Map each SUMO internal connector lane to its external lane pair."""
    try:
        tree = ET.parse(map)
    except FileNotFoundError:
        print(f"Could not find map: {map} from {os.getcwd()}")
        return {}

    root = tree.getroot()
    driving_lane_ids = set(generate_metsr_lane_index_map(map))
    lane_by_road_and_index = {}
    for edge in root.iterfind("edge"):
        road_id = edge.attrib.get("id")
        if not road_id or road_id.startswith(":"):
            continue
        for lane in edge.findall("lane"):
            lane_id = lane.attrib.get("id")
            if lane_id not in driving_lane_ids:
                continue
            raw_index = lane.attrib.get("index")
            if raw_index is not None:
                lane_by_road_and_index[(road_id, str(raw_index))] = lane_id

    internal_connections = {}
    for connection in root.iterfind("connection"):
        from_road = connection.attrib.get("from")
        to_road = connection.attrib.get("to")
        via = connection.attrib.get("via")
        if (
            not from_road
            or not to_road
            or not via
            or from_road.startswith(":")
            or to_road.startswith(":")
        ):
            continue
        from_lane = lane_by_road_and_index.get(
            (from_road, connection.attrib.get("fromLane"))
        )
        to_lane = lane_by_road_and_index.get(
            (to_road, connection.attrib.get("toLane"))
        )
        if from_lane is None or to_lane is None:
            continue
        pair = (from_lane, to_lane)
        if pair not in internal_connections.setdefault(str(via), []):
            internal_connections[str(via)].append(pair)
    return internal_connections


def generate_metsr_internal_edge_connection_map(map):
    """Map each inbound SUMO internal edge to its external lane pairs.

    The ``via`` attribute on an external-road connection names an internal
    lane, not the connector movement owned by METS-R. Resolve that lane
    through its enclosing ``edge`` element so callers never have to infer an
    internal edge ID from a lane-name suffix.
    """
    try:
        tree = ET.parse(map)
    except FileNotFoundError:
        print(f"Could not find map: {map} from {os.getcwd()}")
        return {}

    root = tree.getroot()
    driving_lane_ids = set(generate_metsr_lane_index_map(map))
    lane_by_road_and_index = {}
    internal_edge_by_lane = {}
    for edge in root.iterfind("edge"):
        road_id = edge.attrib.get("id")
        if not road_id:
            continue
        for lane in edge.findall("lane"):
            lane_id = lane.attrib.get("id")
            if not lane_id:
                continue
            if road_id.startswith(":"):
                internal_edge_by_lane[lane_id] = road_id
                continue
            if lane_id not in driving_lane_ids:
                continue
            raw_index = lane.attrib.get("index")
            if raw_index is not None:
                lane_by_road_and_index[(road_id, str(raw_index))] = lane_id

    internal_connections = {}
    for connection in root.iterfind("connection"):
        from_road = connection.attrib.get("from")
        to_road = connection.attrib.get("to")
        via = connection.attrib.get("via")
        if (
            not from_road
            or not to_road
            or not via
            or from_road.startswith(":")
            or to_road.startswith(":")
        ):
            continue
        internal_edge = internal_edge_by_lane.get(via)
        if internal_edge is None:
            continue
        from_lane = lane_by_road_and_index.get(
            (from_road, connection.attrib.get("fromLane"))
        )
        to_lane = lane_by_road_and_index.get(
            (to_road, connection.attrib.get("toLane"))
        )
        if from_lane is None or to_lane is None:
            continue
        pair = (from_lane, to_lane)
        if pair not in internal_connections.setdefault(internal_edge, []):
            internal_connections[internal_edge].append(pair)
    return internal_connections


def generate_signal_map(map):
   
    try:
        tree = ET.parse(map)
    except FileNotFoundError:
        print(f"Could not find map: {map} from {os.getcwd()}")
        return {}
   
    root = tree.getroot()
    signal_mappings = {}
    
    for tl in root.findall(".//tlLogic"):
        tl_id = tl.attrib.get("id")
        for param in tl.findall('param'): 
            
            key = param.attrib.get("key","")

            if key.startswith("linkSignalID:"):
                metsr_key = f"{tl_id}_{key.split(':')[1]}"
            
                values = param.attrib.get("value","").split()
                if values != "":
                    signal_mappings[metsr_key] = values
    
    return signal_mappings


def test_mapping(map, test_pairs):
    mappings = generate_map(map)
    for key,value in test_pairs.items():
        if key in mappings:
            if value == mappings[key]:
                print(f"key value pair {key,value} mapped correctly")
            else:
                print(f"expected {value} returned {mappings[key]}")
                print(f"Value {mappings[key]} for key {key} was incorrect with actual value {value}")
        else:
            print(f"Failed on test case {key}, {value}")


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
    light_state_dict = {'green_time':light.get_green_time(),
                        'red_time':   light.get_red_time(),
                        'yellow_time':light.get_yellow_time(),
                        'state'      :light.get_state() }
    
    return light_state_dict

def disable_carla_autopilot(obj, tm) -> bool:
    if hasattr(obj, 'carlaActor'):
        if obj.carlaActor != None:
            obj.carlaActor.set_autopilot(False, tm.get_port())
            return True
    else:
        return False

            
if __name__ == "__main__":

    map = "Town01.net.xml"

    # key value pairs where key == origID and value == lane id
    town01_test_pairs = {"4_1": "4_2", "0_2 11_-2 8_2": "0_1", "8_-3 11_3 0_-3": "-8_0" }

    test_mapping(map, town01_test_pairs)

    map = "Town02.net.xml"

    # Randomly selected test cases from the file to check accuracy
    town02_test_pairs = {"177_-1": ":132_3_0", "276_-3":":242_2_0", "1_-2 16_-2 12_2 3_-2 15_2": "-1_1"}

    test_mapping(map, town02_test_pairs)

    map = "Town05.net.xml"

    result = generate_signal_map(map)

    print(f'Result was: {result}')
