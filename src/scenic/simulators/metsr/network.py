"""METS-R network metadata, Scenic mappings, and SUMO connector topology."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
import xml.etree.ElementTree as ET

from scenic.core.regions import PolygonalRegion


@dataclass(frozen=True)
class NetworkMetadata:
    """Copied, immutable mappings and derived indexes shared between runs."""

    lane_mapping: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    lane_indices: Mapping[str, int] | None = field(default_factory=dict)
    internal_lane_connections: Mapping[str, tuple[tuple[str, str], ...]] = field(
        default_factory=dict
    )
    internal_edge_connections: Mapping[str, tuple[tuple[str, str], ...]] = field(
        default_factory=dict
    )
    signal_mapping: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    carla_keys_by_lane: Mapping[str, frozenset[str]] = field(init=False)
    roads_by_scenic: Mapping[str, frozenset[str]] = field(init=False)
    scenic_roads_by_native: Mapping[str, tuple[str, ...]] = field(init=False)

    def __post_init__(self):
        # Copy nested inputs so callers cannot mutate metadata shared by runs.
        for name in ("lane_mapping", "signal_mapping"):
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {key: tuple(values) for key, values in getattr(self, name).items()}
                ),
            )
        for name in ("internal_lane_connections", "internal_edge_connections"):
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {
                        key: tuple(tuple(pair) for pair in values)
                        for key, values in getattr(self, name).items()
                    }
                ),
            )
        if self.lane_indices is not None:
            object.__setattr__(
                self, "lane_indices", MappingProxyType(dict(self.lane_indices))
            )
        forward, reverse = build_road_mappings(self.lane_mapping)
        object.__setattr__(
            self,
            "roads_by_scenic",
            MappingProxyType({key: frozenset(values) for key, values in forward.items()}),
        )
        object.__setattr__(
            self,
            "scenic_roads_by_native",
            MappingProxyType({key: tuple(values) for key, values in reverse.items()}),
        )
        lane_keys = {}
        for carla_key, native_lanes in self.lane_mapping.items():
            for lane in native_lanes:
                lane_keys.setdefault(str(lane), set()).add(str(carla_key))
        object.__setattr__(
            self,
            "carla_keys_by_lane",
            MappingProxyType(
                {key: frozenset(values) for key, values in lane_keys.items()}
            ),
        )

    def __reduce__(self):
        # Mapping proxies are immutable but cannot be pickled directly. Rebuild
        # from source mappings so deserialization also recreates derived indexes.
        return type(self), (
            dict(self.lane_mapping),
            None if self.lane_indices is None else dict(self.lane_indices),
            dict(self.internal_lane_connections),
            dict(self.internal_edge_connections),
            dict(self.signal_mapping),
        )


def load_network_metadata(xml_path):
    """Parse one SUMO XML file into the mappings used by METS-R integration.

    Raw SUMO lane suffixes and connection indices differ from METS-R's compact
    lane indices: shoulders are removed before the remaining lanes are sorted.
    Internal edges are resolved from their XML membership, not their ID syntax.
    """
    root = ET.parse(xml_path).getroot()
    lane_mapping = {}
    lane_indices = {}
    internal_lane_connections = {}
    internal_edge_connections = {}
    signal_mapping = {}
    lane_by_road_and_index = {}
    internal_edge_by_lane = {}

    for edge in root.iterfind("edge"):
        road_id = edge.get("id")
        lanes = edge.findall("lane")
        for lane in lanes:
            lane_id = lane.get("id")
            for param in lane.findall("param"):
                if param.get("key") == "origId":
                    for original_id in param.get("value", "").split():
                        lane_mapping.setdefault(original_id, []).append(lane_id)
            if road_id and road_id.startswith(":") and lane_id:
                internal_edge_by_lane[lane_id] = road_id

        edge_type = (edge.get("type") or "").lower()
        if (edge.get("function") or "").lower() == "internal" or not any(
            marker in edge_type for marker in ("highway", "driving")
        ):
            continue
        driving_lanes = []
        for lane in lanes:
            lane_type = lane.get("type")
            if lane_type is not None and lane_type.lower() != "driving":
                continue
            lane_id = lane.get("id")
            if not lane_id:
                continue
            try:
                suffix = int(lane_id.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            driving_lanes.append((suffix, lane_id))
        for lane_index, (_, lane_id) in enumerate(sorted(driving_lanes)):
            lane_indices[lane_id] = lane_index

        if road_id and not road_id.startswith(":"):
            for lane in lanes:
                lane_id, raw_index = lane.get("id"), lane.get("index")
                if lane_id in lane_indices and raw_index is not None:
                    lane_by_road_and_index[road_id, raw_index] = lane_id

    for connection in root.iterfind("connection"):
        source, target, via = (
            connection.get("from"),
            connection.get("to"),
            connection.get("via"),
        )
        if (
            not source
            or not target
            or not via
            or source.startswith(":")
            or target.startswith(":")
        ):
            continue
        source_lane = lane_by_road_and_index.get((source, connection.get("fromLane")))
        target_lane = lane_by_road_and_index.get((target, connection.get("toLane")))
        if source_lane is None or target_lane is None:
            continue
        pair = source_lane, target_lane
        via_pairs = internal_lane_connections.setdefault(via, [])
        if pair not in via_pairs:
            via_pairs.append(pair)
        internal_edge = internal_edge_by_lane.get(via)
        if internal_edge is not None:
            edge_pairs = internal_edge_connections.setdefault(internal_edge, [])
            if pair not in edge_pairs:
                edge_pairs.append(pair)

    for signal in root.findall(".//tlLogic"):
        for param in signal.findall("param"):
            key = param.get("key", "")
            if key.startswith("linkSignalID:"):
                signal_key = f"{signal.get('id')}_{key.split(':')[1]}"
                signal_mapping[signal_key] = param.get("value", "").split()
    return NetworkMetadata(
        lane_mapping=lane_mapping,
        lane_indices=lane_indices,
        internal_lane_connections=internal_lane_connections,
        internal_edge_connections=internal_edge_connections,
        signal_mapping=signal_mapping,
    )


def mapped_lane_query(state, road_lane):
    """Return a road ID and METS-R's filtered index for a SUMO lane ID."""
    road_lane = str(road_lane)
    if (
        state.metadata.lane_indices is not None
        and road_lane in state.metadata.lane_indices
    ):
        road, separator, _ = road_lane.rpartition("_")
        if separator:
            return road, state.metadata.lane_indices[road_lane]
    if road_lane in state.valid_roads:
        return road_lane, -1
    road, separator, lane_text = road_lane.rpartition("_")
    if not separator:
        return road_lane, -1
    if state.metadata.lane_indices is not None:
        return road, None
    # Older callers without an XML lane index retain raw lane-number lookup.
    try:
        return road, int(lane_text)
    except ValueError:
        return road, None


def mapped_driving_region(network, metadata):
    """Return mapped Scenic lane keys and their spawn region without reparsing XML."""
    lane_keys = frozenset(
        key
        for key, lanes in metadata.lane_mapping.items()
        if any(lane in metadata.lane_indices for lane in lanes)
    )
    mapped_lanes = tuple(
        lane for lane in network.lanes if f"{lane.road.id}_{lane.id}" in lane_keys
    )
    if not mapped_lanes:
        raise RuntimeError(
            "SUMO map contains no physical driving lanes mapped from the Scenic/OpenDRIVE network"
        )
    return lane_keys, PolygonalRegion.unionAll(mapped_lanes)


def build_road_mappings(lane_mapping):
    """Index road IDs in both directions without interpreting their contents."""
    roads_by_scenic = {}
    scenic_roads_by_native = {}
    for scenic_lane, native_lanes in lane_mapping.items():
        scenic_road = scenic_lane.rsplit("_", 1)[0]
        mapped_roads = roads_by_scenic.setdefault(scenic_road, set())
        for native_lane in native_lanes:
            native_road = native_lane.rsplit("_", 1)[0]
            if native_road not in mapped_roads:
                mapped_roads.add(native_road)
                scenic_roads_by_native.setdefault(native_road, []).append(scenic_road)
    return roads_by_scenic, scenic_roads_by_native


def map_scenic_to_metsr_road(state, road):
    """Return METS-R roads represented by one Scenic road."""
    road_id = str(road.id)
    mapped = state.metadata.roads_by_scenic.get(road_id)
    assert mapped is not None, f"Error identifying associated ID for {road_id}"
    return mapped


def map_scenic_to_metsr_lanes(state, lane):
    """Return native road IDs represented by one Scenic lane section."""
    lanes = state.metadata.lane_mapping.get(f"{lane.road.id}_{lane.id}")
    if lanes is None:
        return None
    return {lane_id.rsplit("_", 1)[0] for lane_id in lanes}


def generate_metsr_trajectory(state, trajectory):
    """Translate a Scenic lane sequence to the corresponding native road sequence."""
    roads = []
    for lane in trajectory:
        mapped = map_scenic_to_metsr_lanes(state, lane)
        if mapped:
            if len(mapped) > 1:
                print(f"What do you even do in this case: {mapped} ")
            roads.append(mapped.pop())
        else:
            print(
                f"Skipped non existing mapping for laneSection: {lane.road.id}_{lane.id}"
            )
    return roads


def repair_mapping(state, road_ids, bubble_road_ids, scenic_roads_by_id):
    """Include Scenic road fragments represented by the same native bubble road."""
    repaired = []
    for road_id in bubble_road_ids:
        connected = state.metadata.scenic_roads_by_native.get(road_id, ())
        if len(connected) > 1:
            repaired.extend(
                scenic_roads_by_id[scenic_id]
                for scenic_id in connected
                if scenic_id not in road_ids and scenic_id in scenic_roads_by_id
            )
    return repaired
