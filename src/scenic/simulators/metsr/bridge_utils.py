"""METS-R protocol validation and per-run co-simulation ownership state.

The integration calls METSRClient directly. Helpers validate native replies and
prepare ownership changes without depending on CARLA or its visualization stack.
"""

from dataclasses import dataclass, field
import math

from scenic.core.object_types import Object
from scenic.core.simulators import SimulationCreationError
from scenic.core.vectors import Orientation, Vector
from scenic.simulators.metsr.network_utils import NetworkMetadata, mapped_lane_query

COSIM_ADMISSION_SPAWN_MAX_TICKS = 100


@dataclass
class METSRCosimState:
    """Native ownership and pending admissions for one co-simulation run."""

    next_vehicle_id: int = 0
    private_vehicle_ids: dict[Object, int] = field(default_factory=dict)
    metadata: NetworkMetadata = field(
        default_factory=lambda: NetworkMetadata(lane_indices=None)
    )
    road_cache: dict = field(default_factory=dict)
    connector_path_cache: dict = field(default_factory=dict)
    connector_centerline_cache: dict = field(default_factory=dict)
    valid_roads: list[str] = field(default_factory=list)
    controlled_roads: dict[str, bool] = field(default_factory=dict)
    controlled_segments: set[str] = field(default_factory=set)
    internal_edge_to_connector: dict[str, str] = field(default_factory=dict)
    connector_records: dict[str, dict] = field(default_factory=dict)
    road_connectors: dict[str, set[str]] = field(default_factory=dict)
    active_bubble_roads: set[str] = field(default_factory=set)
    admitted_vehicles: dict[int, dict] = field(default_factory=dict)
    authoritative_segments: dict[Object, str | None] = field(default_factory=dict)


class METSRControlError(RuntimeError):
    """A native METS-R control response containing a failed operation."""

    def __init__(self, message, response=None, record=None):
        super().__init__(message)
        self.response = response
        self.record = record
        self.retryable = bool(isinstance(record, dict) and record.get("retryable", False))


def require_metsr_response(response, operation, *, require_all=False):
    """Check native results before applying local ownership changes."""
    if not isinstance(response, dict):
        raise METSRControlError(
            f"METS-R returned no valid response for {operation}: {response!r}",
            response=response,
        )
    if response.get("messageType") != operation:
        raise METSRControlError(
            f"Expected {operation}, received {response.get('messageType')!r}",
            response=response,
        )
    records = response.get("data")
    if not isinstance(records, (list, tuple)):
        raise METSRControlError(
            f"METS-R returned invalid data for {operation}: {records!r}",
            response=response,
        )
    for record in records:
        if isinstance(record, dict) and record.get("status") == "error":
            error_code = record.get("errorCode")
            message = record.get("message")
            if error_code and message:
                detail = f"{error_code}: {message}"
            else:
                detail = message or error_code or "unknown error"
            vehicle_id = record.get("vehicleId")
            subject = f" vehicle {vehicle_id}" if vehicle_id is not None else ""
            raise METSRControlError(
                f"METS-R rejected{subject} during {operation}: {detail}",
                response=response,
                record=record,
            )
    status = response.get("status")
    if status not in ({"ok"} if require_all else {"ok", "partial"}):
        raise METSRControlError(
            response.get("message")
            or f"METS-R rejected {operation} with status {status!r}",
            response=response,
        )
    return response


def index_response_records(response, operation, requested_keys, record_key):
    """Match one result per requested identity without depending on reply order.

    The caller owns envelope and per-record success policy. In particular,
    queue admission can retain retryable errors after identity validation.
    """
    requested = tuple(requested_keys)
    expected = set(requested)
    if len(expected) != len(requested):
        raise ValueError(f"Duplicate identities requested for {operation}")
    records = response.get("data")
    if not isinstance(records, (list, tuple)):
        raise METSRControlError(
            f"METS-R returned invalid result records for {operation}", response=response
        )
    indexed = {}
    for record in records:
        if not isinstance(record, dict):
            raise METSRControlError(
                f"METS-R returned an invalid result record for {operation}",
                response=response,
                record=record,
            )
        key = record_key(record)
        if key not in expected:
            raise METSRControlError(
                f"METS-R acknowledged unexpected identity {key!r} during {operation}",
                response=response,
                record=record,
            )
        if key in indexed:
            raise METSRControlError(
                f"METS-R acknowledged identity {key!r} more than once during {operation}",
                response=response,
                record=record,
            )
        indexed[key] = record
    missing = [key for key in requested if key not in indexed]
    if missing:
        raise METSRControlError(
            f"METS-R omitted acknowledgements for identities {missing!r} during {operation}",
            response=response,
        )
    return indexed


def require_vehicle_batch_response(response, operation, vehicle_ids):
    """Require one successful acknowledgement per requested vehicle."""
    response = require_metsr_response(response, operation, require_all=True)
    acknowledgements = index_response_records(
        response,
        operation,
        map(str, vehicle_ids),
        lambda record: (
            None if record.get("vehicleId") is None else str(record["vehicleId"])
        ),
    )
    for record in acknowledgements.values():
        if record.get("status") not in (None, "ok"):
            raise METSRControlError(
                f"METS-R returned non-success vehicle result during {operation}",
                response=response,
                record=record,
            )
    return response, acknowledgements


def metsr_vehicle_road(record):
    """Return the logical physical road represented by a native vehicle record."""
    if not isinstance(record, dict):
        return None
    on_connector = record.get("onConnector") or (
        str(record.get("segmentType", "")).lower() == "connector"
    )
    if on_connector:
        return record.get("targetRoadId") or record.get("segmentId")
    return record.get("segmentId")


def rebuild_controlled_physical_segments(state):
    """Rebuild the physical-segment set without inserting connector IDs."""
    referenced_connectors = {
        connector_id
        for connector_ids in state.road_connectors.values()
        for connector_id in connector_ids
    }
    state.connector_records = {
        connector_id: record
        for connector_id, record in state.connector_records.items()
        if connector_id in referenced_connectors
    }
    edge_mapping = {}
    for connector_id, record in state.connector_records.items():
        for internal_edge_id in record.get("internalEdgeIds", ()):
            internal_edge_id = str(internal_edge_id)
            previous = edge_mapping.get(internal_edge_id)
            if previous is not None and previous != connector_id:
                raise METSRControlError(
                    f"Internal edge {internal_edge_id!r} belongs to both "
                    f"{previous!r} and {connector_id!r}"
                )
            edge_mapping[internal_edge_id] = connector_id
    state.internal_edge_to_connector = edge_mapping
    state.controlled_segments = set(map(str, state.controlled_roads))
    state.controlled_segments.update(edge_mapping)


def expected_incident_internal_edges(state, road_ids):
    """Return XML-proven connector edges touching each requested road."""
    requested = {str(road_id) for road_id in road_ids}
    expected = {road_id: set() for road_id in requested}
    edge_connections = state.metadata.internal_edge_connections

    for internal_edge_id, pairs in edge_connections.items():
        for source_lane, target_lane in pairs:
            source_road_id, _ = mapped_lane_query(state, source_lane)
            target_road_id, _ = mapped_lane_query(state, target_lane)
            for incident_road_id in {
                str(source_road_id),
                str(target_road_id),
            }:
                if incident_road_id in requested:
                    expected[incident_road_id].add(str(internal_edge_id))
    return expected


def connector_topology(record):
    return (
        record.get("connectorId"),
        record.get("sourceRoadId"),
        record.get("targetRoadId"),
        tuple(record.get("internalEdgeIds", ())),
    )


def normalize_incident_connector(connector, road_id, response, operation):
    """Validate one opaque server connector record touching road_id."""
    if not isinstance(connector, dict):
        raise METSRControlError(
            f"{operation} returned an invalid connector for road {road_id}",
            response=response,
            record=connector,
        )
    connector_id = connector.get("connectorId")
    source_road_id = connector.get("sourceRoadId")
    target_road_id = connector.get("targetRoadId")
    internal_edge_ids = connector.get("internalEdgeIds")
    if connector_id is None or not str(connector_id):
        raise METSRControlError(
            f"{operation} omitted connectorId for road {road_id}",
            response=response,
            record=connector,
        )
    if source_road_id is None or target_road_id is None:
        raise METSRControlError(
            f"{operation} omitted connector endpoints for {connector_id!r}",
            response=response,
            record=connector,
        )
    connector_id = str(connector_id)
    source_road_id = str(source_road_id)
    target_road_id = str(target_road_id)
    if str(road_id) not in {source_road_id, target_road_id}:
        raise METSRControlError(
            f"{operation} returned connector {connector_id!r}, which does "
            f"not touch road {road_id!r}",
            response=response,
            record=connector,
        )
    if not isinstance(internal_edge_ids, (list, tuple)):
        raise METSRControlError(
            f"{operation} omitted internalEdgeIds for {connector_id!r}",
            response=response,
            record=connector,
        )
    normalized = dict(connector)
    normalized["connectorId"] = connector_id
    normalized["sourceRoadId"] = source_road_id
    normalized["targetRoadId"] = target_road_id
    normalized["internalEdgeIds"] = list(
        dict.fromkeys(
            str(edge_id)
            for edge_id in internal_edge_ids
            if edge_id is not None and str(edge_id)
        )
    )
    return normalized


def apply_road_takeover(state, response, requested):
    """Validate the complete road takeover before changing local ownership."""
    require_metsr_response(response, "setCoSimRoad", require_all=True)
    requested = tuple(sorted({str(road_id) for road_id in requested}))
    expected_edges = expected_incident_internal_edges(state, requested)
    records_by_road = {}
    for record in response.get("data", ()):
        if not isinstance(record, dict) or record.get("status") != "ok":
            continue
        road_id = record.get("roadId")
        if road_id is None:
            raise METSRControlError(
                "setCoSimRoad returned a successful record without roadId",
                response=response,
                record=record,
            )
        road_id = str(road_id)
        if road_id not in requested or road_id in records_by_road:
            raise METSRControlError(
                f"setCoSimRoad returned an unexpected or duplicate road {road_id!r}",
                response=response,
                record=record,
            )
        records_by_road[road_id] = record
    missing_roads = set(requested) - set(records_by_road)
    if missing_roads:
        raise METSRControlError(
            "setCoSimRoad omitted successful records for " f"{sorted(missing_roads)}",
            response=response,
        )

    road_connectors = {road_id: set() for road_id in requested}
    connector_records = {}
    internal_edge_to_connector = {}
    returned_edges = {road_id: set() for road_id in requested}
    for road_id, road_record in records_by_road.items():
        connectors = road_record.get("connectors")
        if connectors is None:
            connectors = ()
        if not isinstance(connectors, (list, tuple)):
            raise METSRControlError(
                f"setCoSimRoad returned invalid connectors for road {road_id}",
                response=response,
                record=road_record,
            )
        for connector in connectors:
            normalized = normalize_incident_connector(
                connector, road_id, response, "setCoSimRoad"
            )
            connector_id = normalized["connectorId"]
            previous = connector_records.get(connector_id)
            if previous is not None and connector_topology(
                previous
            ) != connector_topology(normalized):
                raise METSRControlError(
                    f"setCoSimRoad returned conflicting records for {connector_id!r}",
                    response=response,
                    record=connector,
                )
            connector_records[connector_id] = normalized
            road_connectors[road_id].add(connector_id)
            for internal_edge_id in normalized["internalEdgeIds"]:
                previous_owner = internal_edge_to_connector.get(internal_edge_id)
                if previous_owner is not None and previous_owner != connector_id:
                    raise METSRControlError(
                        f"Internal edge {internal_edge_id!r} belongs to both "
                        f"{previous_owner!r} and {connector_id!r}",
                        response=response,
                        record=connector,
                    )
                internal_edge_to_connector[internal_edge_id] = connector_id
                returned_edges[road_id].add(internal_edge_id)

        announced_ids = road_record.get("connectorIds")
        if (
            not isinstance(announced_ids, (list, tuple))
            or {str(connector_id) for connector_id in announced_ids}
            != road_connectors[road_id]
        ):
            raise METSRControlError(
                f"setCoSimRoad connectorIds/connectors disagree for road {road_id}",
                response=response,
                record=road_record,
            )

    for road_id, expected in expected_edges.items():
        missing = expected - returned_edges[road_id]
        if missing:
            raise METSRControlError(
                f"setCoSimRoad omitted expected incident internal edges for "
                f"road {road_id}: {sorted(missing)}",
                response=response,
                record=records_by_road[road_id],
            )
    for connector_id, normalized in connector_records.items():
        previous = state.connector_records.get(connector_id)
        if previous is not None and connector_topology(previous) != connector_topology(
            normalized
        ):
            raise METSRControlError(
                f"Connector {connector_id!r} changed topology during takeover",
                response=response,
                record=normalized,
            )
    for edge_id, connector_id in internal_edge_to_connector.items():
        previous_owner = state.internal_edge_to_connector.get(edge_id)
        if previous_owner is not None and previous_owner != connector_id:
            raise METSRControlError(
                f"Internal edge {edge_id!r} changed connector ownership from "
                f"{previous_owner!r} to {connector_id!r}",
                response=response,
            )
    for road_id in requested:
        state.controlled_roads[road_id] = True
        state.road_connectors[road_id] = set(road_connectors.get(road_id, ()))
    state.connector_records.update(connector_records)
    rebuild_controlled_physical_segments(state)


def collect_metsr_vehicle_data(client, object_to_vehicle_ids):
    """Query native poses for the supplied object-to-private-ID mapping."""
    if not object_to_vehicle_ids:
        return {}
    _, records = require_vehicle_batch_response(
        client.query_vehicle(list(object_to_vehicle_ids.values()), True, True),
        "vehicle",
        object_to_vehicle_ids.values(),
    )
    return {
        obj: records[str(vehicle_id)] for obj, vehicle_id in object_to_vehicle_ids.items()
    }


def is_controlled_internal_segment(state, segment_id):
    """Return whether METS-R assigned this physical edge to a connector."""
    if segment_id is None:
        return False
    segment_id = str(segment_id)
    return (
        segment_id in state.controlled_segments
        and segment_id in state.internal_edge_to_connector
    )


def vehicle_record_in_cosim_region(state, record):
    """Test physical COSIM ownership without reducing connectors to roads."""
    if not isinstance(record, dict):
        return False

    connector_ids = {
        str(value)
        for value in (record.get("connectorId"), record.get("segmentId"))
        if value is not None
    }
    if connector_ids & set(state.connector_records):
        return True

    physical_segments = {
        str(value)
        for value in (
            record.get("segmentId"),
            record.get("observedSegmentId"),
            record.get("physicalSegmentId"),
        )
        if value is not None
    }
    physical_segments.update(
        str(edge_id)
        for edge_id in (record.get("internalEdgeIds") or ())
        if edge_id is not None
    )
    if physical_segments & state.controlled_segments:
        return True

    logical_road = metsr_vehicle_road(record)
    return logical_road is not None and str(logical_road) in state.controlled_roads


def is_controlled_connector_id(state, segment_id):
    """Whether this opaque connector is recorded and referenced by an owned road."""
    if segment_id is None:
        return False
    referenced = {
        str(connector_id)
        for connector_ids in state.road_connectors.values()
        for connector_id in connector_ids
    }
    return str(segment_id) in referenced & set(map(str, state.connector_records))


def initialize_carla_authoritative_segment(state, obj, record):
    """Seed continuity from the current admission record without parsing IDs."""
    if obj in state.authoritative_segments:
        return state.authoritative_segments[obj]
    if not isinstance(record, dict) or record.get("segmentId") is None:
        return None
    segment = str(record["segmentId"])
    if segment not in state.controlled_roads:
        if is_controlled_internal_segment(state, segment):
            segment = state.internal_edge_to_connector[segment]
        elif not is_controlled_connector_id(state, segment):
            return None
    state.authoritative_segments[obj] = segment
    return segment


def releasable_old_roads(state, old_roads, occupied_objects):
    """Keep native segments controlled until their CARLA occupants leave."""
    occupied = set()
    segments = state.authoritative_segments
    for obj in occupied_objects:
        segment = segments.get(obj)
        if segment is None:
            # No accepted observation yet: releasing a road could resume
            # native motion while this actor is still controlled by CARLA.
            return []
        occupied.add(str(segment))
    held = set(occupied)
    for road, connectors in state.road_connectors.items():
        if occupied.intersection(map(str, connectors)):
            held.add(str(road))
    for admission in state.admitted_vehicles.values():
        held.add(str(admission["roadId"]))
    return [str(road) for road in old_roads if str(road) not in held]


def check_queue_admission_spawn_deadlines(state, current_tick, spawned_vehicle_ids):
    """Fail if an admitted vehicle misses its external actor spawn deadline."""
    for vehicle_id, admission in tuple(state.admitted_vehicles.items()):
        if vehicle_id in spawned_vehicle_ids:
            state.admitted_vehicles.pop(vehicle_id, None)
            continue
        admitted_at = admission["admittedAt"]
        if current_tick - admitted_at >= COSIM_ADMISSION_SPAWN_MAX_TICKS:
            raise SimulationCreationError(
                f"Vehicle {vehicle_id} was admitted from the METS-R "
                f"departure queue onto COSIM road {admission['roadId']} "
                f"at step {admitted_at}, but CARLA did not spawn it within "
                f"{COSIM_ADMISSION_SPAWN_MAX_TICKS} Scenic steps"
            )


def select_queue_admissions(state, queue_response):
    """Select ready managed private vehicles on active controlled roads."""
    queue_records = queue_response["data"]
    managed_private_ids = set(state.private_vehicle_ids.values())
    seen_roads = set()
    selected_vehicle_ids = set()
    admission_requests = []
    for road_queue in queue_records:
        if not isinstance(road_queue, dict) or road_queue.get("status") != "ok":
            raise METSRControlError(
                "coSimEnteringVehicleQueue returned an invalid road record",
                response=queue_response,
                record=road_queue,
            )
        road_id = road_queue.get("segmentId")
        if road_id is None:
            raise METSRControlError(
                "coSimEnteringVehicleQueue omitted segmentId",
                response=queue_response,
                record=road_queue,
            )
        road_id = str(road_id)
        if road_id in seen_roads:
            raise METSRControlError(
                f"coSimEnteringVehicleQueue duplicated road {road_id!r}",
                response=queue_response,
                record=road_queue,
            )
        seen_roads.add(road_id)
        queue = road_queue.get("queue")
        if not isinstance(queue, (list, tuple)):
            raise METSRControlError(
                f"coSimEnteringVehicleQueue returned an invalid queue for {road_id}",
                response=queue_response,
                record=road_queue,
            )
        if (
            road_id not in state.controlled_roads
            or road_id not in state.active_bubble_roads
        ):
            continue

        # ID-based admission lets Scenic skip unsupported public entries
        # without requiring them to leave the native queue first.
        for entry in queue:
            if not isinstance(entry, dict) or entry.get("ready") is not True:
                continue
            vehicle_id = entry.get("vehicleId")
            if (
                entry.get("isPrivate") is not True
                or vehicle_id not in managed_private_ids
                or vehicle_id in state.admitted_vehicles
                or vehicle_id in selected_vehicle_ids
            ):
                continue

            request = {
                "vehicleId": vehicle_id,
                "internalVehicleId": entry.get("internalVehicleId"),
                "isPrivate": True,
                "roadId": road_id,
            }
            request = {key: value for key, value in request.items() if value is not None}
            selected_vehicle_ids.add(vehicle_id)
            admission_requests.append(request)

    return admission_requests


def validate_queue_admissions(admission_response, admission_requests):
    """Return acknowledged admissions, leaving retryable rejections queued."""
    accepted = {}
    records_by_key = index_response_records(
        admission_response,
        "enterRoadFromQueue",
        (
            (str(request["vehicleId"]), str(request["roadId"]))
            for request in admission_requests
        ),
        lambda record: (str(record.get("vehicleId")), str(record.get("roadId"))),
    )

    for request in admission_requests:
        vehicle_id = request["vehicleId"]
        road_id = str(request["roadId"])
        admitted = records_by_key[str(vehicle_id), road_id]
        if admitted.get("status") == "error":
            error = METSRControlError(
                admitted.get("message")
                or admitted.get("errorCode")
                or f"METS-R rejected vehicle {vehicle_id} during enterRoadFromQueue",
                response=admission_response,
                record=admitted,
            )
            if error.retryable:
                continue
            raise error
        if (
            admitted.get("status") != "ok"
            or admitted.get("isPrivate") is not True
            or admitted.get("state") is not None
        ):
            raise METSRControlError(
                "enterRoadFromQueue did not confirm the requested "
                f"vehicle {vehicle_id} on road {road_id}",
                response=admission_response,
                record=admitted,
            )
        accepted[vehicle_id] = road_id
    return accepted


def parse_connector_collection(connectors, connector_ids, road_id, response, partition):
    """Validate one release partition of incident connector records and IDs."""
    records = {}
    for connector in connectors:
        normalized = normalize_incident_connector(
            connector, road_id, response, "releaseCoSimRoad"
        )
        connector_id = normalized["connectorId"]
        if connector_id in records:
            raise METSRControlError(
                f"releaseCoSimRoad duplicated connector {connector_id!r}",
                response=response,
                record=connector,
            )
        records[connector_id] = normalized
    if set(records) != {str(connector_id) for connector_id in connector_ids}:
        raise METSRControlError(
            f"releaseCoSimRoad {partition} connector IDs and records disagree for road {road_id}",
            response=response,
        )
    return records


def apply_road_release(state, key, response, *, allow_defer=False):
    """Validate and apply a road release; return False if native placement is blocked."""
    if allow_defer and isinstance(response, dict):
        records = response.get("data")
        if (
            response.get("messageType") == "releaseCoSimRoad"
            and response.get("status") in ("ok", "partial")
            and isinstance(records, (list, tuple))
            and len(records) == 1
            and isinstance(records[0], dict)
            and str(records[0].get("roadId")) == key
            and records[0].get("status") == "error"
            and records[0].get("errorCode") == "RELEASE_BLOCKED"
            and records[0].get("retryable") is True
        ):
            # Keep the road owned and reconsider it on the next step.
            return False
    require_metsr_response(response, "releaseCoSimRoad", require_all=True)
    records = [
        record
        for record in response.get("data", ())
        if isinstance(record, dict)
        and record.get("status") == "ok"
        and str(record.get("roadId")) == key
    ]
    if len(records) != 1:
        raise METSRControlError(
            "releaseCoSimRoad did not return exactly one successful " f"record for {key}",
            response=response,
        )
    record = records[0]
    released_connector_data = record.get("releasedConnectors")
    released_id_data = record.get("releasedConnectorIds")
    remaining_connector_data = record.get("connectors")
    remaining_id_data = record.get("connectorIds")
    if not all(
        isinstance(value, (list, tuple))
        for value in (
            released_connector_data,
            released_id_data,
            remaining_connector_data,
            remaining_id_data,
        )
    ):
        raise METSRControlError(
            f"releaseCoSimRoad omitted connector metadata for road {key}",
            response=response,
            record=record,
        )

    released_records = parse_connector_collection(
        released_connector_data, released_id_data, key, response, "released"
    )
    remaining_records = parse_connector_collection(
        remaining_connector_data, remaining_id_data, key, response, "remaining"
    )
    released_ids, remaining_ids = set(released_records), set(remaining_records)

    owned_ids = set(state.road_connectors.get(key, ()))
    if released_ids & remaining_ids or released_ids | remaining_ids != owned_ids:
        raise METSRControlError(
            f"releaseCoSimRoad connector partition for {key} does not "
            f"match Scenic ownership: owned={sorted(owned_ids)}, "
            f"released={sorted(released_ids)}, "
            f"remaining={sorted(remaining_ids)}",
            response=response,
            record=record,
        )

    active_after_release = set(state.controlled_roads) - {key}
    for connector_id, connector in {
        **released_records,
        **remaining_records,
    }.items():
        previous = state.connector_records.get(connector_id)
        if previous is None or connector_topology(previous) != connector_topology(
            connector
        ):
            raise METSRControlError(
                f"releaseCoSimRoad changed unknown connector topology "
                f"for {connector_id!r}",
                response=response,
                record=connector,
            )

        endpoints = {
            connector["sourceRoadId"],
            connector["targetRoadId"],
        }
        other_references = {
            road_id
            for road_id, connector_ids in state.road_connectors.items()
            if road_id != key and connector_id in connector_ids
        }
        should_remain = bool(endpoints & active_after_release)
        if connector_id in remaining_ids:
            if not should_remain or not other_references:
                raise METSRControlError(
                    f"releaseCoSimRoad retained connector {connector_id!r} "
                    "without another controlled endpoint",
                    response=response,
                    record=connector,
                )
        elif should_remain or other_references:
            raise METSRControlError(
                f"releaseCoSimRoad released connector {connector_id!r} "
                "which is still referenced by another controlled endpoint",
                response=response,
                record=connector,
            )

    del state.controlled_roads[key]
    state.road_connectors.pop(key, None)
    state.connector_records.update(remaining_records)
    for connector_id in released_ids:
        state.connector_records.pop(connector_id, None)
    rebuild_controlled_physical_segments(state)
    return True


def metsr_properties(raw_data, parent_orientation):
    """Convert a METS-R record to Scenic dynamic properties."""
    position = Vector(
        raw_data["x"],
        raw_data["y"],
        raw_data["z"] if raw_data["z"] is not None else 0,
    )
    speed = raw_data["speed"]
    bearing = math.radians(-raw_data["bearing"])
    globalOrientation = Orientation.fromEuler(bearing, 0, 0)
    yaw, pitch, roll = parent_orientation.localAnglesFor(globalOrientation)
    # Velocity is global; yaw above is relative to parentOrientation.
    # Using local yaw here launches admitted CARLA actors sideways.
    velocity = Vector(0, speed, 0).rotatedBy(bearing)
    angularSpeed = 0
    angularVelocity = Vector(0, 0, 0)

    values = dict(
        position=position,
        velocity=velocity,
        speed=speed,
        angularSpeed=angularSpeed,
        angularVelocity=angularVelocity,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        elevation=position.z,
    )
    return values


def validate_pose_update_acknowledgements(response, updates, targets):
    """Validate the entire pose batch before any actor ownership changes."""
    _, acknowledgements = require_vehicle_batch_response(
        response,
        "teleportCoSimVeh",
        [update["vehicle_id"] for update in updates],
    )
    for update, target in zip(updates, targets):
        vehicle_id = update["vehicle_id"]
        record = acknowledgements[str(vehicle_id)]
        if target:
            segment, lane, path = target
            valid = (
                record.get("controlMode") == "native"
                and record.get("releasedFromCoSim") is True
                and str(record.get("segmentId")) == segment
                and (lane is None or record.get("laneIndex") == lane)
                and (path is None or record.get("connectorPathId") == path)
            )
        else:
            valid = (
                record.get("controlMode") == "cosim"
                and record.get("releasedFromCoSim") is False
            )
        if not valid:
            raise METSRControlError(
                f"METS-R did not confirm {'native handback' if target else 'CARLA pose update'} "
                f"for vehicle {vehicle_id}",
                response=response,
                record=record,
            )
    return acknowledgements


def validate_completion_response(retirement, vehicle_id):
    """Require one native completion acknowledgement for the retiring actor."""
    require_metsr_response(retirement, "reachDest", require_all=True)
    records = retirement["data"]
    # Scalar reachDest success records may omit vehicleId.
    if (
        len(records) != 1
        or not isinstance(records[0], dict)
        or records[0].get("status") != "ok"
        or (
            records[0].get("vehicleId") is not None
            and str(records[0]["vehicleId"]) != str(vehicle_id)
        )
    ):
        raise METSRControlError(
            f"reachDest did not confirm completion of vehicle {vehicle_id}",
            response=retirement,
        )


def native_boundary_states(client, state, controlled_private_ids):
    """Query native exit occupancy, excluding vehicles already controlled here."""
    response = require_metsr_response(
        client.query_boundary_vehicle(),
        "boundaryVehicle",
        require_all=True,
    )
    controlled = {(True, str(vehicle_id)) for vehicle_id in controlled_private_ids}
    selected = {}
    for record in response["data"]:
        if (
            not isinstance(record, dict)
            or record.get("vehicleId") is None
            or not isinstance(record.get("isPrivate"), bool)
        ):
            raise METSRControlError(
                "boundaryVehicle returned an invalid identity", record=record
            )
        key = (record["isPrivate"], str(record["vehicleId"]))
        if key in controlled or vehicle_record_in_cosim_region(state, record):
            continue
        selected[key] = record
    if not selected:
        return {}

    response = require_metsr_response(
        client.query_vehicle(
            [record["vehicleId"] for record in selected.values()],
            private_veh=[key[0] for key in selected],
            transform_coords=True,
        ),
        "vehicle",
        require_all=True,
    )
    records = response["data"]
    if len(records) != len(selected):
        raise METSRControlError(
            "METS-R omitted boundary vehicle poses", response=response
        )
    states = {}
    # Vehicle queries preserve request order; public and private IDs can collide.
    for key, vehicle_state in zip(selected, records):
        if (
            not isinstance(vehicle_state, dict)
            or str(vehicle_state.get("vehicleId")) != key[1]
            or vehicle_state.get("vehicleClass") not in (0, 1, 2, 3)
            or (vehicle_state["vehicleClass"] in (0, 3)) != key[0]
        ):
            raise METSRControlError(
                f"METS-R returned the wrong boundary vehicle pose for {key}",
                response=response,
                record=vehicle_state,
            )
        states[key] = vehicle_state
    return states
