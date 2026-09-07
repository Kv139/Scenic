import json
import threading
from types import SimpleNamespace

import pytest

import scenic.simulators.metsr.client as client_module
from scenic.simulators.metsr.client import METSRClient


def make_client(response, sent_messages):
    client = object.__new__(METSRClient)

    def send_receive_msg(message, ignore_heartbeats, **kwargs):
        sent_messages.append((message, ignore_heartbeats))
        return response

    client.send_receive_msg = send_receive_msg
    return client


def ok_response(message_type, data=None):
    return {
        "messageType": message_type,
        "status": "ok",
        "data": list(data or ()),
    }


class CloseTrackingWebSocket:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def readiness_client(response):
    client = object.__new__(METSRClient)
    client.timeout = 2.0
    client.uri = "ws://test"
    client.state = "connected"
    client.config = None
    client._client_config_values = {}
    client.ws = CloseTrackingWebSocket()
    client.receive_msg = lambda **kwargs: response
    return client


def test_client_exports_no_legacy_translation_layer():
    assert not hasattr(client_module, "_to_native_request")
    assert not hasattr(client_module, "_from_native_response")
    assert not hasattr(client_module, "METSRControlError")


@pytest.mark.parametrize("retired", ["TYPE", "DATA", "CODE"])
def test_wire_validator_rejects_retired_v1_fields(retired):
    with pytest.raises(ValueError, match="V1 fields are no longer supported"):
        METSRClient._validate_wire_message(
            {"messageType": "vehicle", retired: "retired"}
        )


def test_wire_validator_requires_message_type():
    with pytest.raises(ValueError, match="require a messageType"):
        METSRClient._validate_wire_message({"data": []})


def test_receive_rejects_retired_v1_response_fields():
    client = object.__new__(METSRClient)
    client.ws = SimpleNamespace(
        recv=lambda timeout: json.dumps(
            {"TYPE": "ANS_vehicle", "CODE": "OK", "DATA": []}
        )
    )
    client.timeout = 0.1
    client.uri = "ws://test"
    client.state = "connected"
    client.verbose = False
    client.current_tick = None
    client.sim_folder = None
    client._cached_fatal_log_error = None
    client._last_fatal_log_check = 0.0

    with pytest.raises(RuntimeError, match="retired V1 fields"):
        client.receive_msg(ignore_heartbeats=True, waiting_forever=False)


def test_readiness_handshake_requires_explicit_ready_response():
    response = {"messageType": "ready", "status": "ok", "tick": 0}
    client = readiness_client(response)

    assert client._await_simulator_ready() is response
    assert client.state == "connected"
    assert client.ws.closed is False


@pytest.mark.parametrize(
    "response,error_type,message",
    [
        (None, TimeoutError, "did not report ready"),
        (
            {"messageType": "step", "status": "ok", "tick": 0},
            RuntimeError,
            "Expected METS-R SIM ready response",
        ),
    ],
)
def test_readiness_handshake_fails_closed(response, error_type, message):
    client = readiness_client(response)
    websocket = client.ws

    with pytest.raises(error_type, match=message):
        client._await_simulator_ready()

    assert client.state == "failed"
    assert websocket.closed is True
    assert client.ws is None


@pytest.mark.parametrize("include_tick_mismatch", [False, True])
def test_active_step_request_is_not_resent_after_tick_progress(include_tick_mismatch):
    client = object.__new__(METSRClient)
    client.current_tick = 0
    client.timeout = 1.0
    client.verbose = False
    client.lock = threading.Lock()
    client._record_server_tick_time = lambda *args: None
    sent = []
    responses = [None]
    if include_tick_mismatch:
        responses.append(
            {
                "messageType": "step",
                "status": "error",
                "errorCode": "TICK_MISMATCH",
                "tick": 1,
            }
        )
    responses.append({"messageType": "step", "status": "ok", "tick": 3})

    client.send_msg = lambda message: sent.append(dict(message))
    client.receive_msg = lambda **kwargs: responses.pop(0)

    def query_tick_locked():
        if client.current_tick == 0:
            client.current_tick = 1
        return client.current_tick

    client._query_tick_locked = query_tick_locked

    client.tick(
        step_num=3,
        wait_forever=True,
        retry_interval=0,
        max_wait_seconds=1,
        poll_timeout=0.1,
    )

    assert client.current_tick == 3
    assert sent == [{"messageType": "step", "tick": 0, "tickCount": 3}]


def test_teleport_cosim_vehicle_builds_native_batch():
    sent = []
    client = make_client(ok_response("teleportCoSimVeh"), sent)

    client.teleport_cosim_vehicle(
        [11, 20],
        [1.0, 2.0],
        [3.0, 4.0],
        z=[5.0, 6.0],
        bearing=[90.0, 180.0],
        speed=[7.0, 8.0],
        private_veh=True,
        transform_coords=True,
        segment_id=["-50", "-51"],
        lane_index=[2, 3],
    )

    assert sent == [
        (
            {
                "messageType": "teleportCoSimVeh",
                "data": [
                    {
                        "vehicleId": 11,
                        "x": 1.0,
                        "y": 3.0,
                        "z": 5.0,
                        "bearing": 90.0,
                        "speed": 7.0,
                        "isPrivate": True,
                        "transformCoordinates": True,
                        "segmentId": "-50",
                        "laneIndex": 2,
                    },
                    {
                        "vehicleId": 20,
                        "x": 2.0,
                        "y": 4.0,
                        "z": 6.0,
                        "bearing": 180.0,
                        "speed": 8.0,
                        "isPrivate": True,
                        "transformCoordinates": True,
                        "segmentId": "-51",
                        "laneIndex": 3,
                    },
                ],
            },
            True,
        )
    ]


def test_teleport_cosim_vehicle_builds_connector_path_selector():
    sent = []
    client = make_client(ok_response("teleportCoSimVeh"), sent)

    client.teleport_cosim_vehicle(
        11,
        1.0,
        3.0,
        bearing=90.0,
        segment_id="cont/58/-14",
        connector_path_id=4,
    )

    record = sent[0][0]["data"][0]
    assert record["segmentId"] == "cont/58/-14"
    assert record["connectorPathId"] == 4
    assert "laneIndex" not in record

def test_teleport_cosim_vehicle_returns_error_envelope_for_adapter():
    response = {
        "messageType": "teleportCoSimVeh",
        "status": "error",
        "data": [
            {
                "vehicleId": 11,
                "status": "error",
                "errorCode": "INVALID_CONNECTOR_PATH",
                "message": "connectorPathId is outside the connector",
            }
        ],
    }
    sent = []
    client = make_client(response, sent)

    result = client.teleport_cosim_vehicle(
        11,
        1.0,
        3.0,
        bearing=90.0,
        segment_id="cont/58/-14",
        connector_path_id=9,
    )

    assert result is response
    assert len(sent) == 1



@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {"segment_id": "A", "road_id": "A"},
            "aliases; use only one",
        ),
        (
            {"lane_index": 2},
            "require segment_id",
        ),
        (
            {
                "segment_id": "cont/A/B",
                "lane_index": 2,
                "connector_path_id": 1,
            },
            "mutually exclusive",
        ),
    ],
)
def test_teleport_cosim_vehicle_rejects_ambiguous_selectors(kwargs, match):
    sent = []
    client = make_client(ok_response("teleportCoSimVeh"), sent)

    with pytest.raises(ValueError, match=match):
        client.teleport_cosim_vehicle(11, 1.0, 3.0, bearing=90.0, **kwargs)
    assert sent == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("x", [1.0]),
        ("bearing", [0.0, 1.0, 2.0]),
        ("segment_id", ["-50"]),
        ("lane_index", [3]),
    ],
)
def test_teleport_cosim_vehicle_rejects_truncated_batches(field, value):
    sent = []
    client = make_client(ok_response("teleportCoSimVeh"), sent)
    arguments = {
        "vehID": [11, 20],
        "x": [1.0, 2.0],
        "y": [3.0, 4.0],
        "bearing": [90.0, 180.0],
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        client.teleport_cosim_vehicle(**arguments)
    assert sent == []


def test_enter_next_road_is_explicitly_removed():
    client = make_client(ok_response("teleportCoSimVeh"), [])
    with pytest.raises(NotImplementedError, match="not supported"):
        client.enter_next_road(11, "-50", private_veh=True, laneID=2)


def test_initialize_cosim_vehicle_builds_native_request():
    sent = []
    client = make_client(ok_response("initializeCoSimVeh"), sent)

    client.initialize_cosim_vehicle(
        7,
        10.0,
        20.0,
        45.0,
        destination_road_id="-52",
        z=0.5,
        private_veh=True,
        transform_coords=True,
        length=4.8,
        segment_id="-51",
    )

    request = sent[0][0]
    assert request["messageType"] == "initializeCoSimVeh"
    assert request["data"] == [
        {
            "vehicleId": 7,
            "isPrivate": True,
            "x": 10.0,
            "y": 20.0,
            "z": 0.5,
            "bearing": 45.0,
            "speed": 0,
            "transformCoordinates": True,
            "destinationRoadId": "-52",
            "vehicleLength": 4.8,
            "segmentId": "-51",
        }
    ]


def test_initialize_cosim_vehicle_builds_connector_path_selector():
    sent = []
    client = make_client(ok_response("initializeCoSimVeh"), sent)

    client.initialize_cosim_vehicle(
        7,
        10.0,
        20.0,
        45.0,
        destination_road_id="-52",
        segment_id="cont/6/5",
        connector_path_id=4,
    )

    record = sent[0][0]["data"][0]
    assert record["segmentId"] == "cont/6/5"
    assert record["connectorPathId"] == 4
    assert "laneIndex" not in record


def test_initialize_cosim_vehicle_connector_path_requires_segment():
    sent = []
    client = make_client(ok_response("initializeCoSimVeh"), sent)

    with pytest.raises(ValueError, match="connector_path_id requires segment_id"):
        client.initialize_cosim_vehicle(
            7,
            10.0,
            20.0,
            45.0,
            destination_road_id="-52",
            connector_path_id=4,
        )

    assert sent == []


def test_digital_twin_coordinate_mode_uses_no_removed_road_id():
    sent = []
    client = make_client(ok_response("teleportDigitalTwinVeh"), sent)

    client.teleport_digital_twin_vehicle(
        7,
        x=10.0,
        y=20.0,
        z=0.5,
        private_veh=True,
        transform_coords=True,
    )

    assert sent[0][0]["data"] == [
        {
            "vehicleId": 7,
            "isPrivate": True,
            "positionType": "coordinate",
            "x": 10.0,
            "y": 20.0,
            "z": 0.5,
            "transformCoordinates": True,
        }
    ]


def test_digital_twin_segment_mode_uses_exact_native_selectors():
    sent = []
    client = make_client(ok_response("teleportDigitalTwinVeh"), sent)

    client.teleport_digital_twin_vehicle(
        7,
        segment_id="-37",
        lane_index=3,
        distance_to_segment_end=12.5,
        private_veh=True,
    )

    assert sent[0][0]["data"] == [
        {
            "vehicleId": 7,
            "isPrivate": True,
            "positionType": "segment",
            "segmentId": "-37",
            "distanceToSegmentEnd": 12.5,
            "laneIndex": 3,
        }
    ]


def test_digital_twin_rejects_mixed_position_modes():
    client = make_client(ok_response("teleportDigitalTwinVeh"), [])

    with pytest.raises(ValueError, match="cannot mix"):
        client.teleport_digital_twin_vehicle(
            7,
            segment_id="-37",
            distance_to_segment_end=12.5,
            x=10.0,
            y=20.0,
        )


def test_route_and_cosim_queries_use_native_envelopes():
    sent = []
    route_client = make_client(
        ok_response("routesBwRoads", [{"roadIds": ["A", "B"]}]), sent
    )
    assert route_client.query_route_between_roads("A", "B")["data"][0][
        "roadIds"
    ] == ["A", "B"]
    assert sent[0][0] == {
        "messageType": "routesBwRoads",
        "data": [{"originRoadId": "A", "destinationRoadId": "B"}],
    }

    sent.clear()
    cosim_client = make_client(ok_response("coSimVehicle"), sent)
    cosim_client.query_cosim_vehicle()
    assert sent[0][0] == {"messageType": "coSimVehicle"}


def test_vehicle_route_query_uses_latest_native_schema():
    sent = []
    route_record = {
        "vehicleId": 7,
        "isPrivate": True,
        "routeScope": "remainingAssignedRoute",
        "roadIds": ["A", "B"],
        "connectorIds": ["cont/A/B"],
        "segmentIds": ["A", "cont/A/B", "B"],
        "travelTime": 12.5,
        "travelTimeP90": 17.0,
        "travelTimeConfidence": 0.8,
        "status": "ok",
    }
    client = make_client(ok_response("vehicleRoute", [route_record]), sent)

    response = client.query_vehicle_route([7, 8], [True, False])

    assert sent == [
        (
            {
                "messageType": "vehicleRoute",
                "data": [
                    {"vehicleId": 7, "isPrivate": True},
                    {"vehicleId": 8, "isPrivate": False},
                ],
            },
            True,
        )
    ]
    assert response["data"][0] == route_record


def test_vehicle_route_query_without_ids_requests_fleet_index():
    sent = []
    client = make_client(ok_response("vehicleRoute"), sent)

    client.query_vehicle_route()

    assert sent == [({"messageType": "vehicleRoute"}, True)]


def test_vehicle_route_query_rejects_mismatched_private_flags():
    sent = []
    client = make_client(ok_response("vehicleRoute"), sent)

    with pytest.raises(ValueError, match="same length as id"):
        client.query_vehicle_route([7, 8], [True])

    assert sent == []


def test_connector_path_query_preserves_opaque_ids_and_optional_path_ids():
    sent = []
    response_record = {
        "connectorId": "cont/58/-14",
        "connectorPathId": 2,
        "internalEdgeIds": [":452_4_0", ":452_14_0"],
        "sourceLaneId": "58_2",
        "targetLaneId": "-14_2",
        "status": "ok",
    }
    client = make_client(ok_response("connectorPath", [response_record]), sent)

    response = client.query_connector_path(
        ["cont/58/-14", "-36_-37"],
        [2, None],
    )

    assert sent == [
        (
            {
                "messageType": "connectorPath",
                "data": [
                    {"connectorId": "cont/58/-14", "connectorPathId": 2},
                    {"connectorId": "-36_-37"},
                ],
            },
            True,
        )
    ]
    assert response["data"][0] == response_record


@pytest.mark.parametrize("path_id", [-1, 1.5, True, "not-an-integer"])
def test_connector_path_query_rejects_invalid_path_id(path_id):
    sent = []
    client = make_client(ok_response("connectorPath"), sent)

    with pytest.raises(ValueError, match="non-negative integer"):
        client.query_connector_path("cont/A/B", path_id)

    assert sent == []


def test_routing_graph_preserves_latest_travel_time_estimator_fields():
    record = {
        "segmentId": "A",
        "length": 100.0,
        "speedLimit": 10.0,
        "travelTime": 12.0,
        "travelTimeP90": 18.0,
        "routingWeight": 13.0,
        "travelTimeConfidence": 0.75,
        "travelTimeEffectiveSampleCount": 4.5,
        "travelTimeSampleAgeSeconds": 2.0,
        "travelTimeLiveVehicleCount": 3,
        "travelTimeStoppedFraction": 0.25,
        "travelTimeLiveLowerBound": 11.0,
        "travelTimeLiveMeanSpeed": 8.5,
        "travelTimeEstimateSource": "completed_and_live",
    }

    attrs = METSRClient._routing_node_attrs_from(record)
    edge_attrs = METSRClient._routing_edge_attrs_from(attrs)

    assert attrs["travel_time_p90"] == 18.0
    assert attrs["travel_time_confidence"] == 0.75
    assert attrs["travel_time_effective_sample_count"] == 4.5
    assert attrs["travel_time_sample_age_seconds"] == 2.0
    assert attrs["travel_time_live_vehicle_count"] == 3
    assert attrs["travel_time_stopped_fraction"] == 0.25
    assert attrs["travel_time_live_lower_bound"] == 11.0
    assert attrs["travel_time_live_mean_speed"] == 8.5
    assert attrs["travel_time_estimate_source"] == "completed_and_live"
    assert edge_attrs["travel_time_p90"] == 18.0
    assert edge_attrs["travel_time_confidence"] == 0.75


def test_static_topology_graph_keeps_latest_metric_version_metadata():
    client = object.__new__(METSRClient)
    client._network_file_sha256 = lambda: None
    client._routing_topology_cache_schema = lambda include_center: 1

    graph = client._build_static_topology_graph(
        [{"segmentId": "A", "downstreamIds": [], "length": 10.0}],
        {
            "tick": 20,
            "topologyVersion": 4,
            "metricVersion": 9,
            "snapshotRequired": False,
        },
    )

    assert graph.graph["tick"] == 20
    assert graph.graph["topology_version"] == 4
    assert graph.graph["metric_version"] == 9
    assert graph.graph["weight_version"] == 9


def test_visualization_discovers_cosim_only_vehicle():
    client = object.__new__(METSRClient)
    client.viz_stream_lock = threading.Lock()
    client._attack_vehicle_keys = set()
    client.query_on_road_vehicles = lambda roadID=None: ok_response(
        "onRoadVehicles"
    )
    client.query_cosim_vehicle = lambda: ok_response(
        "coSimVehicle",
        [{"vehicleId": 11, "isPrivate": True}],
    )
    client.query_vehicle = lambda **kwargs: ok_response(
        "vehicle",
        [
            {
                "vehicleId": 11,
                "isPrivate": True,
                "segmentId": "-51",
                "x": 1.0,
                "y": 2.0,
            }
        ],
    )

    records = client._query_viz_stream_vehicle_records()

    assert [record["vehicleId"] for record in records] == [11]
    assert records[0]["_viz_private_veh"] is True


@pytest.mark.parametrize("method", ["query_boundary_vehicle", "queryBoundaryVeh"])
def test_boundary_vehicle_query_uses_native_read_only_protocol(method):
    response = ok_response("boundaryVehicle", [{
        "vehicleId": 17,
        "isPrivate": False,
        "segmentId": "native/exit",
        "controlMode": "native",
        "coordinateTrail": [[1.0, 2.0]],
    }])
    sent = []
    client = make_client(response, sent)

    assert getattr(client, method)() is response
    assert sent == [({"messageType": "boundaryVehicle"}, True)]
