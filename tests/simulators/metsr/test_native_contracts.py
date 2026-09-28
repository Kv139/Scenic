"""Native batch identity and shared metadata contracts."""

import copy
from dataclasses import FrozenInstanceError
import pickle
from unittest.mock import Mock

import pytest

from scenic.simulators.metsr.bridge_utils import (
    METSRControlError,
    METSRCosimState,
    collect_metsr_vehicle_data,
    validate_queue_admissions,
)
from scenic.simulators.metsr.client import METSRClient
from scenic.simulators.metsr.network_utils import NetworkMetadata, mapped_lane_query


def client_for(operation):
    client = METSRClient.__new__(METSRClient)
    client.send_receive_msg = Mock(
        return_value={"messageType": operation, "status": "ok", "data": []}
    )
    return client


@pytest.mark.parametrize("field", ["private_veh", "transform_coords"])
@pytest.mark.parametrize("value", [[True], (True, False, True)])
def test_vehicle_query_rejects_incomplete_parallel_fields(field, value):
    client = client_for("vehicle")
    with pytest.raises(ValueError, match="same length"):
        client.query_vehicle([10, 20], **{field: value})
    client.send_receive_msg.assert_not_called()


def test_vehicle_query_supports_tuple_ids_and_scalar_flags():
    client = client_for("vehicle")
    client.query_vehicle((10, 20), private_veh=True, transform_coords=(False, True))
    message = client.send_receive_msg.call_args.args[0]
    assert message["data"] == [
        {"vehicleId": 10, "isPrivate": True, "transformCoordinates": False},
        {"vehicleId": 20, "isPrivate": True, "transformCoordinates": True},
    ]


@pytest.mark.parametrize(
    "operation,method,kwargs",
    [
        (
            "initializeCoSimVeh",
            "initialize_cosim_vehicle",
            {"x": 0, "y": 0, "bearing": 0, "destination_road_id": "D"},
        ),
        ("teleportCoSimVeh", "teleport_cosim_vehicle", {"x": 0, "y": 0, "bearing": 0}),
        ("teleportDigitalTwinVeh", "teleport_digital_twin_vehicle", {"x": 0, "y": 0}),
    ],
)
def test_pose_commands_validate_all_batches_before_sending(operation, method, kwargs):
    client = client_for(operation)
    kwargs["x"] = [0]
    with pytest.raises(ValueError, match="same length"):
        getattr(client, method)([10, 20], **kwargs)
    client.send_receive_msg.assert_not_called()


def test_initialization_keeps_segment_hint_precedence_and_zero_selectors():
    client = client_for("initializeCoSimVeh")
    client.initialize_cosim_vehicle(
        10,
        x=0,
        y=0,
        bearing=0,
        destination_road_id="D",
        segment_id="connector",
        road_id="legacy-road",
        connector_path_id=0,
    )
    record = client.send_receive_msg.call_args.args[0]["data"][0]
    assert record["segmentId"] == "connector"
    assert record["connectorPathId"] == 0
    assert record["x"] == record["y"] == record["bearing"] == 0


def test_pose_queries_attach_out_of_order_results_by_identity():
    first, second = object(), object()
    records = [{"vehicleId": "20", "x": 2}, {"vehicleId": 10, "x": 1}]
    client = Mock()
    client.query_vehicle.return_value = {
        "messageType": "vehicle",
        "status": "ok",
        "data": records,
    }
    result = collect_metsr_vehicle_data(client, {first: 10, second: 20})
    assert result[first] is records[1]
    assert result[second] is records[0]


@pytest.mark.parametrize("ids", [[10, 10], [10, 30], [10], [10, None]])
def test_pose_queries_reject_wrong_or_missing_identities(ids):
    client = Mock()
    client.query_vehicle.return_value = {
        "messageType": "vehicle",
        "status": "ok",
        "data": [{"vehicleId": vehicle_id} for vehicle_id in ids],
    }
    with pytest.raises(METSRControlError):
        collect_metsr_vehicle_data(client, {object(): 10, object(): 20})


def test_queue_admission_preserves_retryable_results_after_identity_validation():
    requests = [{"vehicleId": 10, "roadId": "A"}, {"vehicleId": 20, "roadId": "B"}]
    response = {
        "data": [
            {"vehicleId": 20, "roadId": "B", "status": "error", "retryable": True},
            {"vehicleId": 10, "roadId": "A", "status": "ok", "isPrivate": True},
        ]
    }
    assert validate_queue_admissions(response, requests) == {10: "A"}


def test_metadata_copies_nested_inputs_and_shares_derived_indexes():
    lanes = {"scenic_1": ["native_2"]}
    indices = {"native_2": 0}
    connections = {":internal": [["native_2", "next_1"]]}
    signals = {"light_1": ["signal"]}
    metadata = NetworkMetadata(
        lane_mapping=lanes,
        lane_indices=indices,
        internal_edge_connections=connections,
        signal_mapping=signals,
    )
    first, second = METSRCosimState(metadata=metadata), METSRCosimState(metadata=metadata)
    lanes["scenic_1"].append("unrelated_0")
    indices["native_2"] = 9
    connections[":internal"][0][0] = "changed"
    signals["light_1"].clear()
    assert metadata.lane_mapping == {"scenic_1": ("native_2",)}
    assert metadata.internal_edge_connections == {":internal": (("native_2", "next_1"),)}
    assert metadata.signal_mapping == {"light_1": ("signal",)}
    assert mapped_lane_query(first, "native_2") == ("native", 0)
    assert first.metadata.roads_by_scenic is second.metadata.roads_by_scenic
    assert metadata.roads_by_scenic == {"scenic": frozenset({"native"})}
    with pytest.raises(TypeError):
        metadata.lane_indices["native_2"] = 3
    with pytest.raises(FrozenInstanceError):
        metadata.lane_mapping = {}
    first.controlled_roads["native"] = True
    first.admitted_vehicles[10] = {"roadId": "native"}
    assert second.controlled_roads == second.admitted_vehicles == {}


@pytest.mark.parametrize(
    "roundtrip", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))]
)
@pytest.mark.parametrize("lane_indices", [None, {"native_2": 0}])
def test_metadata_serialization_rebuilds_immutable_indexes(roundtrip, lane_indices):
    metadata = NetworkMetadata(
        lane_mapping={"scenic_1": ["native_2"]},
        lane_indices=lane_indices,
        internal_lane_connections={":junction_0": [("native_2", "next_1")]},
        internal_edge_connections={":junction": [("native_2", "next_1")]},
        signal_mapping={"light_0": ["signal"]},
    )
    restored = roundtrip(metadata)
    assert restored == metadata
    assert restored is not metadata
    assert restored.roads_by_scenic == {"scenic": frozenset({"native"})}
    assert restored.carla_keys_by_lane == {"native_2": frozenset({"scenic_1"})}
    assert restored.roads_by_scenic is not metadata.roads_by_scenic
    with pytest.raises(TypeError):
        restored.lane_mapping["scenic_1"] = ("changed_0",)
    with pytest.raises(TypeError):
        restored.roads_by_scenic["scenic"] = frozenset({"changed"})
    with pytest.raises(FrozenInstanceError):
        restored.signal_mapping = {}
