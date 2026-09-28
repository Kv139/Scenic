"""Wire contract for optional co-simulation segment hints and native handback."""
from unittest.mock import Mock

import pytest

from scenic.simulators.metsr.client import METSRClient


def client_with_reply(records):
    client = object.__new__(METSRClient)
    reply = {"messageType": "teleportCoSimVeh", "status": "ok", "data": records}
    client.send_receive_msg = Mock(return_value=reply)
    return client, reply


def test_ordinary_pose_uses_only_native_request_fields():
    client, reply = client_with_reply([{
        "vehicleId": 7, "status": "ok", "segmentId": "A", "segmentType": "road",
        "laneIndex": 0, "controlMode": "cosim", "releasedFromCoSim": False}])
    assert client.teleport_cosim_vehicle(7, 1, 2, 90, speed=5, z=.5,
                                       private_veh=True, transform_coords=True) is reply
    client.send_receive_msg.assert_called_once_with({
        "messageType": "teleportCoSimVeh", "data": [{"vehicleId": 7,
        "x": 1, "y": 2, "z": .5, "bearing": 90, "speed": 5,
        "isPrivate": True, "transformCoordinates": True}]}, ignore_heartbeats=True)


@pytest.mark.parametrize("selector", [{"lane_index": 1}, {"connector_path_id": 2}])
def test_native_handback_sends_target_and_preserves_ack(selector):
    client, reply = client_with_reply([{
        "vehicleId": 7, "status": "ok", "segmentId": "native-target",
        "controlMode": "native", "releasedFromCoSim": True}])
    assert client.teleport_cosim_vehicle(7, 1, 2, 90, segment_id="native-target", **selector) is reply
    record = client.send_receive_msg.call_args.args[0]["data"][0]
    assert record["segmentId"] == "native-target"
    assert set(record) == {"vehicleId", "x", "y", "z", "bearing", "speed",
                           "isPrivate", "transformCoordinates", "segmentId",
                           "laneIndex" if "lane_index" in selector else "connectorPathId"}


def test_mixed_batch_omits_selectors_on_ordinary_records():
    client, reply = client_with_reply([])
    client.teleport_cosim_vehicle([7, 8], [1, 2], [3, 4], 90,
                                  segment_id=[None, "B"], lane_index=[None, 1])
    ordinary, handoff = client.send_receive_msg.call_args.args[0]["data"]
    assert "segmentId" not in ordinary and "laneIndex" not in ordinary
    assert handoff["segmentId"] == "B" and handoff["laneIndex"] == 1


def test_unimplemented_pose_mode_is_not_silently_sent_to_native():
    client, _ = client_with_reply([])
    with pytest.raises(TypeError, match="pose_only"):
        client.teleport_cosim_vehicle(7, 1, 2, 90, pose_only=True)
    client.send_receive_msg.assert_not_called()
