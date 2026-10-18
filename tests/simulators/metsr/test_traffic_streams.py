"""Native traffic streams preserve zone or original-road trip endpoints."""

from unittest.mock import Mock

import pytest

from scenic.simulators.metsr.client import METSRClient
import scenic.simulators.metsr.simulator as metsr_simulator
from scenic.simulators.metsr.traffic_flows import ConstantTrafficFlow, NormalTrafficFlow
from tests.utils import compileScenic, sampleScene


@pytest.fixture
def run_traffic(monkeypatch):
    # Exercise the real Scenic model, simulator dispatch, and request serialization
    # without connecting to a METS-R server.
    client = METSRClient.__new__(METSRClient)
    client.verbose = False
    client.reset = Mock()
    client.tick = Mock()
    client.close = Mock()
    client.send_receive_msg = Mock(
        side_effect=lambda message, **kwargs: {
            "messageType": message["messageType"],
            "status": "ok",
            "data": [],
        }
    )
    client.query_vehicle = Mock(
        side_effect=lambda ids, *args: {
            "data": [
                {"onRoad": True, "state": 1, "x": 0, "y": 0, "speed": 0, "bearing": 0}
                for _ in ids
            ]
        }
    )
    monkeypatch.setattr(metsr_simulator, "METSRClient", lambda **kwargs: client)
    # Always spawn, so these tests check endpoint propagation independently of
    # flow shape or random sampling.
    for flow_class in (ConstantTrafficFlow, NormalTrafficFlow):
        monkeypatch.setattr(flow_class, "expected_vehs", lambda self, start, end: 1)

    def run(code):
        scenario = compileScenic("model scenic.simulators.metsr.model\n" + code)
        scene = sampleScene(scenario)
        simulator = scenario.getSimulator()
        try:
            simulation = simulator.simulate(scene, maxSteps=1, maxIterations=1)
            assert simulation is not None
        finally:
            simulator.destroy()
        return [call.args[0] for call in client.send_receive_msg.call_args_list]

    return run


@pytest.mark.parametrize("flag", ["", ", isRoad=False", ", isRoad=True"])
@pytest.mark.parametrize(
    "stream,arguments,commuter",
    [
        ("GeneratePrivateTrip", "", False),
        ("TrafficStream", ", ConstantTrafficFlow(1)", False),
        ("ConstantTrafficStream", ", 1", False),
        ("NormalTrafficStream", ", 1, 21600, 60", False),
        ("CommuterTrafficStream", ", 1, 21600, 64800, 60", True),
    ],
)
def test_traffic_stream_endpoint_modes(run_traffic, flag, stream, arguments, commuter):
    is_road = flag == ", isRoad=True"
    origin, destination = ("-12345#0", "67890#1") if is_road else (10, 20)
    messages = run_traffic(
        f"scenario Main():\n"
        f"    compose:\n"
        f"        do {stream}({origin!r}, {destination!r}{arguments}{flag})\n"
    )

    assert messages
    operation = "generateTripsByRoad" if is_road else "generateTrip"
    endpoint_type = "Road" if is_road else "Zone"
    actual_trips = []
    for vehicle_id, message in enumerate(messages):
        assert message["messageType"] == operation
        assert len(message["data"]) == 1
        record = message["data"][0]
        assert set(record) == {
            "vehicleId",
            f"origin{endpoint_type}Id",
            f"destination{endpoint_type}Id",
        }
        assert record["vehicleId"] == vehicle_id
        actual_trips.append(
            (
                record[f"origin{endpoint_type}Id"],
                record[f"destination{endpoint_type}Id"],
            )
        )
    expected_trips = {(origin, destination)}
    if commuter:
        expected_trips.add((destination, origin))
    assert set(actual_trips) == expected_trips


@pytest.mark.parametrize("object_type", ["PrivateCar", "Object"])
def test_objects_without_road_flag_keep_zone_trips(run_traffic, object_type):
    messages = run_traffic(
        f"new {object_type} with origin 10, with destination 20\n"
    )
    assert messages == [
        {
            "messageType": "generateTrip",
            "data": [{"vehicleId": 0, "originZoneId": 10, "destinationZoneId": 20}],
        }
    ]
