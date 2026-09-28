"""Native road ownership changes validate the whole reply before committing."""

from copy import deepcopy

import pytest

from scenic.simulators.metsr.bridge_utils import (
    METSRControlError,
    METSRCosimState,
    apply_road_release,
    apply_road_takeover,
)


def takeover_response(*roads):
    return {
        "messageType": "setCoSimRoad",
        "status": "ok",
        "data": [
            {
                "roadId": road,
                "status": "ok",
                "connectors": [],
                "connectorIds": [],
            }
            for road in roads
        ],
    }


def test_road_takeover_does_not_change_another_simulations_state():
    first = METSRCosimState()
    second = METSRCosimState()
    connector = {
        "connectorId": "opaque-connector",
        "sourceRoadId": "A",
        "targetRoadId": "B",
        "internalEdgeIds": [":junction_0"],
    }
    response = takeover_response("A")
    response["data"][0].update(connectors=[connector], connectorIds=["opaque-connector"])

    apply_road_takeover(first, response, ["A"])
    first.admitted_vehicles[7] = {"roadId": "A"}
    first.road_cache["A", 0] = [(1.0, 2.0)]

    assert first.controlled_segments == {"A", ":junction_0"}
    assert second == METSRCosimState()


@pytest.mark.parametrize("failure", ["envelope", "topology", "missing_road"])
def test_failed_takeover_preserves_all_previous_ownership(failure):
    state = METSRCosimState()
    apply_road_takeover(state, takeover_response("existing"), ["existing"])
    before = deepcopy(state)
    response = takeover_response("A", "B")
    if failure == "envelope":
        response["status"] = "error"
    elif failure == "topology":
        response["data"][1]["connectorIds"] = ["missing"]
    else:
        response["data"].pop()

    with pytest.raises(METSRControlError):
        apply_road_takeover(state, response, ["A", "B"])

    assert state == before


def test_confirmed_road_release_removes_owned_segments():
    state = METSRCosimState()
    apply_road_takeover(state, takeover_response("A"), ["A"])
    response = {
        "messageType": "releaseCoSimRoad",
        "status": "ok",
        "data": [
            {
                "roadId": "A",
                "status": "ok",
                "releasedConnectors": [],
                "releasedConnectorIds": [],
                "connectors": [],
                "connectorIds": [],
            }
        ],
    }

    assert apply_road_release(state, "A", response)
    assert not state.controlled_roads
    assert not state.controlled_segments
    assert not state.road_connectors


@pytest.mark.parametrize("allow_defer", [False, True])
def test_blocked_road_release_keeps_ownership(allow_defer):
    state = METSRCosimState()
    apply_road_takeover(state, takeover_response("A"), ["A"])
    before = deepcopy(state)
    response = {
        "messageType": "releaseCoSimRoad",
        "status": "partial",
        "data": [
            {
                "roadId": "A",
                "status": "error",
                "errorCode": "RELEASE_BLOCKED",
                "retryable": True,
            }
        ],
    }

    if allow_defer:
        assert apply_road_release(state, "A", response, allow_defer=True) is False
    else:
        with pytest.raises(METSRControlError):
            apply_road_release(state, "A", response)

    assert state == before
