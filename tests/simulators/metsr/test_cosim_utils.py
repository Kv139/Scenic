"""Per-run native ownership remains independent across co-simulations."""

from scenic.simulators.metsr.cosim_utils import (
    METSRCosimState,
    clear_carla_authoritative_segment,
    commit_takeover_ownership,
)


def test_road_takeover_does_not_change_another_simulations_state():
    first = METSRCosimState()
    second = METSRCosimState()
    connector = {
        "sourceRoadId": "A",
        "targetRoadId": "B",
        "internalEdgeIds": [":junction_0"],
    }

    commit_takeover_ownership(
        first,
        ["A"],
        {"A": {"opaque-connector"}},
        {"opaque-connector": connector},
    )
    first.admitted_vehicles[7] = {"roadId": "A"}
    first.road_cache["A", 0] = [(1.0, 2.0)]

    assert first.controlled_segments == {"A", ":junction_0"}
    assert second == METSRCosimState()


def test_native_segment_cleanup_leaves_carla_path_ownership_to_the_caller():
    class Vehicle:
        pass

    obj = Vehicle()
    path = [object()]
    obj.trajectory = path
    state = METSRCosimState(authoritative_segments={obj: "current-road"})

    clear_carla_authoritative_segment(state, obj)

    assert obj not in state.authoritative_segments
    assert obj.trajectory is path
