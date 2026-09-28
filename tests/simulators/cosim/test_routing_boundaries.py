"""Native routing results keep selection separate from coordinator mutations."""

import math
from types import SimpleNamespace

from scenic.simulators.metsr import cosim_routing_utils as routing
from scenic.simulators.metsr.cosim_utils import METSRCosimState
from scenic.simulators.metsr.network import NetworkMetadata


def test_bubble_selection_repairs_shared_fragments_without_committing_state():
    first = SimpleNamespace(id="first")
    second = SimpleNamespace(id="second")
    metadata = NetworkMetadata(
        lane_mapping={
            "first_-1": ["native_0"],
            "second_-1": ["native_1"],
        }
    )
    state = METSRCosimState(metadata=metadata)
    state.controlled_roads = {"old": True}
    state.active_bubble_roads = {"old"}
    source = [first]

    selection = routing.classify_bubble_roads(
        state, source, {"first": first, "second": second}
    )

    assert source == [first]
    assert selection.roads == (first, second)
    assert selection.native_road_ids == frozenset({"native"})
    assert selection.entering == ("native",)
    assert selection.leaving == ("old",)
    assert state.controlled_roads == {"old": True}
    assert state.active_bubble_roads == {"old"}
    # Selecting both fragments still proposes only one native takeover.
    together = routing.classify_bubble_roads(
        state, [first, second], {"first": first, "second": second}
    )
    assert together == selection


def test_road_and_lane_selection_share_projection_cache_and_tie_order():
    state = METSRCosimState(
        metadata=NetworkMetadata(
            lane_indices={"road_4": 0, "road_5": 1},
        )
    )
    state.valid_roads = ["road"]
    calls = []

    def centerline(road, *, lane_index, transform_coords):
        calls.append((road, lane_index, transform_coords))
        y = 2.0 * lane_index
        return {"data": [{"centerline": [(0.0, y, 0.0), (10.0, y, 0.0)]}]}

    client = SimpleNamespace(query_centerline=centerline)
    lanes = ["road_9", "road_4", "road_5"]  # The first lane is filtered out.
    position = (12.0, 1.0, 0.0)
    assert routing.identify_nearest_road(state, client, position, lanes) == "road"
    lane, distance, terminal = routing.nearest_mapped_lane_projection(
        state, client, position, lanes, "road"
    )
    assert lane == 0
    assert distance == math.sqrt(5.0)
    assert terminal
    assert calls == [("road", 0, True), ("road", 1, True)]
