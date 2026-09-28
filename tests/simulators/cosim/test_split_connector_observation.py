"""Sequential SUMO junction fragments share one canonical ownership domain."""

import unittest

from scenic.simulators.cosim.simulator import CosimSimulation
from scenic.simulators.metsr import cosim_routing_utils as routing
from scenic.simulators.metsr.cosim_utils import METSRCosimState
from scenic.simulators.metsr.network import NetworkMetadata


class SplitConnectorObservationTests(unittest.TestCase):
    def fixture(self, different_owners=False):
        simulation = CosimSimulation.__new__(CosimSimulation)
        simulation.metsr_state = METSRCosimState(
            metadata=NetworkMetadata(lane_indices={"-2_3": 0})
        )
        simulation.metsr_state.road_cache = {}
        simulation.metsr_state.controlled_roads = {"-2": True}
        simulation.metsr_state.connector_records = {
            "cont/-49/-2": {
                "sourceRoadId": "-49",
                "targetRoadId": "-2",
                "internalEdgeIds": [":979_16", ":979_26"],
            }
        }
        simulation.metsr_state.road_connectors = {"-2": {"cont/-49/-2"}}
        simulation.metsr_state.internal_edge_to_connector = {
            ":979_16": "cont/-49/-2",
            ":979_26": "cont/-49/-2",
        }
        if different_owners:
            simulation.metsr_state.internal_edge_to_connector[":979_26"] = "other"
        simulation.metsr_state.controlled_segments = {"-2", ":979_16", ":979_26"}
        return simulation

    def test_split_internal_lane_resolves_shared_owner(self):
        simulation = self.fixture()
        road = routing.identify_nearest_road(
            simulation.metsr_state,
            None,
            (-132.2672576904297, 12.554266929626465, 0),
            [":979_16_0", ":979_26_0"],
        )
        self.assertEqual(
            simulation.metsr_state.internal_edge_to_connector.get(road), "cont/-49/-2"
        )

    def test_different_connector_owners_remain_ambiguous(self):
        simulation = self.fixture(different_owners=True)
        self.assertIsNone(
            routing.identify_nearest_road(
                simulation.metsr_state, None, (0, 0, 0), [":979_16_0", ":979_26_0"]
            )
        )


if __name__ == "__main__":
    unittest.main()
