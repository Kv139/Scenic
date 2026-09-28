"""Bubble ownership follows the ego road's local connected component."""

from types import SimpleNamespace
import unittest

from scenic.core.simulators import SimulationCreationError
from scenic.core.vectors import Vector
from scenic.simulators.cosim import cosim_routing_utils as bubble_routing
from scenic.simulators.cosim.simulator import CosimSimulation, carla
from scenic.simulators.cosim.utils.network_helper import network_cache
from scenic.simulators.metsr.cosim_utils import METSRCosimState


class Road:
    def __init__(self, road_id, inside=True):
        self.id = road_id
        self.inside = inside
        self.lanes = []

    def intersects(self, region):
        return self.inside


def connect(source, target):
    incoming = SimpleNamespace(road=source, _predecessor=None, _successor=None)
    outgoing = SimpleNamespace(road=target, _predecessor=None, _successor=None)
    incoming._successor = outgoing
    source.lanes.append(incoming)
    target.lanes.append(outgoing)


class BubbleRoadTopologyTests(unittest.TestCase):
    def fixture(self):
        roads = {
            name: Road(name)
            for name in ("ground", "junction", "ground_exit", "bridge", "bridge_exit")
        }
        connect(roads["ground"], roads["junction"])
        connect(roads["junction"], roads["ground_exit"])
        connect(roads["bridge"], roads["bridge_exit"])
        workspace = SimpleNamespace(
            network=SimpleNamespace(
                allRoads=list(roads.values()), laneSections=[], intersections=[]
            )
        )
        return network_cache(workspace), roads

    def selected(self, cache, anchor=None):
        return {
            road.id for road in cache._get_bubble_roads(object(), anchor_road_id=anchor)
        }

    def test_ground_ego_excludes_overlapping_bridge(self):
        cache, _ = self.fixture()
        self.assertEqual(
            self.selected(cache, "ground"), {"ground", "junction", "ground_exit"}
        )

    def test_bridge_ego_selects_bridge_component(self):
        cache, _ = self.fixture()
        self.assertEqual(self.selected(cache, "bridge"), {"bridge", "bridge_exit"})

    def test_incoming_roads_are_included_against_traffic_direction(self):
        cache, _ = self.fixture()
        self.assertEqual(
            self.selected(cache, "ground_exit"), {"ground", "junction", "ground_exit"}
        )

    def test_connected_ramp_is_retained_without_height_cutoff(self):
        cache, roads = self.fixture()
        connect(roads["ground_exit"], roads["bridge"])
        self.assertEqual(self.selected(cache, "ground"), set(roads))

    def test_connection_outside_bubble_does_not_join_local_components(self):
        cache, roads = self.fixture()
        outside = Road("outside", inside=False)
        cache.network_roads.append(outside)
        connect(roads["ground_exit"], outside)
        connect(outside, roads["bridge"])
        self.assertEqual(
            self.selected(cache, "ground"), {"ground", "junction", "ground_exit"}
        )

    def test_without_anchor_preserves_geometric_selection(self):
        cache, roads = self.fixture()
        self.assertEqual(self.selected(cache), set(roads))

    def test_live_actor_height_selects_layer_before_stale_scenic_position(self):
        cache, _ = self.fixture()
        simulation = CosimSimulation.__new__(CosimSimulation)
        simulation.metsr_state = METSRCosimState()
        location = carla.Location(1, 2, 10)
        simulation.ego = SimpleNamespace(
            position=Vector(1, -2, 0),
            bubble=object(),
            carlaActor=SimpleNamespace(get_location=lambda: location),
        )
        simulation.map = SimpleNamespace(
            get_waypoint=lambda loc, **kwargs: SimpleNamespace(
                road_id="bridge" if loc.z == 10 else "ground"
            )
        )
        simulation.network_helper = cache
        self.assertEqual(
            {road.id for road in bubble_routing.get_bubble_roads(simulation)},
            {"bridge", "bridge_exit"},
        )

    def test_initial_ego_height_selects_layer_before_actor_exists(self):
        cache, _ = self.fixture()
        simulation = CosimSimulation.__new__(CosimSimulation)
        simulation.metsr_state = METSRCosimState()
        simulation.ego = SimpleNamespace(position=Vector(1, -2, 10), bubble=object())
        simulation.map = SimpleNamespace(
            get_waypoint=lambda loc, **kwargs: SimpleNamespace(
                road_id="bridge" if loc.z == 10 else "ground"
            )
        )
        simulation.network_helper = cache
        self.assertEqual(
            {road.id for road in bubble_routing.get_bubble_roads(simulation)},
            {"bridge", "bridge_exit"},
        )

    def test_missing_ego_map_anchor_fails_before_acquiring_other_roads(self):
        cache, _ = self.fixture()
        simulation = CosimSimulation.__new__(CosimSimulation)
        simulation.metsr_state = METSRCosimState()
        simulation.ego = SimpleNamespace(position=Vector(1, -2, 0), bubble=object())
        simulation.map = SimpleNamespace(get_waypoint=lambda loc, **kwargs: None)
        simulation.network_helper = cache
        with self.assertRaises(SimulationCreationError):
            bubble_routing.get_bubble_roads(simulation)

    def test_intersections_follow_selected_roads_instead_of_xy_overlap(self):
        cache, roads = self.fixture()
        ground = SimpleNamespace(
            roads=[roads["ground"]], maneuvers=[], intersects=lambda region: True
        )
        bridge = SimpleNamespace(
            roads=[roads["bridge"]], maneuvers=[], intersects=lambda region: True
        )
        cache.network_intersections = [ground, bridge]
        simulation = CosimSimulation.__new__(CosimSimulation)
        simulation.metsr_state = METSRCosimState()
        simulation.network_helper = cache
        self.assertEqual(
            bubble_routing.get_bubble_intersections(simulation, [roads["ground"]]),
            [ground],
        )

    def test_intersection_containing_only_selected_connector_is_retained(self):
        cache, roads = self.fixture()
        intersection = SimpleNamespace(
            roads=[],
            maneuvers=[
                SimpleNamespace(connectingLane=SimpleNamespace(road=roads["junction"]))
            ],
            intersects=lambda region: False,
        )
        cache.network_intersections = [intersection]
        simulation = CosimSimulation.__new__(CosimSimulation)
        simulation.metsr_state = METSRCosimState()
        simulation.network_helper = cache
        self.assertEqual(
            bubble_routing.get_bubble_intersections(simulation, [roads["junction"]]),
            [intersection],
        )


if __name__ == "__main__":
    unittest.main()
