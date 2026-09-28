"""Bubble ownership follows the ego road's local connected component."""

from types import SimpleNamespace
import unittest

import carla

from scenic.core.simulators import SimulationCreationError
from scenic.core.vectors import Vector
from scenic.simulators.cosim import bubble_utils


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


def make_topology(roads, intersections=()):
    return bubble_utils.BubbleTopology.from_network(
        SimpleNamespace(allRoads=roads, intersections=intersections)
    )


class BubbleRoadTopologyTests(unittest.TestCase):
    def fixture(self):
        roads = {
            name: Road(name)
            for name in ("ground", "junction", "ground_exit", "bridge", "bridge_exit")
        }
        connect(roads["ground"], roads["junction"])
        connect(roads["junction"], roads["ground_exit"])
        connect(roads["bridge"], roads["bridge_exit"])
        return make_topology(roads.values()), roads

    def selected(self, topology, anchor=None):
        return {
            road.id
            for road in bubble_utils.connected_bubble_roads(
                topology, object(), anchor_road_id=anchor
            )
        }

    def test_ground_ego_excludes_overlapping_bridge(self):
        topology, _ = self.fixture()
        self.assertEqual(
            self.selected(topology, "ground"), {"ground", "junction", "ground_exit"}
        )

    def test_bridge_ego_selects_bridge_component(self):
        topology, _ = self.fixture()
        self.assertEqual(self.selected(topology, "bridge"), {"bridge", "bridge_exit"})

    def test_incoming_roads_are_included_against_traffic_direction(self):
        topology, _ = self.fixture()
        self.assertEqual(
            self.selected(topology, "ground_exit"),
            {"ground", "junction", "ground_exit"},
        )

    def test_connected_ramp_is_retained_without_height_cutoff(self):
        topology, roads = self.fixture()
        connect(roads["ground_exit"], roads["bridge"])
        topology = make_topology(roads.values())
        self.assertEqual(self.selected(topology, "ground"), set(roads))

    def test_connection_outside_bubble_does_not_join_local_components(self):
        topology, roads = self.fixture()
        outside = Road("outside", inside=False)
        roads["outside"] = outside
        connect(roads["ground_exit"], outside)
        connect(outside, roads["bridge"])
        topology = make_topology(roads.values())
        self.assertEqual(
            self.selected(topology, "ground"), {"ground", "junction", "ground_exit"}
        )

    def test_without_anchor_preserves_geometric_selection(self):
        topology, roads = self.fixture()
        self.assertEqual(self.selected(topology), set(roads))

    def test_live_actor_height_selects_layer_before_stale_scenic_position(self):
        topology, _ = self.fixture()
        simulation = SimpleNamespace()
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
        simulation.bubble_topology = topology
        self.assertEqual(
            {road.id for road in bubble_utils.get_bubble_roads(simulation)},
            {"bridge", "bridge_exit"},
        )

    def test_initial_ego_height_selects_layer_before_actor_exists(self):
        topology, _ = self.fixture()
        simulation = SimpleNamespace()
        simulation.ego = SimpleNamespace(position=Vector(1, -2, 10), bubble=object())
        simulation.map = SimpleNamespace(
            get_waypoint=lambda loc, **kwargs: SimpleNamespace(
                road_id="bridge" if loc.z == 10 else "ground"
            )
        )
        simulation.bubble_topology = topology
        self.assertEqual(
            {road.id for road in bubble_utils.get_bubble_roads(simulation)},
            {"bridge", "bridge_exit"},
        )

    def test_missing_ego_map_anchor_fails_before_acquiring_other_roads(self):
        topology, _ = self.fixture()
        simulation = SimpleNamespace()
        simulation.ego = SimpleNamespace(position=Vector(1, -2, 0), bubble=object())
        simulation.map = SimpleNamespace(get_waypoint=lambda loc, **kwargs: None)
        simulation.bubble_topology = topology
        with self.assertRaises(SimulationCreationError):
            bubble_utils.get_bubble_roads(simulation)

    def test_intersections_follow_selected_roads_instead_of_xy_overlap(self):
        topology, roads = self.fixture()
        ground = SimpleNamespace(
            roads=[roads["ground"]], maneuvers=[], intersects=lambda region: True
        )
        bridge = SimpleNamespace(
            roads=[roads["bridge"]], maneuvers=[], intersects=lambda region: True
        )
        topology = make_topology(roads.values(), [ground, bridge])
        self.assertEqual(
            bubble_utils.intersections_for_roads(topology, [roads["ground"]]),
            [ground],
        )

    def test_intersection_containing_only_selected_connector_is_retained(self):
        topology, roads = self.fixture()
        intersection = SimpleNamespace(
            roads=[],
            maneuvers=[
                SimpleNamespace(connectingLane=SimpleNamespace(road=roads["junction"]))
            ],
            intersects=lambda region: False,
        )
        topology = make_topology(roads.values(), [intersection])
        self.assertEqual(
            bubble_utils.intersections_for_roads(topology, [roads["junction"]]),
            [intersection],
        )


if __name__ == "__main__":
    unittest.main()
