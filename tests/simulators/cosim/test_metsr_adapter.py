from contextlib import redirect_stdout
from io import StringIO
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from scenic.core.simulators import SimulationCreationError
from scenic.simulators.cosim.simulator import (
    carla,
    CARLA_OWNED_PROJECTION_TOLERANCE,
    COSIM_ADMISSION_SPAWN_MAX_TICKS,
    ROAD_OBSERVATION_DIRECT_SUCCESSOR,
    ROAD_OBSERVATION_ONE_SKIP,
    ROAD_OBSERVATION_PREDECESSOR,
    ROAD_OBSERVATION_SAME,
    ROAD_OBSERVATION_UNSUPPORTED,
    ROAD_OBSERVATION_UNKNOWN,
    CosimSimulation,
    METSRControlError,
    _render_step_interval,
)
from scenic.simulators.cosim.utils.network_helper import network_cache
from scenic.simulators.cosim.utils import utils as cosim_utils


def metsr_response(message_type, data=(), status="ok", **fields):
    """Build one current native METS-R response envelope for test doubles."""
    response = {
        "messageType": message_type,
        "status": status,
        "data": list(data),
    }
    response.update(fields)
    return response


class NetworkMappingTests(unittest.TestCase):
    def test_default_render_interval_is_one_simulated_second(self):
        self.assertEqual(_render_step_interval(0.1, None), 10)
        self.assertEqual(_render_step_interval(0.05, None), 20)
        self.assertEqual(_render_step_interval(0.1, 4), 4)

    def test_lane_suffix_removal_preserves_internal_road_id(self):
        workspace = SimpleNamespace(
            network=SimpleNamespace(
                laneSections=[], allRoads=[], intersections=[]
            )
        )
        cache = network_cache(
            workspace,
            {"1034_-1": [":1034_0_0"]},
            {":1034_0"},
        )

        self.assertEqual(
            cache.scenic_to_metsr_map_roads["1034"], {":1034_0"}
        )

    def test_metsr_lane_index_excludes_sumo_shoulders(self):
        network = """\
<net>
  <edge id="-29" type="highway.primary">
    <lane id="-29_0" type="shoulder"/>
    <lane id="-29_1" type="shoulder"/>
    <lane id="-29_2" type="driving"/>
    <lane id="-29_3" type="driving"/>
    <lane id="-29_4" type="driving"/>
    <lane id="-29_5" type="driving"/>
    <lane id="-29_6" type="driving"/>
    <lane id="-29_7" type="shoulder"/>
  </edge>
</net>
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "map.net.xml"
            path.write_text(network, encoding="utf-8")
            indices = cosim_utils.generate_metsr_lane_index_map(path)

        self.assertEqual(indices["-29_2"], 0)
        self.assertEqual(indices["-29_6"], 4)
        self.assertNotIn("-29_0", indices)

    def test_lane_connections_use_full_driving_lane_ids(self):
        network = """\
<net>
  <edge id="A" type="highway.primary">
    <lane id="A_0" index="0" type="shoulder"/>
    <lane id="A_1" index="1" type="driving"/>
  </edge>
  <edge id="B" type="highway.primary">
    <lane id="B_0" index="0" type="shoulder"/>
    <lane id="B_2" index="2" type="driving"/>
  </edge>
  <connection from="A" to="B" fromLane="1" toLane="2" via=":J_0_0"/>
  <connection from="A" to="B" fromLane="0" toLane="0"/>
</net>
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "map.net.xml"
            path.write_text(network, encoding="utf-8")
            connections = cosim_utils.generate_metsr_lane_connection_map(path)
            internal = cosim_utils.generate_metsr_internal_lane_connection_map(path)

        self.assertEqual(connections, {("A", "B"): [("A_1", "B_2")]})
        self.assertEqual(internal, {":J_0_0": [("A_1", "B_2")]})

    def test_internal_edge_connections_preserve_opaque_road_characters(self):
        network = """\
<net>
  <edge id="src_under/%:road" type="highway.primary">
    <lane id="src_under/%:road_0" index="0" type="driving"/>
  </edge>
  <edge id="dst_under/%:road" type="highway.primary">
    <lane id="dst_under/%:road_0" index="0" type="driving"/>
  </edge>
  <edge id=":452_4" function="internal">
    <lane id=":452_4_0" index="0"/>
  </edge>
  <edge id=":452_14" function="internal">
    <lane id=":452_14_0" index="0"/>
  </edge>
  <connection from="src_under/%:road" to="dst_under/%:road"
              fromLane="0" toLane="0" via=":452_4_0"/>
  <connection from=":452_4" to="dst_under/%:road"
              fromLane="0" toLane="0" via=":452_14_0"/>
</net>
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "map.net.xml"
            path.write_text(network, encoding="utf-8")
            edges = cosim_utils.generate_metsr_internal_edge_connection_map(path)

        self.assertEqual(
            edges,
            {
                ":452_4": [
                    ("src_under/%:road_0", "dst_under/%:road_0")
                ]
            },
        )


class AdapterStateTests(unittest.TestCase):
    @staticmethod
    def _route_waypoint(road_id, lane_id, s, x):
        location = SimpleNamespace(x=float(x), y=0.0, z=0.0)
        return SimpleNamespace(
            road_id=road_id,
            section_id=0,
            lane_id=lane_id,
            s=float(s),
            is_junction=False,
            transform=SimpleNamespace(location=location),
        )

    def _route_stitch_fixture(self, divergent=False):
        simulation = object.__new__(CosimSimulation)
        waypoints = {
            "A": self._route_waypoint(1, -1, 1.0, 0.0),
            "B": self._route_waypoint(2, -1, 1.0, 10.0),
            "C": self._route_waypoint(3, -1, 1.0, 20.0),
            "X": self._route_waypoint(4, -1, 1.0, 5.0),
        }
        simulation.scenic_to_metsr_map = {
            "1_-1": ["A_0"],
            "2_-1": ["B_0"],
            "3_-1": ["C_0"],
            "4_-1": ["X_0"],
        }
        simulation.metsr_lane_indices = {
            "A_0": 0,
            "B_0": 0,
            "C_0": 0,
            "X_0": 0,
        }
        simulation.metsr_lane_connections = {
            ("A", "B"): [("A_0", "B_0")],
            ("B", "C"): [("B_0", "C_0")],
        }
        simulation._sumo_lane_to_carla_keys = {
            "A_0": {"1_-1"},
            "B_0": {"2_-1"},
            "C_0": {"3_-1"},
        }
        simulation._carla_waypoints_by_key = {
            "1_-1": [waypoints["A"]],
            "2_-1": [waypoints["B"]],
            "3_-1": [waypoints["C"]],
        }
        simulation.metsr_road_cache = {
            ("A", 0): [(0.0, 0.0, 0.0), (9.0, 0.0, 0.0)],
            ("B", 0): [(10.0, 0.0, 0.0), (19.0, 0.0, 0.0)],
            ("C", 0): [(20.0, 0.0, 0.0), (29.0, 0.0, 0.0)],
        }

        def waypoint_at(location, **kwargs):
            return min(
                (waypoints["A"], waypoints["B"], waypoints["C"]),
                key=lambda waypoint: abs(waypoint.transform.location.x - location.x),
            )

        calls = []

        def trace_route(start, target):
            calls.append((start.x, target.x))
            if target.x == 10.0:
                if divergent:
                    return [
                        (waypoints["A"], None),
                        (waypoints["X"], None),
                        (waypoints["B"], None),
                    ]
                return [(waypoints["A"], None), (waypoints["B"], None)]
            return [(waypoints["B"], None), (waypoints["C"], None)]

        simulation.map = SimpleNamespace(get_waypoint=waypoint_at)
        simulation.grp = SimpleNamespace(trace_route=trace_route)
        obj = SimpleNamespace(
            name="route-car",
            position=SimpleNamespace(x=0.0, y=0.0),
            carlaActor=SimpleNamespace(
                get_location=lambda: waypoints["A"].transform.location
            ),
        )
        return simulation, obj, calls

    def _route_matcher(self, connections=()):
        simulation = object.__new__(CosimSimulation)
        simulation.metsr_lane_connections = {
            connection: [("source_lane", "target_lane")]
            for connection in connections
        }
        return simulation._route_candidate_sequence_matches

    def test_route_candidate_matching_requires_every_macro_road(self):
        matches = self._route_matcher()
        self.assertTrue(
            matches([{"A"}, {"A", "B"}, {"B"}, {"C"}], ["A", "B", "C"])
        )
        self.assertFalse(
            matches([{"A"}, {"C"}], ["A", "B", "C"])
        )

    def test_route_candidate_matching_accepts_one_direct_predecessor_run(self):
        matches = self._route_matcher({("4", "A")})

        self.assertTrue(matches([{"4"}, {"4"}, {"A"}, {"B"}], ["A", "B"]))

    def test_route_candidate_matching_rejects_unconnected_leading_road(self):
        matches = self._route_matcher()

        self.assertFalse(matches([{"4"}, {"A"}, {"B"}], ["A", "B"]))

    def test_route_candidate_matching_rejects_intermediate_extra_road(self):
        matches = self._route_matcher()

        self.assertFalse(matches([{"A"}, {"X"}, {"B"}], ["A", "B"]))

    def test_route_candidate_matching_rejects_trailing_extra_road(self):
        matches = self._route_matcher()

        self.assertFalse(matches([{"A"}, {"B"}, {"X"}], ["A", "B"]))

    def test_route_candidate_matching_rejects_multiple_predecessor_runs(self):
        matches = self._route_matcher({("3", "A"), ("4", "A")})

        self.assertFalse(
            matches([{"3"}, {"4"}, {"A"}, {"B"}], ["A", "B"])
        )

    def test_carla_trajectory_stitches_every_adjacent_macro_road(self):
        simulation, obj, calls = self._route_stitch_fixture()

        locations = simulation.generate_carla_trajectory(["A", "B", "C"], obj)

        self.assertEqual([location.x for location in locations], [0.0, 10.0, 20.0])
        self.assertEqual(calls, [(0.0, 10.0), (10.0, 20.0)])

    def test_carla_trajectory_fails_closed_on_segment_divergence(self):
        simulation, obj, _ = self._route_stitch_fixture(divergent=True)

        with self.assertRaisesRegex(
            SimulationCreationError, "diverged from the METS-R macro route"
        ):
            simulation.generate_carla_trajectory(["A", "B", "C"], obj)

    def test_carla_trajectory_trims_one_confirmed_stale_prefix(self):
        simulation, obj, calls = self._route_stitch_fixture()
        start = SimpleNamespace(x=10.0, y=0.0, z=0.0)
        obj.carlaActor.get_location = lambda: start

        locations = simulation.generate_carla_trajectory(["A", "B", "C"], obj)

        self.assertEqual([location.x for location in locations], [10.0, 20.0])
        self.assertEqual(calls, [(10.0, 20.0)])

    def test_carla_trajectory_rejects_unconfirmed_stale_prefix(self):
        simulation, obj, _ = self._route_stitch_fixture()
        start = SimpleNamespace(x=10.0, y=0.0, z=0.0)
        obj.carlaActor.get_location = lambda: start
        simulation.scenic_to_metsr_map["2_-1"] = ["B_9"]
        simulation.metsr_lane_indices["B_9"] = 9

        with self.assertRaisesRegex(
            SimulationCreationError, "not direct arrivals"
        ):
            simulation.generate_carla_trajectory(["A", "B", "C"], obj)

    def test_carla_trajectory_keeps_arrival_lane_separate_from_departure(self):
        simulation, obj, _ = self._route_stitch_fixture()
        simulation.metsr_lane_indices.update({"B_5": 2, "B_7": 3})
        simulation._align_carla_route_start = lambda route, start: tuple(route)
        calls = {"candidate": [], "stitch": []}

        def candidates(route, start, required_initial_lane=None):
            calls["candidate"].append(required_initial_lane)
            # B_7 is the route-selected departure lane; B_5 is the
            # independent authoritative pending-arrival lane.
            return [["B_7", "C_0"]]

        simulation._candidate_metsr_lane_chains = candidates
        waypoint = self._route_waypoint(2, -1, 1.0, 10.0)

        def stitch(route, start, chain, required_initial_lane=None):
            calls["stitch"].append((tuple(chain), required_initial_lane))
            return [(waypoint, None)]

        simulation._stitch_carla_route = stitch

        locations = simulation.generate_carla_trajectory(
            ["B", "C"], obj, required_first_lane_id=2
        )

        self.assertEqual(calls["candidate"], ["B_5"])
        self.assertEqual(calls["stitch"], [(('B_7', 'C_0'), "B_5")])
        self.assertEqual([location.x for location in locations], [10.0])

        with self.assertRaisesRegex(
            SimulationCreationError,
            "authoritative compact lane 4 of road B; found",
        ):
            simulation.generate_carla_trajectory(
                ["B", "C"], obj, required_first_lane_id=4
            )

    def test_forward_lane_anchor_ignores_waypoints_behind_current_pose(self):
        simulation = object.__new__(CosimSimulation)
        current = self._route_waypoint(1, -1, 60.0, 60.0)
        behind = self._route_waypoint(1, -2, 50.0, 50.0)
        ahead = self._route_waypoint(1, -2, 85.0, 85.0)
        simulation.map = SimpleNamespace(
            get_waypoint=lambda *args, **kwargs: current
        )
        simulation.scenic_to_metsr_map = {
            "1_-1": ["A_0"],
            "1_-2": ["A_1"],
        }
        simulation.metsr_lane_indices = {"A_0": 0, "A_1": 1}
        simulation.metsr_lane_connections = {}
        simulation._sumo_lane_to_carla_keys = {"A_1": {"1_-2"}}
        simulation._carla_waypoints_by_key = {"1_-2": [behind, ahead]}
        simulation.metsr_road_cache = {
            ("A", 1): [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        }
        simulation.grp = SimpleNamespace(
            _sampling_resolution=2.0,
            trace_route=lambda start, target: [(current, None), (ahead, None)],
        )

        anchor, _ = simulation._forward_lane_anchor_trace(
            "A_1", current.transform.location, "A", 1
        )

        self.assertIs(anchor, ahead)

    def test_forward_lane_anchor_retries_when_planner_overshoots_candidate(self):
        simulation = object.__new__(CosimSimulation)
        current = self._route_waypoint(1, -1, 0.0, 0.0)
        near = self._route_waypoint(1, -2, 10.0, 10.0)
        reachable = self._route_waypoint(1, -2, 32.0, 32.0)
        simulation.map = SimpleNamespace(
            get_waypoint=lambda *args, **kwargs: current
        )
        simulation.scenic_to_metsr_map = {
            "1_-1": ["A_0"],
            "1_-2": ["A_3"],
        }
        simulation.metsr_lane_indices = {"A_0": 0, "A_3": 3}
        simulation.metsr_lane_connections = {}
        simulation._sumo_lane_to_carla_keys = {"A_3": {"1_-2"}}
        simulation._carla_waypoints_by_key = {"1_-2": [near, reachable]}
        simulation.metsr_road_cache = {
            ("A", 3): [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        }
        requested_targets = []

        def trace_route(start, target):
            requested_targets.append(target.x)
            # A three-lane correction can overshoot a nearby requested anchor.
            # Appending that anchor would make the installed path run backward.
            return [(current, None), (reachable, None)]

        simulation.grp = SimpleNamespace(
            _sampling_resolution=2.0,
            trace_route=trace_route,
        )

        anchor, trace = simulation._forward_lane_anchor_trace(
            "A_3",
            current.transform.location,
            "A",
            1,
            prefer_early=True,
        )

        self.assertIs(anchor, reachable)
        self.assertEqual(requested_targets, [10.0])
        self.assertIs(trace[-1][0], reachable)
        self.assertNotIn(near, [item[0] for item in trace])

    def test_forward_lane_anchor_fails_when_road_is_too_short(self):
        simulation = object.__new__(CosimSimulation)
        current = self._route_waypoint(1, -1, 93.0, 93.0)
        ahead = self._route_waypoint(1, -2, 98.0, 98.0)
        simulation.metsr_lane_indices = {"A_1": 1}
        simulation._sumo_lane_to_carla_keys = {"A_1": {"1_-2"}}
        simulation._carla_waypoints_by_key = {"1_-2": [ahead]}
        simulation.metsr_road_cache = {
            ("A", 1): [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        }
        simulation.grp = SimpleNamespace(_sampling_resolution=2.0)
        simulation.map = SimpleNamespace(
            get_waypoint=lambda *args, **kwargs: None
        )

        with self.assertRaisesRegex(
            SimulationCreationError, "Insufficient road remains"
        ):
            simulation._forward_lane_anchor_trace(
                "A_1", current.transform.location, "A", 1
            )

    def test_stitch_anchors_a_different_first_road_departure_lane(self):
        simulation = object.__new__(CosimSimulation)
        current = self._route_waypoint(1, -1, 1.0, 0.0)
        first_anchor = self._route_waypoint(1, -2, 7.0, 7.0)
        second_anchor = self._route_waypoint(2, -2, 7.0, 17.0)
        simulation.map = SimpleNamespace(get_waypoint=lambda *args, **kwargs: current)
        simulation.scenic_to_metsr_map = {"1_-1": ["A_0"]}
        simulation.metsr_lane_indices = {
            "A_0": 0,
            "A_1": 1,
            "B_1": 1,
        }
        calls = []

        def trace(start, anchor, roads):
            calls.append((start.x, anchor.transform.location.x, tuple(roads)))
            return [(anchor, None)]

        def forward_trace(lane, start, road, required_lane_changes):
            self.assertEqual((lane, road, required_lane_changes), ("A_1", "A", 1))
            return first_anchor, trace(start, first_anchor, (road,))

        simulation._forward_lane_anchor_trace = forward_trace
        simulation._lane_anchor_waypoint = lambda lane, final=False: {
            "B_1": second_anchor,
        }[lane]
        simulation._trace_route_segment = trace
        simulation._validate_carla_route_trace = lambda *args, **kwargs: None

        result = simulation._stitch_carla_route(
            ["A", "B"], current.transform.location, ["A_1", "B_1"]
        )

        self.assertEqual([item[0] for item in result], [first_anchor, second_anchor])
        self.assertEqual(calls, [(0.0, 7.0, ("A",)), (7.0, 17.0, ("A", "B"))])

    def test_committed_route_refresh_waits_for_both_clients(self):
        class Vehicle:
            pass

        obj = Vehicle()
        obj.carlaActor = object()
        simulation = object.__new__(CosimSimulation)
        simulation.pending_route_refreshes = {obj: "B"}
        simulation.generate_carla_trajectory = lambda *args: self.fail(
            "route generation must wait for road alignment"
        )

        refreshed = simulation._refresh_committed_carla_route(
            obj,
            {"segmentId": "B", "transitionPending": False},
            "A",
            ["B", "C"],
        )

        self.assertFalse(refreshed)
        self.assertEqual(simulation.pending_route_refreshes[obj], "B")

    def test_committed_route_refresh_installs_path_exactly_once(self):
        class Vehicle:
            pass

        events = []
        path = [object(), object()]
        obj = Vehicle()
        obj.carlaActor = object()
        simulation = object.__new__(CosimSimulation)
        simulation.pending_route_refreshes = {obj: "B"}
        simulation.generate_carla_trajectory = lambda route, target: (
            events.append(("route", route, target)) or path
        )
        simulation.tm = SimpleNamespace(
            set_path=lambda actor, trajectory: events.append(
                ("path", actor, trajectory)
            ),
            auto_lane_change=lambda actor, enabled: events.append(
                ("lane_change", actor, enabled)
            ),
        )
        state = {"segmentId": "B", "transitionPending": False}

        first = simulation._refresh_committed_carla_route(
            obj, state, "B", ["B", "C", "D"]
        )
        second = simulation._refresh_committed_carla_route(
            obj, state, "B", ["B", "C", "D"]
        )

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertIs(obj.trajectory, path)
        self.assertNotIn(obj, simulation.pending_route_refreshes)
        self.assertEqual(
            events,
            [
                ("route", ["B", "C", "D"], obj),
                ("path", obj.carlaActor, path),
                ("lane_change", obj.carlaActor, False),
            ],
        )

    def test_remote_mapped_lane_distance_is_detected_before_teleport(self):
        simulation = object.__new__(CosimSimulation)
        simulation.metsr_road_cache = {
            ("A", 0): [(0.0, 0.0), (10.0, 0.0)]
        }

        distance = simulation._distance_to_mapped_road(
            (100.0, 100.0), ["A_0"], "A"
        )

        self.assertGreater(distance, 100.0)

    def test_projection_guard_translates_raw_sumo_lane_suffix(self):
        calls = []
        simulation = object.__new__(CosimSimulation)
        simulation.metsr_lane_indices = {"-29_6": 4}
        simulation.metsr_road_cache = {}
        simulation.metsr_client = SimpleNamespace(
            query_centerline=lambda road, lane_index, transform_coords: (
                calls.append((road, lane_index, transform_coords))
                or metsr_response(
                    "centerLine",
                    [{"centerline": [(0.0, 0.0), (10.0, 0.0)]}],
                )
            )
        )

        distance = simulation._distance_to_mapped_road(
            (5.0, 1.0), ["-29_6"], "-29"
        )

        self.assertEqual(distance, 1.0)
        self.assertEqual(calls, [("-29", 4, True)])

    def test_nearest_mapped_lane_returns_exact_compact_index(self):
        simulation = object.__new__(CosimSimulation)
        simulation.metsr_lane_indices = {"4_5": 3, "4_6": 4}
        simulation.metsr_road_cache = {
            ("4", 3): [(0.0, 3.5), (10.0, 3.5)],
            ("4", 4): [(0.0, 0.0), (10.0, 0.0)],
        }

        lane_index, distance = simulation._nearest_mapped_lane(
            (5.0, 0.5), ["4_5", "4_6"], "4"
        )

        self.assertEqual(lane_index, 4)
        self.assertEqual(distance, 0.5)


    def test_internal_mapped_lane_uses_connector_path_centerline(self):
        calls = []
        connector_id = "cont/6/5"
        simulation = object.__new__(CosimSimulation)
        simulation.metsr_internal_lane_connections = {
            ":332_0_4": [("6_4", "5_4")]
        }
        simulation.metsr_internal_edge_to_connector = {
            ":332_0": connector_id
        }
        simulation.metsr_connector_records = {
            connector_id: {
                "connectorId": connector_id,
                "sourceRoadId": "6",
                "targetRoadId": "5",
                "internalEdgeIds": [":332_0"],
                "paths": [
                    {
                        "connectorPathId": 4,
                        "sourceLaneId": "6_4",
                        "targetLaneId": "5_4",
                        "viaLaneIds": [":332_0_4"],
                        "internalEdgeIds": [":332_0"],
                    }
                ],
            }
        }
        simulation.metsr_connector_path_cache = {}
        simulation.metsr_connector_centerline_cache = {}
        centerlines = [[], [], [], [], [(0.0, 0.0), (10.0, 0.0)]]
        simulation.metsr_client = SimpleNamespace(
            query_centerline=lambda segment, lane_index, transform_coords: (
                calls.append((segment, lane_index, transform_coords))
                or metsr_response(
                    "centerLine",
                    [{"segmentId": connector_id, "centerlines": centerlines}],
                )
            )
        )

        resolved = simulation._nearest_mapped_connector_path_projection(
            (5.0, 1.0), [":332_0_4"]
        )

        self.assertEqual(resolved[:2], (connector_id, 4))
        self.assertEqual(resolved[2], 1.0)
        self.assertEqual(resolved[4]["sourceRoadId"], "6")
        self.assertEqual(resolved[4]["targetRoadId"], "5")
        self.assertEqual(calls, [(connector_id, -1, True)])
    def test_connector_spawn_uses_connector_path_id_not_lane_index(self):
        connector_id = "cont/6/5"
        initialize_calls = []
        route_calls = []

        class Position(tuple):
            def __new__(cls, x, y, z=0.0):
                return super().__new__(cls, (x, y, z))

            x = property(lambda self: self[0])
            y = property(lambda self: self[1])
            z = property(lambda self: self[2])

        class ParentOrientation:
            def __mul__(self, orientation):
                return SimpleNamespace(eulerAngles=(0.0, 0.0, 0.0))

        obj = SimpleNamespace(
            position=Position(5.0, 1.0),
            width=2.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            parentOrientation=ParentOrientation(),
        )
        origin_lane = SimpleNamespace(road=SimpleNamespace(id=333), id=5)
        origin_waypoint = SimpleNamespace(
            road_id=333, lane_id=5, lane_width=3.5
        )
        destination_waypoint = SimpleNamespace(road_id=100, lane_id=1)

        simulation = object.__new__(CosimSimulation)
        simulation.network_helper = SimpleNamespace(
            _nearest_lane=lambda target: origin_lane,
            scenic_to_metsr_map_lanes={
                "333_5": [":332_0_4"],
                "100_1": ["D_0"],
            },
            metsr_represented_roads={"5", "6", "D"},
        )
        simulation.map = SimpleNamespace(
            get_waypoint=lambda location, *args, **kwargs: (
                destination_waypoint if location.x > 50 else origin_waypoint
            )
        )
        simulation.spawn_points = [
            SimpleNamespace(location=carla.Location(x=100.0, y=0.0, z=0.0))
        ]
        simulation.metsr_lane_indices = {"D_0": 0}
        simulation.metsr_road_cache = {
            ("D", 0): [(90.0, 0.0), (110.0, 0.0)]
        }
        simulation.metsr_internal_lane_connections = {
            ":332_0_4": [("6_4", "5_4")]
        }
        simulation.metsr_internal_edge_to_connector = {
            ":332_0": connector_id
        }
        simulation.metsr_connector_records = {
            connector_id: {
                "connectorId": connector_id,
                "sourceRoadId": "6",
                "targetRoadId": "5",
                "internalEdgeIds": [":332_0"],
                "paths": [
                    {
                        "connectorPathId": 4,
                        "sourceLaneId": "6_4",
                        "targetLaneId": "5_4",
                        "viaLaneIds": [":332_0_4"],
                        "internalEdgeIds": [":332_0"],
                    }
                ],
            }
        }
        simulation.metsr_connector_path_cache = {}
        simulation.metsr_connector_centerline_cache = {
            (connector_id, 4): [(0.0, 0.0), (10.0, 0.0)]
        }
        simulation.carla_control_roads = {"5": True, "6": True}
        simulation.carla_control_segments = {"5", "6", ":332_0"}
        simulation.metsr_road_connectors = {
            "5": {connector_id},
            "6": {connector_id},
        }

        def initialize(**kwargs):
            initialize_calls.append(kwargs)
            return metsr_response(
                "initializeCoSimVeh",
                [
                    {
                        "vehicleId": 7,
                        "segmentId": connector_id,
                        "connectorPathId": 4,
                        "laneIndex": -1,
                    }
                ],
            )

        simulation.metsr_client = SimpleNamespace(
            query_route_between_roads=lambda source, target: metsr_response(
                "route", [{"roadIds": [source, target]}]
            ),
            initialize_cosim_vehicle=initialize,
            query_vehicle=lambda *args, **kwargs: metsr_response(
                "vehicle", [{
                    "x": 5.0, "y": 1.0,
                    "segmentId": connector_id,
                    "onConnector": True,
                    "sourceRoadId": "6",
                    "targetRoadId": "5",
                    "destinationRoadId": "D",
                }]
            ),
            update_vehicle_route=lambda vehicle_id, route, private_veh: (
                route_calls.append((vehicle_id, route, private_veh))
                # Native METS-R rejects a physical-road route while the
                # vehicle still occupies cont/6/5 (the seed-37 failure).
                or metsr_response("updateVehicleRoute", [
                    {"vehicleId": vehicle_id, "status": "error"}
                ])
            ),
            generate_trip_between_roads=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("a controlled connector must use co-sim placement")
            ),
        )
        simulation.getMetsrPrivateVehId = lambda target: 7

        with redirect_stdout(StringIO()):
            simulation.createObjectInMetsr(obj)

        self.assertEqual(obj.route, ["6", "5", "D"])
        self.assertEqual(len(initialize_calls), 1)
        self.assertEqual(initialize_calls[0]["segment_id"], connector_id)
        self.assertEqual(initialize_calls[0]["connector_path_id"], 4)
        self.assertNotIn("lane_index", initialize_calls[0])
        # initializeCoSimVeh supplies a native route to D through the
        # connector. Do not overwrite it with a route starting at road 5:
        # the vehicle's current segment is still cont/6/5.
        self.assertEqual(initialize_calls[0]["destination_road_id"], "D")
        self.assertEqual(route_calls, [])

    def _controlled_road_spawn_fixture(self):
        class ParentOrientation:
            def __mul__(self, orientation):
                return SimpleNamespace(eulerAngles=(0.0, 0.0, 0.0))

        obj = SimpleNamespace(
            position=SimpleNamespace(x=5.0, y=1.0, z=0.0),
            width=2.0, yaw=0.0, pitch=0.0, roll=0.0,
            parentOrientation=ParentOrientation(),
        )
        calls = {"initialize": [], "route": []}
        simulation = object.__new__(CosimSimulation)
        simulation.network_helper = SimpleNamespace(
            _nearest_lane=lambda target: SimpleNamespace(
                road=SimpleNamespace(id=1), id=1
            ),
            scenic_to_metsr_map_lanes={"1_1": ["A_0"], "2_1": ["D_0"]},
        )
        simulation._nearest_mapped_lane_projection = lambda *args: (0, 0.0, False)
        simulation._source_projection_error_limit = lambda *args: 3.0
        simulation.identify_nearest_road = lambda *args: "D"
        simulation.map = SimpleNamespace(
            get_waypoint=lambda *args: SimpleNamespace(road_id=2, lane_id=1)
        )
        simulation.spawn_points = [
            SimpleNamespace(location=carla.Location(x=100.0, y=0.0, z=0.0))
        ]
        simulation.carla_control_roads = {"A": True}
        simulation.getMetsrPrivateVehId = lambda target: 7
        simulation.metsr_client = SimpleNamespace(
            query_route_between_roads=lambda source, target: metsr_response(
                "routesBwRoads", [{"roadIds": [source, target]}]
            ),
            initialize_cosim_vehicle=lambda **kwargs: (
                calls["initialize"].append(kwargs) or metsr_response(
                    "initializeCoSimVeh", [{
                        "vehicleId": 7, "status": "ok",
                        "segmentId": "A", "laneIndex": 0,
                    }]
                )
            ),
            query_vehicle=lambda *args, **kwargs: metsr_response(
                "vehicle", [{"x": 5.0, "y": 1.0, "segmentId": "A"}]
            ),
            update_vehicle_route=lambda vehicle_id, route, private_veh: (
                calls["route"].append((vehicle_id, route, private_veh))
                or metsr_response("updateVehicleRoute", [
                    {"vehicleId": vehicle_id, "status": "error"}
                ])
            ),
        )
        return simulation, obj, calls

    def test_controlled_physical_road_spawn_retains_native_initialized_route(self):
        simulation, obj, calls = self._controlled_road_spawn_fixture()
        with redirect_stdout(StringIO()):
            simulation.createObjectInMetsr(obj, origin="A")
        self.assertEqual(obj.route, ["A", "D"])
        self.assertEqual(len(calls["initialize"]), 1)
        self.assertEqual(calls["initialize"][0]["destination_road_id"], "D")
        self.assertNotIn("routeRoadIds", calls["initialize"][0])
        self.assertEqual(calls["route"], [])

    def test_spawn_initialization_rejection_is_still_propagated(self):
        response = metsr_response(
            "initializeCoSimVeh", [{
                "vehicleId": 7, "status": "error",
                "errorCode": "INITIALIZATION_FAILED",
                "message": "No route from the matched road to the destination",
            }]
        )
        simulation, obj, calls = self._controlled_road_spawn_fixture()
        simulation.metsr_client.initialize_cosim_vehicle = lambda **kwargs: response
        with redirect_stdout(StringIO()), self.assertRaises(METSRControlError) as raised:
            simulation.createObjectInMetsr(obj, origin="A")
        self.assertIn("INITIALIZATION_FAILED", str(raised.exception))
        self.assertIs(raised.exception.response, response)
        self.assertIs(raised.exception.record, response["data"][0])
        self.assertEqual(calls["route"], [])

    def _native_road_spawn_fixture(self):
        generated = []

        class Position(tuple):
            def __new__(cls, x, y, z=0.0):
                return super().__new__(cls, (x, y, z))

            x = property(lambda self: self[0])
            y = property(lambda self: self[1])
            z = property(lambda self: self[2])

        obj = SimpleNamespace(position=Position(5.0, 1.0), width=2.0)
        origin_lane = SimpleNamespace(road=SimpleNamespace(id=333), id=5)
        origin_waypoint = SimpleNamespace(road_id=333, lane_id=5)
        destination_waypoint = SimpleNamespace(road_id=100, lane_id=1)

        simulation = object.__new__(CosimSimulation)
        simulation.network_helper = SimpleNamespace(
            _nearest_lane=lambda target: origin_lane,
            scenic_to_metsr_map_lanes={
                "333_5": ["A_0"],
                "100_1": ["D_0"],
            },
            metsr_represented_roads={"A", "D"},
        )
        simulation.map = SimpleNamespace(
            get_waypoint=lambda location, *args, **kwargs: (
                destination_waypoint if location.x > 50 else origin_waypoint
            )
        )
        simulation.spawn_points = [
            SimpleNamespace(location=carla.Location(x=100.0, y=0.0, z=0.0))
        ]
        simulation.metsr_lane_indices = {"A_0": 0, "D_0": 0}
        simulation.metsr_road_cache = {
            ("A", 0): [(0.0, 0.0), (10.0, 0.0)],
            ("D", 0): [(90.0, 0.0), (110.0, 0.0)],
        }
        simulation.metsr_internal_lane_connections = {}
        simulation.metsr_internal_edge_to_connector = {}
        simulation.metsr_connector_records = {}
        simulation.metsr_connector_path_cache = {}
        simulation.metsr_connector_centerline_cache = {}
        simulation.metsr_road_connectors = {}
        simulation.carla_control_roads = {}
        simulation.carla_control_segments = set()

        def forbidden(*args, **kwargs):
            raise AssertionError("native Scenic spawns must use only generateTripsByRoad")

        simulation.metsr_client = SimpleNamespace(
            query_route_between_roads=lambda source, target: metsr_response(
                "route", [{"roadIds": [source, target]}]
            ),
            generate_trip_between_roads=lambda **kwargs: (
                generated.append(kwargs) or metsr_response("generateTripsByRoad")
            ),
            initialize_cosim_vehicle=forbidden,
            teleport_trace_replay_vehicle=forbidden,
            query_vehicle=forbidden,
            update_vehicle_route=forbidden,
        )
        simulation.getMetsrPrivateVehId = lambda target: 7
        return simulation, obj, generated

    def test_native_spawn_uses_trip_queue_without_digital_twin_teleport(self):
        simulation, obj, generated = self._native_road_spawn_fixture()
        with redirect_stdout(StringIO()):
            simulation.createObjectInMetsr(obj)

        self.assertEqual(obj.route, ["A", "D"])
        self.assertEqual(
            generated, [{"vehID": 7, "origin": "A", "destination": "D"}]
        )

    def _town06_short_lane_spawn_fixture(self):
        simulation, obj, generated = self._native_road_spawn_fixture()
        # Reported demo2 seed-34 spawn, and Town06.net.xml lane -64_4 with
        # netOffset removed. SUMO truncates this driving lane to four metres.
        obj.position = type(obj.position)(-5.303203, -378.247755)
        obj.width = 2.1
        simulation.network_helper._nearest_lane = lambda target: SimpleNamespace(
            road=SimpleNamespace(id=64), id=-3
        )
        simulation.network_helper.scenic_to_metsr_map_lanes = {
            "64_-3": ["-64_4"], "100_1": ["D_0"],
        }
        simulation.network_helper.metsr_represented_roads = {"-64", "D"}
        simulation.metsr_lane_indices = {"-64_4": 2, "D_0": 0}
        simulation.metsr_road_cache[("-64", 2)] = [
            (-3.19, -360.49), (-3.24, -361.51), (-3.31, -362.58),
            (-3.39, -363.64), (-3.47, -364.48),
        ]
        simulation.map.get_waypoint = lambda location, *args, **kwargs: (
            SimpleNamespace(road_id=100, lane_id=1)
            if location.x > 50 else
            SimpleNamespace(road_id=64, lane_id=-3, lane_width=3.5)
        )
        return simulation, obj, generated

    def test_native_spawn_beyond_short_lane_endpoint_uses_road_queue(self):
        simulation, obj, generated = self._town06_short_lane_spawn_fixture()
        with redirect_stdout(StringIO()):
            simulation.createObjectInMetsr(obj)

        self.assertEqual(obj.route, ["-64", "D"])
        self.assertEqual(
            generated, [{"vehID": 7, "origin": "-64", "destination": "D"}]
        )

    def test_controlled_spawn_beyond_short_lane_endpoint_is_rejected(self):
        simulation, obj, generated = self._town06_short_lane_spawn_fixture()
        simulation.carla_control_roads = {"-64": True}
        with self.assertRaisesRegex(
            SimulationCreationError,
            r"13\.89 m from mapped METS-R segment -64; refusing a remote projection",
        ):
            simulation.createObjectInMetsr(obj)

        self.assertEqual(generated, [])

    def test_pending_native_spawn_is_not_marked_complete(self):
        class Vehicle:
            pass

        obj = Vehicle()
        obj.name = "queued-native"
        obj.spawn_guard = 0
        obj.carla_actor_flag = False
        simulation = object.__new__(CosimSimulation)
        simulation.objects = [SimpleNamespace(name="ego"), obj]
        simulation.bubble_roads = []
        simulation.sim_ticks_per_carla = 1
        simulation.count = 4
        simulation.completed_route = {}
        simulation.pending_road_entries = {}
        simulation.pv_id_map = {obj: 7}
        simulation.admitted_queue_vehicles = {}
        simulation.carla_control_roads = {}
        simulation.carla_control_segments = set()
        simulation.metsr_connector_records = {}
        simulation.metsr_road_connectors = {}
        simulation.metsr_internal_edge_to_connector = {}
        simulation.road_pop_density = {}
        simulation.carla_actors = []
        simulation.metsr_actors = [obj]
        record = {
            "vehicleId": 7,
            "onRoad": False,
            "queuedRoadId": "A",
            "originRoadId": "A",
            "destinationRoadId": "D",
        }

        simulation.update_bubble_objects([], [], vehicle_data={obj: record})

        self.assertNotIn(obj, simulation.completed_route)
        self.assertEqual(simulation.carla_actors, [])
        self.assertEqual(simulation.metsr_actors, [obj])

    def test_polyline_projection_distinguishes_terminal_from_interior(self):
        polyline = [(0.0, 0.0), (0.0, 0.0), (10.0, 0.0)]

        terminal_distance, terminal = (
            CosimSimulation._point_to_polyline_projection((-4.0, 1.0), polyline)
        )
        interior_distance, interior_terminal = (
            CosimSimulation._point_to_polyline_projection((5.0, 4.0), polyline)
        )

        self.assertAlmostEqual(terminal_distance, math.sqrt(17.0))
        self.assertTrue(terminal)
        self.assertEqual(interior_distance, 4.0)
        self.assertFalse(interior_terminal)

    def test_terminal_allowance_requires_matching_carla_lane(self):
        simulation = object.__new__(CosimSimulation)
        waypoint = SimpleNamespace(road_id=52, lane_id=-6, lane_width=3.5)
        simulation.map = SimpleNamespace(
            get_waypoint=lambda *args, **kwargs: waypoint
        )
        obj = SimpleNamespace(
            position=SimpleNamespace(x=-32.487, y=18.810, z=0.0)
        )

        self.assertEqual(
            simulation._source_projection_error_limit(
                obj, "52_-6", 3.0, at_terminal=True
            ),
            6.5,
        )
        self.assertEqual(
            simulation._source_projection_error_limit(
                obj, "52_-5", 3.0, at_terminal=True
            ),
            3.0,
        )
        self.assertEqual(
            simulation._source_projection_error_limit(
                obj, "52_-6", 3.0, at_terminal=False
            ),
            3.0,
        )

    def test_authoritative_lane_map_rejects_an_unknown_lane(self):
        simulation = object.__new__(CosimSimulation)
        simulation.metsr_lane_indices = {}
        simulation.network_helper = SimpleNamespace(
            metsr_represented_roads={"-29"}
        )

        self.assertEqual(
            simulation._mapped_lane_query("-29_6"), ("-29", None)
        )

    def test_road_only_mapping_preserves_underscores(self):
        simulation = object.__new__(CosimSimulation)
        simulation.metsr_lane_indices = {}
        simulation.network_helper = SimpleNamespace(
            metsr_represented_roads={":1034_0"}
        )

        self.assertEqual(
            simulation._mapped_lane_query(":1034_0"), (":1034_0", -1)
        )

    def test_malformed_centerline_response_fails_closed(self):
        simulation = object.__new__(CosimSimulation)
        simulation.metsr_lane_indices = {"-29_6": 4}
        simulation.metsr_road_cache = {}
        simulation.metsr_client = SimpleNamespace(
            query_centerline=lambda *args, **kwargs: metsr_response("centerLine")
        )

        distance = simulation._distance_to_mapped_road(
            (5.0, 1.0), ["-29_6"], "-29"
        )

        self.assertTrue(math.isinf(distance))

    def test_client_sync_diagnostic_uses_200_step_cadence(self):
        due = CosimSimulation._client_sync_diagnostic_due

        self.assertTrue(due(0, True))
        self.assertFalse(due(1, True))
        self.assertFalse(due(199, True))
        self.assertTrue(due(200, True))
        self.assertFalse(due(200, False))

    def test_road_hint_does_not_infer_from_retired_transition_state(self):
        simulation = object.__new__(CosimSimulation)
        obj = SimpleNamespace(route=["A", "B", "C"])
        state = {
            "transitionPending": True,
            "segmentId": "B",
        }

        self.assertIsNone(simulation._observed_road_hint(obj, state, "C"))
        self.assertIsNone(simulation._observed_road_hint(obj, state, "A"))

    def test_road_observation_classifier_including_current(self):
        classify = CosimSimulation._classify_road_observation
        route = ["A", "B", "C", "D"]

        self.assertEqual(classify("B", None, route), ROAD_OBSERVATION_UNKNOWN)
        self.assertEqual(classify("B", "B", route), ROAD_OBSERVATION_SAME)
        self.assertEqual(
            classify("B", "A", route), ROAD_OBSERVATION_PREDECESSOR
        )
        self.assertEqual(
            classify("B", "C", route), ROAD_OBSERVATION_DIRECT_SUCCESSOR
        )
        self.assertEqual(classify("B", "D", route), ROAD_OBSERVATION_ONE_SKIP)
        self.assertEqual(
            classify("B", "X", route), ROAD_OBSERVATION_UNSUPPORTED
        )

    def test_road_observation_classifier_rejects_route_without_current(self):
        classify = CosimSimulation._classify_road_observation

        self.assertEqual(
            classify("A", "B", ["B", "C"]),
            ROAD_OBSERVATION_UNSUPPORTED,
        )
        self.assertEqual(
            classify("A", "C", ["B", "C"]), ROAD_OBSERVATION_UNSUPPORTED
        )

    def test_pending_source_can_be_supplied_explicitly(self):
        self.assertEqual(
            CosimSimulation._classify_road_observation(
                "B", "A", ["B", "C"], predecessor_road_id="A"
            ),
            ROAD_OBSERVATION_PREDECESSOR,
        )

    def test_forward_route_match_precedes_stale_predecessor_hint(self):
        self.assertEqual(
            CosimSimulation._classify_road_observation(
                "A", "B", ["A", "B", "C"], predecessor_road_id="B"
            ),
            ROAD_OBSERVATION_DIRECT_SUCCESSOR,
        )

    @staticmethod
    def _sync_guard_fixture(
        state,
        observed_road,
        live_route,
        *,
        observed_lane=2,
        object_route=None,
        pending_entry=None,
        transition_result=None,
        transition_error=None,
        teleport_result=None,
        lane_indices=None,
        lane_connections=None,
        generated_trajectory=None,
        cosim_records=None,
        controlled_roads=None,
    ):
        class Vehicle:
            pass

        class Actor:
            def __init__(self):
                self.id = 99
                self.type_id = "vehicle.test"
                self.location = carla.Location(x=10.0, y=-2.0, z=0.0)
                self.transform = carla.Transform(
                    self.location, carla.Rotation(yaw=0.0)
                )
                self.bounding_box = SimpleNamespace(
                    extent=SimpleNamespace(x=1.0, y=0.5, z=0.75)
                )
                self.destroyed = False

            def get_location(self):
                return self.location

            def get_transform(self):
                return self.transform

            def get_velocity(self):
                return SimpleNamespace(x=0.0, y=0.0, z=0.0)

            def set_target_velocity(self, velocity):
                pass

            def apply_control(self, control):
                pass

            def enable_constant_velocity(self, velocity):
                pass

            def disable_constant_velocity(self):
                pass

            def destroy(self):
                self.destroyed = True
                return True

        obj = Vehicle()
        obj.name = "guarded_car"
        obj.route = list(object_route or live_route)
        obj.carlaActor = Actor()
        obj.active_autopilot = False
        obj.autopilot_action = False
        obj.carla_actor_flag = True
        obj.spawn_guard = 0
        calls = {
            "teleport": [],
            "generate": [],
            "path": [],
            "lane_change": [],
            "transform": [],
            "generate_start": [],
            "route_query": [],
        }

        transition_reply_used = False

        def teleport_cosim_vehicle(*args, **kwargs):
            nonlocal transition_reply_used
            calls["teleport"].append((args, kwargs))
            if not transition_reply_used and transition_error is not None:
                transition_reply_used = True
                raise transition_error
            if not transition_reply_used and isinstance(transition_result, dict):
                transition_reply_used = True
                records = transition_result.get("data", ())
            elif isinstance(teleport_result, dict):
                records = teleport_result.get("data", ())
            else:
                record = dict(state)
                segment_hint = kwargs.get("segment_id")
                if isinstance(segment_hint, (list, tuple)):
                    segment_hint = segment_hint[0] if segment_hint else None
                if segment_hint not in (None, ""):
                    record["segmentId"] = str(segment_hint)
                lane_hint = kwargs.get("lane_index")
                if isinstance(lane_hint, (list, tuple)):
                    lane_hint = lane_hint[0] if lane_hint else None
                if lane_hint is not None:
                    record["laneIndex"] = int(lane_hint)
                records = [record]
            records = [
                dict(
                    record,
                    vehicleId=record.get("vehicleId", args[0]),
                    status=record.get("status", "ok"),
                )
                for record in records
            ]
            return metsr_response("teleportCoSimVeh", records)

        def query_route_between_roads(origin, destination):
            calls["route_query"].append((origin, destination))
            route = [str(road) for road in live_route]
            if not route or route[0] != str(origin):
                route.insert(0, str(origin))
            if route[-1] != str(destination):
                route.append(str(destination))
            return metsr_response("routesBwRoads", [{"roadIds": route}])

        trajectory = generated_trajectory or [SimpleNamespace(name="recovery-path")]

        def generate_carla_trajectory(
            route, obj, required_first_lane_id=None, *, target_start=None
        ):
            calls["generate"].append(
                (list(route), obj, required_first_lane_id)
            )
            calls["generate_start"].append(target_start)
            return trajectory

        simulation = object.__new__(CosimSimulation)
        simulation.carla_actors = [obj]
        simulation.metsr_actors = []
        simulation._carla_authoritative_segments = {}
        simulation.pending_road_entries = (
            {obj: pending_entry} if pending_entry is not None else {}
        )
        simulation.pending_route_refreshes = {}
        simulation.pending_lane_reconciliation_verifications = {}
        simulation.same_road_departure_lane_verifications = {}
        simulation.road_entry_holds = {}
        simulation.road_entry_hold_diagnostics = {}
        simulation.completed_route = {}
        if controlled_roads is None:
            controlled_roads = set(object_route or live_route)
            if observed_road is not None:
                controlled_roads.add(str(observed_road))
        simulation.carla_control_roads = {
            str(road_id): True for road_id in controlled_roads
        }
        simulation.carla_control_segments = set(simulation.carla_control_roads)
        simulation.metsr_internal_edge_to_connector = {}
        simulation.metsr_connector_records = {}
        simulation.metsr_road_connectors = {}
        simulation.count = 1
        simulation.metsr_lane_indices = dict(lane_indices or {})
        simulation.metsr_lane_connections = dict(lane_connections or {})
        simulation.metsr_road_cache = {}
        simulation.carla_world = SimpleNamespace(get_actors=lambda: [obj.carlaActor])
        simulation._collect_metsr_vehicle_data = lambda objects: {obj: state}
        simulation.getMetsrPrivateVehId = lambda target: 7
        simulation._mapped_carla_observation = lambda location: (
            ("B", 2)
            if abs(float(location.y) + 5.0) < 1e-6
            else (observed_road, observed_lane)
        )
        if cosim_records is None:
            cosim_records = [
                {"vehicleId": 7, "isPrivate": True, "routeRoadIds": list(live_route)}
            ]
        confirmation_record = {
            "segmentId": "B",
            "laneIndex": 2,
            "transitionPending": False,
        }
        simulation.metsr_client = SimpleNamespace(
            query_cosim_vehicle=lambda: metsr_response(
                "coSimVehicle", cosim_records
            ),
            query_vehicle=lambda *args, **kwargs: metsr_response(
                "vehicle", [dict(confirmation_record)]
            ),
            query_route_between_roads=query_route_between_roads,
            teleport_cosim_vehicle=teleport_cosim_vehicle,
        )
        simulation.generate_carla_trajectory = generate_carla_trajectory
        old_transform = obj.carlaActor.get_transform()
        target_transform = carla.Transform(
            carla.Location(x=10.0, y=-5.0, z=0.0),
            carla.Rotation(yaw=0.0),
        )
        simulation._pending_lane_reconciliation_target = (
            lambda target, road, lane: ("B_8", old_transform, target_transform)
        )
        simulation._assert_pending_lane_reconciliation_clear = (
            lambda actor, old, target: None
        )

        def apply_transform(actor, transform, context):
            calls["transform"].append((actor, transform, context))
            actor.transform = transform
            actor.location = transform.location

        simulation._apply_carla_transform_without_tick = apply_transform
        simulation.metsr_trajectory_to_carla = lambda target: (_ for _ in ()).throw(
            AssertionError("the constrained recovery path must not be overwritten")
        )
        simulation.tm = SimpleNamespace(
            set_path=lambda actor, path: calls["path"].append((actor, path)),
            auto_lane_change=lambda actor, enabled: calls["lane_change"].append(
                (actor, enabled)
            ),
        )
        simulation.check_client_synchronization = lambda: None
        return simulation, obj, calls

    def test_direct_successor_asserts_observed_road_and_lane(self):
        state = {
            "segmentId": "A",
            "destinationRoadId": "Z",
            "transitionPending": False,
            "laneIndex": 1,
        }
        simulation, _, calls = self._sync_guard_fixture(
            state,
            "B",
            ["A", "B", "C"],
            observed_lane=3,
            lane_indices={"A_5": 1, "B_7": 3},
            lane_connections={("A", "B"): [("A_5", "B_7")]},
        )

        simulation.synchronize_clients()

        self.assertEqual(len(calls["teleport"]), 1)
        self.assertEqual(calls["teleport"][0][1]["segment_id"], "B")
        self.assertEqual(calls["teleport"][0][1]["lane_index"], 3)

    def _legacy_same_road_route_preparation_reconciles_adjacent_departure_lane(self):
        state = {
            "segmentId": "-52",
            "destinationRoadId": "20",
            "transitionPending": False,
            "laneIndex": 3,
        }
        route = ["-52", "-59", "69"]
        recovery_path = [SimpleNamespace(name="departure-lane-4")]
        simulation, obj, calls = self._sync_guard_fixture(
            state,
            "-52",
            route,
            observed_lane=3,
            lane_indices={
                "-52_5": 3,
                "-52_6": 4,
                "-59_2": 0,
                "69_2": 0,
            },
            lane_connections={
                ("-52", "-59"): [("-52_6", "-59_2")],
                ("-59", "69"): [("-59_2", "69_2")],
            },
            generated_trajectory=recovery_path,
        )
        simulation.carla_control_roads = {"-52": True}
        simulation._candidate_metsr_lane_chains = (
            lambda requested, start, required_initial_lane=None: [
                ["-52_6", "-59_2", "69_2"]
            ]
        )
        old_transform = obj.carlaActor.get_transform()
        target_transform = carla.Transform(
            carla.Location(x=10.0, y=-5.0, z=0.0),
            carla.Rotation(yaw=0.0),
        )
        simulation._pending_lane_reconciliation_target = (
            lambda target, road, lane: (
                "-52_6", old_transform, target_transform
            )
        )
        simulation._mapped_carla_observation = lambda location: (
            ("-52", 4)
            if abs(float(location.y) + 5.0) < 1e-6
            else ("-52", 3)
        )
        simulation.metsr_client.teleport_cosim_vehicle = lambda *args, **kwargs: (
            calls["teleport"].append((args, kwargs))
            or metsr_response(
                "teleportCoSimVeh",
                [
                    {
                        "segmentId": "-52",
                        "laneIndex": 4,
                        "laneSynchronized": True,
                        "transitionPending": False,
                    }
                ],
            )
        )
        simulation.metsr_client.query_vehicle = lambda *args, **kwargs: metsr_response(
            "vehicle",
            [
                {"segmentId": "-52", "laneIndex": 4, "transitionPending": False}
            ],
        )

        simulation.synchronize_clients()

        self.assertEqual(calls["generate"], [(route, obj, 4)])
        self.assertEqual(calls["path"], [(obj.carlaActor, recovery_path)])
        self.assertEqual(len(calls["transform"]), 1)
        self.assertEqual(calls["teleport"], [])
        self.assertIn(obj, simulation.road_entry_holds)
        self.assertEqual(
            simulation.pending_road_entries[obj]["target"], "-52"
        )
        verification = simulation.same_road_departure_lane_verifications[obj]
        self.assertEqual(
            (verification["roadID"], verification["sourceLaneID"], verification["laneID"]),
            ("-52", 3, 4),
        )

        simulation.synchronize_clients()

        self.assertEqual(len(calls["teleport"]), 1)
        self.assertEqual(calls["teleport"][0][1]["segment_id"], "-52")
        self.assertEqual(calls["teleport"][0][1]["lane_index"], 4)
        self.assertEqual(calls["teleport"][0][1]["speed"], 0.0)
        self.assertNotIn(obj, simulation.road_entry_holds)
        self.assertNotIn(obj, simulation.pending_road_entries)
        self.assertNotIn(obj, simulation.same_road_departure_lane_verifications)

    def _legacy_same_road_route_preparation_retries_occupied_swept_corridor(self):
        state = {
            "segmentId": "-52",
            "destinationRoadId": "20",
            "transitionPending": False,
            "laneIndex": 3,
        }
        route = ["-52", "-59", "69"]
        recovery_path = [SimpleNamespace(name="departure-lane-4")]
        simulation, obj, calls = self._sync_guard_fixture(
            state,
            "-52",
            route,
            observed_lane=3,
            lane_indices={
                "-52_5": 3,
                "-52_6": 4,
                "-59_2": 0,
                "69_2": 0,
            },
            lane_connections={
                ("-52", "-59"): [("-52_6", "-59_2")],
                ("-59", "69"): [("-59_2", "69_2")],
            },
            generated_trajectory=recovery_path,
        )
        simulation.carla_control_roads = {"-52": True}
        simulation._candidate_metsr_lane_chains = (
            lambda requested, start, required_initial_lane=None: [
                ["-52_6", "-59_2", "69_2"]
            ]
        )
        old_transform = obj.carlaActor.get_transform()
        target_transform = carla.Transform(
            carla.Location(x=10.0, y=-5.0, z=0.0),
            carla.Rotation(yaw=0.0),
        )
        simulation._pending_lane_reconciliation_target = (
            lambda target, road, lane: (
                "-52_6", old_transform, target_transform
            )
        )
        simulation._mapped_carla_observation = lambda location: (
            ("-52", 4)
            if abs(float(location.y) + 5.0) < 1e-6
            else ("-52", 3)
        )
        simulation.metsr_client.teleport_cosim_vehicle = lambda *args, **kwargs: (
            calls["teleport"].append((args, kwargs))
            or metsr_response(
                "teleportCoSimVeh",
                [
                    {
                        "segmentId": "-52",
                        "laneIndex": 4,
                        "laneSynchronized": True,
                        "transitionPending": False,
                    }
                ],
            )
        )
        simulation.metsr_client.query_vehicle = lambda *args, **kwargs: metsr_response(
            "vehicle",
            [
                {"segmentId": "-52", "laneIndex": 4, "transitionPending": False}
            ],
        )
        blocker = SimpleNamespace(
            id=100,
            type_id="vehicle.blocker",
            bounding_box=SimpleNamespace(
                extent=SimpleNamespace(x=1.0, y=0.5, z=0.75)
            ),
            get_location=lambda: carla.Location(x=10.0, y=-3.5, z=0.0),
        )
        actors = [obj.carlaActor, blocker]
        simulation.carla_world = SimpleNamespace(get_actors=lambda: list(actors))
        del simulation._assert_pending_lane_reconciliation_clear

        simulation.synchronize_clients()

        self.assertIn(obj, simulation.road_entry_holds)
        self.assertEqual(calls["generate"], [])
        self.assertEqual(calls["path"], [])
        self.assertEqual(calls["transform"], [])
        self.assertNotIn(obj, simulation.pending_road_entries)
        self.assertNotIn(obj, simulation.same_road_departure_lane_verifications)

        actors.remove(blocker)
        simulation.synchronize_clients()

        self.assertEqual(calls["generate"], [(route, obj, 4)])
        self.assertEqual(calls["path"], [(obj.carlaActor, recovery_path)])
        self.assertEqual(len(calls["transform"]), 1)
        self.assertIn(obj, simulation.road_entry_holds)
        self.assertEqual(simulation.pending_road_entries[obj]["target"], "-52")
        self.assertIn(obj, simulation.same_road_departure_lane_verifications)

        simulation.synchronize_clients()

        self.assertEqual(len(calls["teleport"]), 1)
        self.assertNotIn(obj, simulation.road_entry_holds)
        self.assertNotIn(obj, simulation.pending_road_entries)
        self.assertNotIn(obj, simulation.same_road_departure_lane_verifications)

    def test_predecessor_boundary_never_ordinary_teleports(self):
        state = {
            "segmentId": "A",
            "destinationRoadId": "Z",
            "transitionPending": False,
        }
        simulation, _, calls = self._sync_guard_fixture(
            state,
            "P",
            ["A", "B"],
            object_route=["P", "A", "B"],
        )

        simulation.metsr_lane_connections = {("P", "A"): [("P_0", "A_0")]}
        simulation.synchronize_clients()

        self.assertEqual(len(calls["teleport"]), 1)
        self.assertEqual(calls["teleport"][0][1]["segment_id"], "A")
        self.assertIsNone(calls["teleport"][0][1]["lane_index"])

    def _legacy_direct_successor_without_lane_topology_fails_closed(self):
        state = {
            "segmentId": "A",
            "destinationRoadId": "Z",
            "transitionPending": False,
            "laneIndex": 1,
        }
        simulation, _, calls = self._sync_guard_fixture(
            state, "B", ["A", "B", "C"], observed_lane=3
        )

        with self.assertRaisesRegex(
            SimulationCreationError, "direct_successor_without_lane_topology"
        ):
            simulation.synchronize_clients()

        self.assertEqual(calls["teleport"], [])

    def _legacy_ego_waits_until_target_road_is_under_carla_control(self):
        state = {
            "segmentId": "A",
            "destinationRoadId": "Z",
            "transitionPending": False,
            "laneIndex": 1,
        }
        simulation, obj, calls = self._sync_guard_fixture(
            state,
            "B",
            ["A", "B", "C"],
            observed_lane=3,
            lane_indices={"A_5": 1, "B_7": 3},
            lane_connections={("A", "B"): [("A_5", "B_7")]},
        )
        simulation.ego = obj
        simulation.carla_control_roads = {}

        simulation.synchronize_clients()

        self.assertEqual(calls["teleport"], [])
        self.assertIn(obj, simulation.road_entry_holds)
        self.assertEqual(simulation.pending_road_entries[obj]["target"], "B")

        simulation.carla_control_roads["B"] = True
        simulation.synchronize_clients()
        self.assertEqual(len(calls["teleport"]), 2)

    def _legacy_in_bubble_npc_waits_until_target_road_is_under_carla_control(self):
        state = {
            "segmentId": "A",
            "destinationRoadId": "Z",
            "transitionPending": False,
            "laneIndex": 1,
        }
        simulation, obj, calls = self._sync_guard_fixture(
            state,
            "B",
            ["A", "B", "C"],
            observed_lane=3,
            lane_indices={"A_5": 1, "B_7": 3},
            lane_connections={("A", "B"): [("A_5", "B_7")]},
        )
        simulation.carla_control_roads = {}
        simulation.active_bubble_metsr_roads = {"B"}

        simulation.synchronize_clients()

        self.assertEqual(calls["teleport"], [])
        self.assertIn(obj, simulation.road_entry_holds)
        self.assertEqual(simulation.pending_road_entries[obj]["target"], "B")

        simulation.carla_control_roads["B"] = True
        simulation.synchronize_clients()
        self.assertEqual(len(calls["teleport"]), 2)

    def _legacy_outside_bubble_npc_can_hand_back_to_native_target(self):
        state = {
            "segmentId": "A",
            "destinationRoadId": "Z",
            "transitionPending": False,
            "laneIndex": 1,
        }
        simulation, obj, calls = self._sync_guard_fixture(
            state,
            "B",
            ["A", "B", "C"],
            observed_lane=3,
            lane_indices={"A_5": 1, "B_7": 3},
            lane_connections={("A", "B"): [("A_5", "B_7")]},
        )
        simulation.carla_control_roads = {}
        simulation.active_bubble_metsr_roads = {"C"}

        simulation.synchronize_clients()

        self.assertEqual(len(calls["teleport"]), 2)
        self.assertNotIn(obj, simulation.road_entry_holds)

    def _legacy_retry_rejection_does_not_teleport_mismatched_target_pose(self):
        state = {
            "segmentId": "A",
            "destinationRoadId": "Z",
            "transitionPending": False,
            "laneIndex": 1,
        }
        error = METSRControlError(
            "blocked", record={"segmentId": "B", "retryable": True}
        )
        simulation, _, calls = self._sync_guard_fixture(
            state,
            "B",
            ["A", "B", "C"],
            observed_lane=4,
            pending_entry={"source": "A", "target": "B"},
            transition_error=error,
            lane_indices={"A_5": 1, "B_7": 4},
            lane_connections={("A", "B"): [("A_5", "B_7")]},
        )

        simulation.synchronize_clients()

        self.assertEqual(calls["teleport"][0][1]["lane_index"], 4)

    def _legacy_direct_lane_mismatch_installs_constrained_pending_path(self):
        state = {
            "segmentId": "A",
            "destinationRoadId": "Z",
            "transitionPending": False,
            "laneIndex": 1,
        }
        recovery_path = [SimpleNamespace(name="authoritative-lane-2")]
        simulation, obj, calls = self._sync_guard_fixture(
            state,
            "B",
            ["A", "B", "C"],
            observed_lane=3,
            transition_result={
                "data": [
                    {
                        "segmentId": "B",
                        "laneIndex": 2,
                        "transitionTargetRoadId": "B",
                        "transitionPending": True,
                    }
                ]
            },
            teleport_result={
                "data": [
                    {
                        "segmentId": "B",
                        "laneIndex": 2,
                        "transitionPending": False,
                        "transitionCommitted": True,
                    }
                ]
            },
            lane_indices={"A_5": 1, "B_8": 2, "B_7": 3},
            lane_connections={("A", "B"): [("A_5", "B_8")]},
            generated_trajectory=recovery_path,
        )

        simulation.synchronize_clients()

        self.assertIsNone(calls["teleport"][0][1]["lane_index"])
        self.assertEqual(calls["generate"], [(["B", "C"], obj, 2)])
        self.assertEqual(calls["path"], [(obj.carlaActor, recovery_path)])
        self.assertEqual(calls["lane_change"], [(obj.carlaActor, False)])
        self.assertEqual(len(calls["transform"]), 1)
        self.assertEqual(len(calls["teleport"]), 1)
        self.assertEqual(simulation.pending_route_refreshes, {})
        self.assertIn(obj, simulation.road_entry_holds)
        self.assertIn(obj, simulation.pending_road_entries)
        self.assertIn(obj, simulation.pending_lane_reconciliation_verifications)
        self.assertIs(obj.trajectory, recovery_path)
        self.assertEqual((calls["generate_start"][0].x, calls["generate_start"][0].y), (10.0, -5.0))

        state.update(
            {
                "segmentId": "B",
                "transitionPending": True,
                "transitionTargetRoadId": "B",
            }
        )
        simulation.synchronize_clients()

        self.assertEqual(len(calls["teleport"]), 2)
        args, kwargs = calls["teleport"][1]
        self.assertEqual(args[1:3], (10.0, 5.0))
        self.assertEqual(kwargs["speed"], 0.0)
        self.assertIsNone(kwargs["segment_id"])
        self.assertIsNone(kwargs["lane_index"])
        self.assertNotIn(obj, simulation.road_entry_holds)
        self.assertNotIn(obj, simulation.pending_road_entries)
        self.assertNotIn(obj, simulation.pending_lane_reconciliation_verifications)

    def test_pending_lane_reconciliation_target_is_exact_parallel_lane(self):
        simulation = object.__new__(CosimSimulation)
        simulation.metsr_lane_indices = {"B_8": 2}
        simulation.metsr_road_cache = {}
        simulation._sumo_lane_to_carla_keys = {"B_8": {"2_-2"}}
        simulation.scenic_to_metsr_map = {"2_-2": ["B_8"]}
        simulation._index_carla_waypoints = lambda: None
        simulation._query_mapped_centerline = lambda road, lane: [
            (0.0, 5.0, 0.0),
            (20.0, 5.0, 0.0),
        ]
        actor = SimpleNamespace(
            get_location=lambda: carla.Location(x=10.0, y=-2.0, z=0.0),
            get_transform=lambda: carla.Transform(
                carla.Location(x=10.0, y=-2.0, z=0.0),
                carla.Rotation(yaw=0.0),
            ),
        )
        obj = SimpleNamespace(carlaActor=actor)

        def waypoint(*, x=10.0, y=-5.0, yaw=0.0, junction=False, mapped=True):
            candidate = SimpleNamespace(
                road_id=2,
                lane_id=-2,
                is_junction=junction,
                transform=carla.Transform(
                    carla.Location(x=x, y=y, z=0.0),
                    carla.Rotation(yaw=yaw),
                ),
            )
            simulation._carla_waypoints_by_key = {"2_-2": [candidate]}
            simulation.scenic_to_metsr_map = {
                "2_-2": ["B_8" if mapped else "B_9"]
            }
            simulation.map = SimpleNamespace(get_waypoint=lambda *args, **kwargs: candidate)
            return candidate

        expected_waypoint = waypoint()
        raw, old, target = simulation._pending_lane_reconciliation_target(
            obj, "B", 2
        )
        self.assertEqual(raw, "B_8")
        self.assertEqual((target.location.x, target.location.y), (10.0, -5.0))
        self.assertEqual(target.rotation.yaw, expected_waypoint.transform.rotation.yaw)
        self.assertEqual((old.location.x, old.location.y), (10.0, -2.0))

        for label, candidate in (
            ("junction", dict(junction=True)),
            ("wrong raw mapping", dict(mapped=False)),
            ("too far", dict(y=-18.0)),
            ("wrong heading", dict(yaw=31.0)),
            ("wrong progress", dict(x=14.0)),
        ):
            with self.subTest(label=label):
                waypoint(**candidate)
                with self.assertRaisesRegex(
                    SimulationCreationError, "No safe parallel CARLA waypoint"
                ):
                    simulation._pending_lane_reconciliation_target(obj, "B", 2)

    def test_pending_lane_reconciliation_rejects_swept_collision(self):
        simulation = object.__new__(CosimSimulation)
        actor = SimpleNamespace(
            id=1,
            type_id="vehicle.ego",
            bounding_box=SimpleNamespace(
                extent=SimpleNamespace(x=1.0, y=0.5, z=0.75)
            ),
        )
        blocker = SimpleNamespace(
            id=2,
            type_id="walker.pedestrian.0001",
            bounding_box=SimpleNamespace(
                extent=SimpleNamespace(x=0.25, y=0.25, z=0.9)
            ),
            get_location=lambda: carla.Location(x=0.0, y=1.5, z=0.0),
        )
        simulation.carla_world = SimpleNamespace(
            get_actors=lambda: [actor, blocker]
        )
        old = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        target = carla.Transform(carla.Location(x=0.0, y=3.0, z=0.0))

        with self.assertRaisesRegex(SimulationCreationError, "occupied swept corridor"):
            simulation._assert_pending_lane_reconciliation_clear(
                actor, old, target
            )

    def _legacy_pending_lane_reconciliation_requires_immediate_commit(self):
        state = {
            "segmentId": "A",
            "destinationRoadId": "Z",
            "transitionPending": False,
            "laneIndex": 1,
        }
        simulation, obj, calls = self._sync_guard_fixture(
            state,
            "B",
            ["A", "B", "C"],
            observed_lane=3,
            transition_result={
                "data": [
                    {
                        "segmentId": "B",
                        "laneIndex": 2,
                        "transitionPending": True,
                    }
                ]
            },
            teleport_result={
                "data": [
                    {
                        "segmentId": "B",
                        "laneIndex": 2,
                        "transitionPending": True,
                        "transitionCommitted": False,
                    }
                ]
            },
            lane_indices={"A_5": 1, "B_8": 2, "B_7": 3},
            lane_connections={("A", "B"): [("A_5", "B_8")]},
        )

        simulation.synchronize_clients()
        state.update(
            {
                "segmentId": "B",
                "transitionPending": True,
                "transitionTargetRoadId": "B",
            }
        )
        with self.assertRaisesRegex(
            SimulationCreationError, "did not commit the verified lane reconciliation"
        ):
            simulation.synchronize_clients()

        self.assertEqual(len(calls["transform"]), 1)
        self.assertEqual(len(calls["teleport"]), 2)
        self.assertIn(obj, simulation.road_entry_holds)
        self.assertNotEqual(calls["teleport"][1][1].get("lane_index"), 2)

    def _legacy_direct_lane_mismatch_rejects_nonpending_entry(self):
        state = {
            "segmentId": "A",
            "destinationRoadId": "Z",
            "transitionPending": False,
            "laneIndex": 1,
        }
        simulation, obj, calls = self._sync_guard_fixture(
            state,
            "B",
            ["A", "B", "C"],
            observed_lane=3,
            transition_result={
                "data": [{"segmentId": "B", "laneIndex": 2, "transitionPending": False}]
            },
            lane_indices={"A_5": 1, "B_8": 2, "B_7": 3},
            lane_connections={("A", "B"): [("A_5", "B_8")]},
        )

        with self.assertRaisesRegex(
            SimulationCreationError,
            "lane_mismatch_entry_committed_without_recovery",
        ):
            simulation.synchronize_clients()

        self.assertIsNone(calls["teleport"][0][1]["lane_index"])
        self.assertEqual(calls["generate"], [])
        self.assertEqual(len(calls["teleport"]), 1)
        self.assertIn(obj, simulation.road_entry_holds)

    def _legacy_pending_skipped_target_reconciles_route_compatible_successor_lane(self):
        state = {
            "segmentId": "B",
            "destinationRoadId": "Z",
            "transitionPending": True,
            "transitionTargetRoadId": "B",
            "transitionTargetLaneIndex": 0,
        }
        recovery_path = [SimpleNamespace(name="successor-lane-0")]
        simulation, obj, calls = self._sync_guard_fixture(
            state,
            "C",
            ["B", "C", "D"],
            observed_lane=2,
            object_route=["A", "B", "C", "D"],
            lane_indices={
                "B_2": 0,
                "C_2": 0,
                "C_3": 1,
                "C_4": 2,
                "D_2": 0,
            },
            lane_connections={
                ("B", "C"): [
                    ("B_2", "C_2"),
                    ("B_2", "C_3"),
                    ("B_2", "C_4"),
                ],
                ("C", "D"): [("C_2", "D_2")],
            },
            generated_trajectory=recovery_path,
        )
        old_transform = obj.carlaActor.get_transform()
        target_transform = carla.Transform(
            carla.Location(x=10.0, y=-5.0, z=0.0),
            carla.Rotation(yaw=0.0),
        )
        simulation._pending_lane_reconciliation_target = (
            lambda target, road, lane: ("C_2", old_transform, target_transform)
        )
        simulation._mapped_carla_observation = lambda location: (
            ("C", 0)
            if abs(float(location.y) + 5.0) < 1e-6
            else ("C", 2)
        )

        def teleport(*args, **kwargs):
            calls["teleport"].append((args, kwargs))
            if len(calls["teleport"]) == 1:
                return metsr_response(
                    "teleportCoSimVeh",
                    [
                        {
                            "segmentId": "B",
                            "laneIndex": 0,
                            "transitionPending": True,
                            "transitionCommitted": False,
                        }
                    ],
                )
            return metsr_response(
                "teleportCoSimVeh",
                [
                    {
                        "transitionPending": False,
                        "transitionCommitted": True,
                    }
                ],
            )

        simulation.metsr_client.teleport_cosim_vehicle = teleport
        simulation.metsr_client.query_vehicle = lambda *args, **kwargs: metsr_response(
            "vehicle",
            [
                {"segmentId": "B", "laneIndex": 0, "transitionPending": False}
            ],
        )

        simulation.synchronize_clients()

        self.assertEqual(calls["generate"][0][:3], (["C", "D"], obj, 0))
        self.assertEqual(len(calls["transform"]), 1)
        verification = simulation.pending_lane_reconciliation_verifications[obj]
        self.assertEqual(
            (
                verification["pendingRoadID"],
                verification["pendingLaneID"],
                verification["roadID"],
                verification["laneID"],
                verification["observedRoadID"],
            ),
            ("B", 0, "C", 0, "C"),
        )
        self.assertIn(obj, simulation.road_entry_holds)

        simulation.synchronize_clients()

        self.assertEqual(len(calls["teleport"]), 2)
        self.assertEqual(calls["teleport"][1][1]["segment_id"], "C")
        self.assertIsNone(calls["teleport"][1][1]["lane_index"])
        self.assertNotIn(obj, simulation.pending_lane_reconciliation_verifications)
        self.assertEqual(
            (
                simulation.pending_road_entries[obj]["source"],
                simulation.pending_road_entries[obj]["target"],
            ),
            ("B", "C"),
        )
        self.assertIn(obj, simulation.road_entry_holds)

        # The skipped-target commit leaves METS-R on B. A third fresh query
        # must perform B->C before releasing the safety hold.
        state.update({"segmentId": "B", "laneIndex": 0, "transitionPending": False})
        simulation.carla_control_roads["C"] = True

        simulation.synchronize_clients()

        self.assertEqual(len(calls["teleport"]), 4)
        self.assertEqual(calls["teleport"][2][1]["segment_id"], "C")
        self.assertEqual(calls["teleport"][2][1]["lane_index"], 0)
        self.assertEqual(calls["teleport"][3][1]["segment_id"], "C")
        self.assertEqual(calls["teleport"][3][1]["lane_index"], 0)
        self.assertNotIn(obj, simulation.pending_road_entries)
        self.assertNotIn(obj, simulation.road_entry_holds)

    def test_pending_successor_hint_remaining_pending_is_acknowledged(self):
        state = {
            "segmentId": "B",
            "destinationRoadId": "Z",
            "transitionPending": True,
            "transitionTargetRoadId": "B",
        }
        simulation, _, calls = self._sync_guard_fixture(
            state,
            "C",
            ["B", "C", "D"],
            object_route=["A", "B", "C", "D"],
            teleport_result={"data": [dict(state)]},
        )

        simulation.metsr_lane_connections = {("B", "C"): [("B_0", "C_2")]}
        simulation.synchronize_clients()

        self.assertEqual(len(calls["teleport"]), 1)
        self.assertEqual(calls["teleport"][0][1]["segment_id"], "C")
        self.assertEqual(calls["transform"], [])

    def test_predecessor_boundary_flicker_is_not_forward(self):
        self.assertFalse(
            CosimSimulation._is_forward_road_observation(
                "-47", "4", ["4", "-47", "-48", "-49"]
            )
        )

    def test_immediate_successor_observation_is_forward(self):
        self.assertTrue(
            CosimSimulation._is_forward_road_observation(
                "-47", "-48", ["-47", "-48", "-49"]
            )
        )

    def test_skipped_forward_observation_is_forward(self):
        self.assertTrue(
            CosimSimulation._is_forward_road_observation(
                "-47", "-49", ["-47", "-48", "-49"]
            )
        )

    def test_multi_edge_skip_is_not_forward(self):
        self.assertFalse(
            CosimSimulation._is_forward_road_observation(
                "A", "D", ["A", "B", "C", "D"]
            )
        )

    def test_unrelated_road_observation_is_not_forward(self):
        self.assertFalse(
            CosimSimulation._is_forward_road_observation(
                "-47", "27", ["-47", "-48", "-49"]
            )
        )

    def test_current_carla_waypoint_overrides_stale_scenic_lane(self):
        simulation = object.__new__(CosimSimulation)
        simulation.metsr_lane_indices = {"-48_4": 2, "-48_3": 1}
        simulation.metsr_road_cache = {
            ("-48", 2): [(0.0, 0.0), (10.0, 0.0)],
            ("-48", 1): [(0.0, 3.5), (10.0, 3.5)],
        }
        current_waypoint = SimpleNamespace(
            road_id=48,
            lane_id=-6,
            transform=carla.Transform(
                carla.Location(x=5.0, y=-3.5, z=0.0), carla.Rotation()
            ),
        )
        stale_lane = SimpleNamespace(
            road=SimpleNamespace(id=48), id=-5
        )
        simulation.map = SimpleNamespace(
            get_waypoint=lambda *args, **kwargs: current_waypoint
        )
        simulation.network_helper = SimpleNamespace(
            scenic_to_metsr_map_lanes={
                "48_-5": ["-48_4"],
                "48_-6": ["-48_3"],
            },
            metsr_represented_roads={"-48"},
            _nearest_lane=lambda obj: stale_lane,
        )
        location = SimpleNamespace(x=5.0, y=-3.5, z=0.0)

        road_id, lane_id = simulation._mapped_carla_observation(location)

        self.assertEqual((road_id, lane_id), ("-48", 1))

    def test_queue_service_ignores_old_roads_and_unmanaged_vehicles(self):
        class ManagedObject:
            pass

        managed = ManagedObject()
        admitted = []
        simulation = object.__new__(CosimSimulation)
        simulation.carla_control_roads = {"A": True, "OLD": True}
        simulation.active_bubble_metsr_roads = {"A"}
        simulation.pv_id_map = {managed: 11}
        simulation.admitted_queue_vehicles = {}
        simulation.count = 7

        def admit(requests):
            admitted.append(requests)
            return {
                "messageType": "enterRoadFromQueue",
                "status": "ok",
                "data": [
                    {
                        "vehicleId": requests["vehicleId"],
                        "internalVehicleId": requests["internalVehicleId"],
                        "isPrivate": True,
                        "roadId": requests["roadId"],
                        "status": "ok",
                    }
                ],
            }

        simulation.metsr_client = SimpleNamespace(
            query_cosim_entering_vehicle_queue=lambda: {
                "messageType": "coSimEnteringVehicleQueue",
                "status": "ok",
                "data": [
                    {
                        "segmentId": "A",
                        "status": "ok",
                        "queue": [
                            {
                                "vehicleId": 99,
                                "isPrivate": False,
                                "ready": True,
                            },
                            {
                                "vehicleId": 11,
                                "internalVehicleId": 101,
                                "isPrivate": True,
                                "ready": True,
                            },
                        ],
                    },
                    {
                        "segmentId": "OLD",
                        "status": "ok",
                        "queue": [
                            {
                                "vehicleId": 11,
                                "isPrivate": True,
                                "ready": True,
                            }
                        ],
                    },
                ]
            },
            enter_road_from_queue=admit,
        )

        simulation._service_metsr_entering_queues()

        self.assertEqual(
            admitted,
            [
                {
                    "vehicleId": 11,
                    "internalVehicleId": 101,
                    "isPrivate": True,
                    "roadId": "A",
                }
            ],
        )
        self.assertEqual(
            simulation.admitted_queue_vehicles,
            {11: {"roadId": "A", "admittedAt": 7}},
        )

    def test_queue_service_batches_ready_managed_vehicles(self):
        class ManagedObject:
            pass

        first = ManagedObject()
        second = ManagedObject()
        requests_seen = []
        simulation = object.__new__(CosimSimulation)
        simulation.carla_control_roads = {"A": True}
        simulation.active_bubble_metsr_roads = {"A"}
        simulation.pv_id_map = {first: 11, second: 12}
        simulation.admitted_queue_vehicles = {}
        simulation.count = 9

        def admit(requests):
            requests_seen.append(requests)
            return metsr_response(
                "enterRoadFromQueue",
                [
                    {
                        "vehicleId": 12,
                        "internalVehicleId": 102,
                        "isPrivate": True,
                        "roadId": "A",
                        "status": "ok",
                    },
                    {
                        "vehicleId": 11,
                        "internalVehicleId": 101,
                        "isPrivate": True,
                        "roadId": "A",
                        "status": "ok",
                    },
                ],
            )

        simulation.metsr_client = SimpleNamespace(
            query_cosim_entering_vehicle_queue=lambda: metsr_response(
                "coSimEnteringVehicleQueue",
                [
                    {
                        "segmentId": "A",
                        "status": "ok",
                        "queue": [
                            {
                                "vehicleId": 11,
                                "internalVehicleId": 101,
                                "isPrivate": True,
                                "ready": True,
                            },
                            {
                                "vehicleId": 12,
                                "internalVehicleId": 102,
                                "isPrivate": True,
                                "ready": True,
                            },
                        ],
                    }
                ],
            ),
            enter_road_from_queue=admit,
        )

        simulation._service_metsr_entering_queues()

        self.assertEqual(len(requests_seen), 1)
        self.assertEqual(
            [request["vehicleId"] for request in requests_seen[0]],
            [11, 12],
        )
        self.assertEqual(
            simulation.admitted_queue_vehicles,
            {
                11: {"roadId": "A", "admittedAt": 9},
                12: {"roadId": "A", "admittedAt": 9},
            },
        )

    def test_queue_service_requires_explicit_ready_and_native_envelope(self):
        class ManagedObject:
            pass

        managed = ManagedObject()
        managed.carla_actor_flag = False
        simulation = object.__new__(CosimSimulation)
        simulation.carla_control_roads = {"A": True}
        simulation.active_bubble_metsr_roads = {"A"}
        simulation.pv_id_map = {managed: 11}
        simulation.admitted_queue_vehicles = {}
        simulation.count = 0
        admitted = []
        simulation.metsr_client = SimpleNamespace(
            query_cosim_entering_vehicle_queue=lambda: {
                "messageType": "coSimEnteringVehicleQueue",
                "status": "ok",
                "data": [
                    {
                        "segmentId": "A",
                        "status": "ok",
                        "queue": [
                            {
                                "vehicleId": 11,
                                "internalVehicleId": 101,
                                "isPrivate": True,
                            }
                        ],
                    }
                ],
            },
            enter_road_from_queue=lambda **kwargs: admitted.append(kwargs),
        )

        simulation._service_metsr_entering_queues()

        self.assertEqual(admitted, [])

    def test_admitted_queue_vehicle_has_bounded_carla_spawn_deadline(self):
        class ManagedObject:
            pass

        managed = ManagedObject()
        managed.carla_actor_flag = False
        simulation = object.__new__(CosimSimulation)
        simulation.pv_id_map = {managed: 11}
        simulation.admitted_queue_vehicles = {
            11: {"roadId": "A", "admittedAt": 2}
        }
        simulation.count = 2 + COSIM_ADMISSION_SPAWN_MAX_TICKS

        with self.assertRaisesRegex(
            SimulationCreationError, "did not spawn it within"
        ):
            simulation._check_queue_admission_spawn_deadlines()

    def _connector_control_fixture(self, represented_roads, edge_connections=None):
        simulation = object.__new__(CosimSimulation)
        simulation.carla_control_roads = {}
        simulation.carla_control_segments = set()
        simulation.metsr_internal_edge_to_connector = {}
        simulation.metsr_connector_records = {}
        simulation.metsr_road_connectors = {}
        simulation.metsr_internal_edge_connections = dict(edge_connections or {})
        simulation.metsr_internal_lane_connections = {}
        lane_ids = {
            str(lane)
            for pairs in simulation.metsr_internal_edge_connections.values()
            for pair in pairs
            for lane in pair
        }
        simulation.metsr_lane_indices = {
            lane: index
            for index, lane in enumerate(sorted(lane_ids))
        }
        simulation.network_helper = SimpleNamespace(
            metsr_represented_roads=set(map(str, represented_roads))
        )
        return simulation

    @staticmethod
    def _takeover_response(
        road_id,
        connector_id,
        internal_edge_ids,
        source_id,
        target_id=None,
    ):
        if target_id is None:
            target_id = road_id
        return {
            "messageType": "setCoSimRoad",
            "status": "ok",
            "data": [
                {
                    "roadId": road_id,
                    "status": "ok",
                    "connectorIds": [connector_id],
                    "connectors": [
                        {
                            "connectorId": connector_id,
                            "sourceRoadId": source_id,
                            "targetRoadId": target_id,
                            "internalEdgeIds": list(internal_edge_ids),
                            "controlMode": "COSIM",
                        }
                    ],
                }
            ],
        }

    def test_freeze_maps_exact_1146_internal_edge_and_opaque_connector(self):
        simulation = self._connector_control_fixture(
            {"-37"}, {":1146_3": [("-36_6", "-37_5")]}
        )
        calls = []
        connector_id = "cont/-36/-37"
        simulation.metsr_client = SimpleNamespace(
            set_cosim_road=lambda roads: (
                calls.append(list(roads))
                or self._takeover_response(
                    "-37", connector_id, [":1146_3"], "-36"
                )
            )
        )

        self.assertIsNone(simulation.freeze_roads(["-37"]))

        self.assertEqual(calls, [["-37"]])
        self.assertEqual(
            simulation.metsr_internal_edge_to_connector,
            {":1146_3": connector_id},
        )
        self.assertEqual(simulation.carla_control_segments, {"-37", ":1146_3"})
        self.assertNotIn(connector_id, simulation.carla_control_segments)

    def test_freeze_accepts_outbound_connector_touching_controlled_source(self):
        connector_id = "cont/A/B"
        simulation = self._connector_control_fixture(
            {"A"}, {":A_B": [("A_0", "B_0")]}
        )
        simulation.metsr_client = SimpleNamespace(
            set_cosim_road=lambda roads: self._takeover_response(
                "A",
                connector_id,
                [":A_B"],
                "A",
                target_id="B",
            )
        )

        simulation.freeze_roads(["A"])

        self.assertEqual(
            simulation.metsr_road_connectors, {"A": {connector_id}}
        )
        self.assertEqual(
            simulation.metsr_internal_edge_to_connector,
            {":A_B": connector_id},
        )

    def test_freeze_adds_every_internal_edge_in_multi_edge_connector(self):
        simulation = self._connector_control_fixture(
            {"-14_2"}, {":452_4": [("58_2_0", "-14_2_0")]}
        )
        connector_id = "cont/58_2/-14_2"
        simulation.metsr_client = SimpleNamespace(
            set_cosim_road=lambda roads: self._takeover_response(
                "-14_2",
                connector_id,
                [":452_4", ":452_14"],
                "58_2",
            )
        )

        simulation.freeze_roads(["-14_2"])

        self.assertEqual(
            simulation.carla_control_segments,
            {"-14_2", ":452_4", ":452_14"},
        )
        self.assertEqual(
            simulation.metsr_internal_edge_to_connector,
            {":452_4": connector_id, ":452_14": connector_id},
        )

    def test_connector_id_is_opaque_for_special_and_legacy_formats(self):
        cases = (
            ("dst_under/%:road", "opaque/_/%/:/connector"),
            ("-37", "-36_-37"),
        )
        for target_road, connector_id in cases:
            with self.subTest(connector_id=connector_id):
                edge_id = ":edge_under/%:part"
                source_road = "src_under/%:road"
                simulation = self._connector_control_fixture(
                    {target_road},
                    {edge_id: [(f"{source_road}_0", f"{target_road}_0")]},
                )
                simulation.metsr_client = SimpleNamespace(
                    set_cosim_road=lambda roads: self._takeover_response(
                        target_road, connector_id, [edge_id], source_road
                    )
                )

                simulation.freeze_roads([target_road])

                self.assertEqual(
                    simulation.metsr_internal_edge_to_connector[edge_id],
                    connector_id,
                )
                self.assertNotIn(connector_id, simulation.carla_control_segments)

    def test_exact_1146_physical_observation_is_accepted_without_lane_claim(self):
        simulation = object.__new__(CosimSimulation)
        simulation.map = SimpleNamespace(
            get_waypoint=lambda *args, **kwargs: SimpleNamespace(
                road_id=1146,
                lane_id=-3,
                transform=carla.Transform(
                    carla.Location(x=181.81, y=136.67, z=0.0),
                    carla.Rotation(),
                ),
            )
        )
        simulation.network_helper = SimpleNamespace(
            scenic_to_metsr_map_lanes={"1146_-3": [":1146_3_5"]},
            metsr_represented_roads={"-36", "-37"},
        )
        simulation.metsr_lane_indices = {}
        simulation.carla_control_roads = {"-37": True}
        simulation.carla_control_segments = {"-37", ":1146_3"}
        simulation.metsr_internal_edge_to_connector = {
            ":1146_3": "cont/-36/-37"
        }
        simulation.metsr_connector_records = {}
        simulation.metsr_road_connectors = {}

        observed = simulation._mapped_carla_observation(
            SimpleNamespace(x=181.81, y=136.67, z=0.0)
        )

        self.assertEqual(observed, (":1146_3", None))
        self.assertEqual(
            simulation._classify_physical_road_observation(
                "-37", observed[0], ["-37", "next"]
            ),
            ROAD_OBSERVATION_UNKNOWN,
        )

    @staticmethod
    def _unmapped_21_to_20_connector_fixture(
        *, junction=True, projection_distance=0.0, mapped_road=None
    ):
        connector_id = "cont/21/20"

        class Vehicle:
            pass

        obj = Vehicle()
        obj.name = "ego"
        obj.route = ["20", "destination"]
        obj.pcla = object()

        waypoint = SimpleNamespace(
            road_id=900,
            lane_id=-1,
            is_junction=junction,
            transform=carla.Transform(
                carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation()
            ),
        )
        location = carla.Location(
            x=projection_distance,
            y=0.0,
            z=0.0,
        )
        mappings = (
            {}
            if mapped_road is None
            else {"900_-1": [f"{mapped_road}_0"]}
        )

        simulation = object.__new__(CosimSimulation)
        simulation.ego = obj
        simulation.map = SimpleNamespace(
            get_waypoint=lambda *args, **kwargs: waypoint
        )
        simulation.scenic_to_metsr_map = mappings
        simulation.network_helper = SimpleNamespace(
            scenic_to_metsr_map_lanes=mappings,
            metsr_represented_roads={"20", "unrelated"},
        )
        simulation.metsr_lane_indices = {"20_0": 0, "unrelated_0": 0}
        # Keep road 20 geometrically close so the rejection tests prove that
        # invalid waypoint evidence is not rescued by centerline fallback.
        simulation.metsr_road_cache = {
            ("20", 0): [(-10.0, 0.0), (10.0, 0.0)],
            ("unrelated", 0): [(-10.0, 0.0), (10.0, 0.0)],
        }
        simulation.carla_control_roads = {"20": True}
        simulation.carla_control_segments = {"20", ":junction_internal"}
        simulation.metsr_internal_edge_to_connector = {
            ":junction_internal": connector_id
        }
        simulation.metsr_connector_records = {
            connector_id: {
                "connectorId": connector_id,
                "sourceRoadId": "21",
                "targetRoadId": "20",
                "internalEdgeIds": [":junction_internal"],
            }
        }
        simulation.metsr_road_connectors = {"20": {connector_id}}
        simulation._carla_authoritative_segments = {obj: connector_id}
        return simulation, obj, location, connector_id

    def test_unmapped_owned_junction_preserves_21_to_20_connector(self):
        simulation, obj, location, connector_id = (
            self._unmapped_21_to_20_connector_fixture(
                projection_distance=CARLA_OWNED_PROJECTION_TOLERANCE
            )
        )

        raw_road, raw_lane = simulation._mapped_carla_observation(location)
        resolved = simulation._resolve_carla_owned_observation(
            obj,
            location,
            raw_road,
            raw_lane,
            {},
            {},
            vehicle_id=1,
        )

        self.assertEqual((raw_road, raw_lane), (None, None))
        self.assertEqual(resolved, (connector_id, None, "20"))
        self.assertEqual(
            simulation._carla_authoritative_segments[obj], connector_id
        )

    def test_mapped_target_observation_switches_connector_to_road_20(self):
        simulation, obj, location, _ = (
            self._unmapped_21_to_20_connector_fixture(mapped_road="20")
        )

        raw_road, raw_lane = simulation._mapped_carla_observation(location)
        resolved = simulation._resolve_carla_owned_observation(
            obj,
            location,
            raw_road,
            raw_lane,
            {},
            {},
            vehicle_id=1,
        )

        self.assertEqual((raw_road, raw_lane), ("20", 0))
        self.assertEqual(resolved, ("20", 0, "20"))
        self.assertEqual(simulation._carla_authoritative_segments[obj], "20")

    def test_unmapped_nonjunction_gap_does_not_preserve_connector(self):
        simulation, obj, location, _ = (
            self._unmapped_21_to_20_connector_fixture(junction=False)
        )

        raw_road, raw_lane = simulation._mapped_carla_observation(location)
        with self.assertRaisesRegex(
            SimulationCreationError, "outside its METS-R ownership domain"
        ):
            simulation._resolve_carla_owned_observation(
                obj,
                location,
                raw_road,
                raw_lane,
                {},
                {},
                vehicle_id=1,
            )

    def test_unmapped_junction_beyond_projection_tolerance_fails(self):
        simulation, obj, location, _ = (
            self._unmapped_21_to_20_connector_fixture(
                projection_distance=CARLA_OWNED_PROJECTION_TOLERANCE + 0.001
            )
        )

        raw_road, raw_lane = simulation._mapped_carla_observation(location)
        with self.assertRaisesRegex(
            SimulationCreationError, "projection error=4.251 m"
        ):
            simulation._resolve_carla_owned_observation(
                obj,
                location,
                raw_road,
                raw_lane,
                {},
                {},
                vehicle_id=1,
            )

    def test_pose_without_carla_driving_waypoint_fails(self):
        simulation, obj, location, _ = (
            self._unmapped_21_to_20_connector_fixture()
        )
        simulation.map = SimpleNamespace(
            get_waypoint=lambda *args, **kwargs: None
        )

        raw_road, raw_lane = simulation._mapped_carla_observation(location)
        with self.assertRaisesRegex(
            SimulationCreationError, "projection error=unavailable"
        ):
            simulation._resolve_carla_owned_observation(
                obj,
                location,
                raw_road,
                raw_lane,
                {},
                {},
                vehicle_id=1,
            )

    def test_explicit_unrelated_road_does_not_preserve_connector(self):
        simulation, obj, location, _ = (
            self._unmapped_21_to_20_connector_fixture(
                mapped_road="unrelated"
            )
        )

        raw_road, raw_lane = simulation._mapped_carla_observation(location)
        self.assertEqual((raw_road, raw_lane), ("unrelated", 0))
        with self.assertRaisesRegex(
            SimulationCreationError, "raw CARLA road/lane=unrelated/0"
        ):
            simulation._resolve_carla_owned_observation(
                obj,
                location,
                raw_road,
                raw_lane,
                {},
                {},
                vehicle_id=1,
            )

    def _town06_merging_connector_fixture(self):
        class Ego:
            name = "ego"
            route = ["6", "5", "-37"]

        obj = Ego()
        # Last published seed-37 pose and path geometry from the live METS-R
        # server. CARLA's nearest driving waypoint flickers onto :332_8.
        location = carla.Location(x=666.48364, y=136.82701, z=0.0)
        previous = "cont/6/5"
        other = "cont/-38/5"
        simulation = object.__new__(CosimSimulation)
        waypoint = SimpleNamespace(
            road_id=334, lane_id=4, is_junction=True,
            transform=carla.Transform(
                carla.Location(x=location.x + 1.292, y=location.y, z=0.0),
                carla.Rotation(),
            ),
        )
        simulation.map = SimpleNamespace(get_waypoint=lambda *args, **kwargs: waypoint)
        simulation.network_helper = SimpleNamespace(
            scenic_to_metsr_map_lanes={"334_4": [":332_8_2"]},
            metsr_represented_roads={"6", "5", "-38"},
        )
        simulation.carla_control_roads = {"6": True, "5": True, "-38": True}
        simulation.carla_control_segments = {"6", "5", "-38", ":332_0", ":332_8"}
        simulation.metsr_internal_edge_to_connector = {":332_0": previous, ":332_8": other}
        simulation.metsr_connector_records = {
            previous: {
                "connectorId": previous, "sourceRoadId": "6", "targetRoadId": "5",
                "intersectionId": 14, "internalEdgeIds": [":332_0"],
                "paths": [{"connectorPathId": 2, "viaLaneIds": [":332_0_4"]}],
            },
            other: {
                "connectorId": other, "sourceRoadId": "-38", "targetRoadId": "5",
                "intersectionId": 14, "internalEdgeIds": [":332_8"],
                "paths": [{"connectorPathId": 2, "viaLaneIds": [":332_8_2"]}],
            },
        }
        simulation.metsr_road_connectors = {"5": {previous, other}}
        simulation.metsr_connector_centerline_cache = {
            (previous, 2): [(664.8, -168.58, 0.0), (665.32, -120.35, 0.0)]
        }
        simulation._carla_authoritative_segments = {obj: previous}
        state = {"segmentId": previous, "connectorId": previous, "connectorPathId": 2}
        return simulation, obj, location, waypoint, state

    def test_merging_332_8_observation_preserves_occupied_6_to_5_connector(self):
        simulation, obj, location, _, state = self._town06_merging_connector_fixture()
        resolved = simulation._resolve_carla_owned_observation(
            obj, location, ":332_8", None, state, state, vehicle_id=0
        )
        self.assertEqual(resolved, ("cont/6/5", None, "5"))
        self.assertEqual(simulation._carla_authoritative_segments[obj], "cont/6/5")
        self.assertEqual(
            simulation._connector_path_id_for_pose(resolved[0], location, state), 2
        )

    def test_connector_merge_continuity_requires_geometry_and_matching_exit(self):
        for invalid_evidence in (
            "far_from_occupied_path", "far_from_waypoint", "nonjunction",
            "different_exit", "different_intersection", "missing_geometry",
        ):
            with self.subTest(invalid_evidence=invalid_evidence):
                simulation, obj, location, waypoint, state = self._town06_merging_connector_fixture()
                if invalid_evidence == "far_from_occupied_path":
                    simulation.metsr_connector_centerline_cache[("cont/6/5", 2)] = [
                        (650.0, -168.58), (650.0, -120.35)
                    ]
                elif invalid_evidence == "far_from_waypoint":
                    waypoint.transform.location = carla.Location(
                        x=location.x + CARLA_OWNED_PROJECTION_TOLERANCE + 0.1,
                        y=location.y, z=location.z,
                    )
                elif invalid_evidence == "nonjunction":
                    waypoint.is_junction = False
                elif invalid_evidence == "different_exit":
                    simulation.metsr_connector_records["cont/-38/5"]["targetRoadId"] = "4"
                elif invalid_evidence == "different_intersection":
                    simulation.metsr_connector_records["cont/-38/5"]["intersectionId"] = 15
                else:
                    simulation.metsr_connector_centerline_cache[("cont/6/5", 2)] = []
                with self.assertRaises(SimulationCreationError):
                    simulation._resolve_carla_owned_observation(
                        obj, location, ":332_8", None, state, state, vehicle_id=0
                    )

    def test_connector_merge_still_accepts_observed_downstream_road(self):
        simulation, obj, location, _, state = self._town06_merging_connector_fixture()
        resolved = simulation._resolve_carla_owned_observation(
            obj, location, "5", 2, state, state, vehicle_id=0
        )
        self.assertEqual(resolved, ("5", 2, "5"))

    def _town06_fork_connector_fixture(self, y=149.12698):
        simulation, obj, _, waypoint, _ = self._town06_merging_connector_fixture()
        # Last seed-36 pose and occupied path from the native server.
        location = carla.Location(x=-224.19861, y=y, z=0.0)
        waypoint.transform.location = carla.Location(
            x=location.x + 2.767, y=location.y, z=0.0
        )
        previous, observed = "cont/21/20", "cont/21/-7"
        obj.route = ["21", "20", "-67"]
        simulation.carla_control_roads = {"21": True, "20": True, "-7": True}
        simulation.carla_control_segments = {"21", "20", "-7", ":396_0", ":396_6"}
        simulation.metsr_internal_edge_to_connector = {":396_0": previous, ":396_6": observed}
        simulation.network_helper.scenic_to_metsr_map_lanes = {"334_4": [":396_6_0"]}
        simulation.metsr_connector_records = {
            previous: {
                "connectorId": previous, "sourceRoadId": "21", "targetRoadId": "20",
                "intersectionId": 16, "internalEdgeIds": [":396_0"],
                "paths": [{"connectorPathId": 3, "viaLaneIds": [":396_0_5"]}],
            },
            observed: {
                "connectorId": observed, "sourceRoadId": "21", "targetRoadId": "-7",
                "intersectionId": 16, "internalEdgeIds": [":396_6"],
                "paths": [{"connectorPathId": 0, "viaLaneIds": [":396_6_0"]}],
            },
        }
        simulation.metsr_road_connectors = {"21": {previous, observed}}
        simulation.metsr_connector_centerline_cache = {
            (previous, 3): [(-205.88, -145.66), (-239.44, -145.58)],
            (observed, 0): [
                (-205.89, -149.16), (-215.26, -149.45), (-221.69, -150.85),
                (-227.50, -154.06), (-234.99, -159.78),
            ],
        }
        simulation._carla_authoritative_segments = {obj: previous}
        state = {"segmentId": previous, "connectorId": previous, "connectorPathId": 3}
        return simulation, obj, location, state

    def test_fork_396_6_observation_preserves_occupied_path_while_ambiguous(self):
        simulation, obj, location, state = self._town06_fork_connector_fixture()
        resolved = simulation._resolve_carla_owned_observation(
            obj, location, ":396_6", None, state, state, vehicle_id=0
        )
        self.assertEqual(resolved, ("cont/21/20", None, "20"))

    def test_fork_accepts_carla_chosen_turn_when_paths_diverge(self):
        simulation, obj, location, state = self._town06_fork_connector_fixture(y=151.5)
        resolved = simulation._resolve_carla_owned_observation(
            obj, location, ":396_6", None, state, state, vehicle_id=0
        )
        self.assertEqual(resolved, ("cont/21/-7", None, "-7"))
        self.assertEqual(simulation._carla_authoritative_segments[obj], "cont/21/-7")
        self.assertEqual(obj.route, ["21", "20", "-67"])
        self.assertEqual(
            simulation._connector_path_id_for_pose(resolved[0], location, state), 0
        )
        downstream = simulation._resolve_carla_owned_observation(
            obj, location, "-7", 0, state, state, vehicle_id=0
        )
        self.assertEqual(downstream, ("-7", 0, "-7"))

    def test_fork_rejects_pose_outside_both_connector_paths(self):
        simulation, obj, location, state = self._town06_fork_connector_fixture(y=165)
        with self.assertRaises(SimulationCreationError):
            simulation._resolve_carla_owned_observation(
                obj, location, ":396_6", None, state, state, vehicle_id=0
            )

    def test_exact_1146_is_sent_as_canonical_connector_not_logical_route(self):
        calls = []

        class Vehicle:
            pass

        actor = SimpleNamespace(
            get_transform=lambda: carla.Transform(
                carla.Location(), carla.Rotation(yaw=10.0)
            ),
            get_velocity=lambda: SimpleNamespace(x=2.0, y=0.0, z=0.0),
        )
        obj = Vehicle()
        obj.name = "car_4"
        obj.route = ["-37", "destination"]
        obj.carlaActor = actor
        simulation = object.__new__(CosimSimulation)
        simulation.carla_control_roads = {"-37": True}
        simulation.carla_control_segments = {"-37", ":1146_3"}
        connector_id = "cont/-36/-37"
        simulation.metsr_internal_edge_to_connector = {
            ":1146_3": connector_id
        }
        simulation.metsr_connector_records = {
            connector_id: {
                "connectorId": connector_id,
                "sourceRoadId": "-36",
                "targetRoadId": "-37",
                "internalEdgeIds": [":1146_3"],
            }
        }
        simulation.metsr_road_connectors = {"-37": {connector_id}}
        simulation.road_entry_holds = {}
        simulation.road_entry_hold_diagnostics = {}
        simulation.pending_road_entries = {}
        simulation.pending_route_refreshes = {}
        simulation.pending_lane_reconciliation_verifications = {}
        simulation.same_road_departure_lane_verifications = {}
        simulation._pending_pcla_pose_resets = set()
        simulation.destination_route_cache = {}

        def teleport(*args, **kwargs):
            calls.append((args, kwargs))
            return {
                "messageType": "teleportCoSimVeh",
                "status": "ok",
                "data": [{"vehicleId": 5, "status": "ok"}],
            }

        simulation.metsr_client = SimpleNamespace(
            teleport_cosim_vehicle=teleport,
            query_route_between_roads=lambda *args: (_ for _ in ()).throw(
                AssertionError("an internal physical edge is not a logical route road")
            ),
            update_vehicle_route=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("an internal physical edge must not replace the route")
            ),
        )

        simulation._synchronize_pcla_ego_shadow(
            obj,
            5,
            {
                "segmentId": "-37",
                "destinationRoadId": "destination",
                "transitionPending": False,
            },
            {"routeRoadIds": ["-37", "destination"]},
            carla.Location(x=181.81, y=136.67, z=0.0),
            connector_id,
            None,
            "-37",
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["segment_id"], connector_id)
        self.assertIsNone(calls[0][1]["lane_index"])
        self.assertEqual(obj.route, ["-37", "destination"])

    def test_exact_car_4_source_boundary_uses_canonical_connector(self):
        simulation = object.__new__(CosimSimulation)
        connector_id = "cont/-36/-37"
        simulation.carla_control_roads = {"-37": True}
        simulation.carla_control_segments = {"-37", ":1146_3"}
        simulation.metsr_internal_edge_to_connector = {
            ":1146_3": connector_id
        }
        simulation.metsr_connector_records = {
            connector_id: {
                "connectorId": connector_id,
                "sourceRoadId": "-36",
                "targetRoadId": "-37",
                "internalEdgeIds": [":1146_3"],
            }
        }
        simulation.metsr_road_connectors = {"-37": {connector_id}}
        record = {
            "vehicleId": 5,
            "segmentId": connector_id,
            "segmentType": "connector",
            "connectorId": connector_id,
            "internalEdgeIds": [":1146_3"],
            "sourceRoadId": "-36",
            "targetRoadId": "-37",
            "transitionSourceRoadId": "-36",
            "transitionTargetRoadId": "-37",
            "transitionPending": True,
        }

        hint = simulation._canonical_connector_hint(record, "-36")

        self.assertEqual(hint, connector_id)
        self.assertEqual(
            simulation._observed_road_hint(
                SimpleNamespace(route=["-37", "destination"]), record, "-36"
            ),
            connector_id,
        )

    @staticmethod
    def _car_4_authority_fixture(raw_observations, teleport_response=None):
        connector_id = "cont/-36/-37"

        class Vehicle:
            pass

        class Actor:
            def __init__(self):
                self.location = carla.Location(x=181.81, y=136.67, z=0.0)
                self.transform = carla.Transform(
                    self.location, carla.Rotation(yaw=7.0)
                )
                self.transform_calls = []

            def get_location(self):
                return self.location

            def get_transform(self):
                return self.transform

            def get_velocity(self):
                return SimpleNamespace(x=2.0, y=0.0, z=0.0)

            def set_transform(self, transform):
                self.transform_calls.append(transform)
                self.transform = transform
                self.location = transform.location

        obj = Vehicle()
        obj.name = "car_4"
        obj.route = ["-37", "destination"]
        obj.carlaActor = Actor()
        obj.carla_actor_flag = True
        obj.autopilot_action = False
        obj.active_autopilot = False
        obj.spawn_guard = 0

        vehicle_state = {
            "vehicleId": 5,
            "segmentId": "-37",
            "laneIndex": 4,
            "destinationRoadId": "destination",
            "transitionSourceRoadId": "-36",
            "transitionTargetRoadId": "-37",
            "transitionPending": True,
        }
        cosim_record = {
            "vehicleId": 5,
            "isPrivate": True,
            "segmentId": connector_id,
            "segmentType": "connector",
            "connectorId": connector_id,
            "connectorPathId": 0,
            "sourceRoadId": "-36",
            "targetRoadId": "-37",
            "transitionSourceRoadId": "-36",
            "transitionTargetRoadId": "-37",
            "transitionPending": True,
            "routeRoadIds": ["-37", "destination"],
            "destinationRoadId": "destination",
        }
        simulation = object.__new__(CosimSimulation)
        simulation.carla_actors = [obj]
        simulation.metsr_actors = []
        simulation._carla_authoritative_segments = {}
        simulation.carla_control_roads = {"-37": True}
        simulation.carla_control_segments = {"-37", ":1146_3"}
        simulation.metsr_internal_edge_to_connector = {
            ":1146_3": connector_id
        }
        simulation.metsr_connector_records = {
            connector_id: {
                "connectorId": connector_id,
                "sourceRoadId": "-36",
                "targetRoadId": "-37",
                "internalEdgeIds": [":1146_3"],
            }
        }
        simulation.metsr_road_connectors = {"-37": {connector_id}}
        simulation.metsr_lane_indices = {}
        simulation.metsr_road_cache = {}
        simulation.pending_road_entries = {}
        simulation.pending_route_refreshes = {}
        simulation.pending_lane_reconciliation_verifications = {}
        simulation.same_road_departure_lane_verifications = {}
        simulation.road_entry_holds = {}
        simulation.road_entry_hold_diagnostics = {}
        simulation.completed_route = {}
        simulation.count = 36
        simulation.ego = None
        simulation._collect_metsr_vehicle_data = lambda objects: {
            obj: vehicle_state
        }
        simulation.getMetsrPrivateVehId = lambda target: 5

        observations = list(raw_observations)
        observation_index = 0

        def mapped_observation(location):
            nonlocal observation_index
            observation = observations[min(observation_index, len(observations) - 1)]
            observation_index += 1
            return observation

        simulation._mapped_carla_observation = mapped_observation
        calls = []

        def teleport(*args, **kwargs):
            calls.append((args, kwargs))
            if teleport_response is not None:
                return teleport_response
            return metsr_response(
                "teleportCoSimVeh",
                [
                    {
                        "vehicleId": 5,
                        "status": "ok",
                        "segmentId": connector_id,
                        "connectorPathId": 0,
                        "laneIndex": -1,
                        "transitionPending": True,
                    }
                ],
            )

        simulation.metsr_client = SimpleNamespace(
            query_cosim_vehicle=lambda: metsr_response(
                "coSimVehicle", [cosim_record]
            ),
            teleport_cosim_vehicle=teleport,
        )
        simulation.tm = SimpleNamespace(
            set_path=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("a connector acknowledgement must not rebuild TM route")
            ),
            auto_lane_change=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("a shadow response must not alter CARLA lane control")
            ),
        )
        simulation._apply_carla_transform_without_tick = (
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("a shadow response must not transform CARLA")
            )
        )
        simulation.check_client_synchronization = lambda: None
        return simulation, obj, calls

    def test_car_4_source_boundary_pending_ack_keeps_carla_authoritative(self):
        connector_id = "cont/-36/-37"
        simulation, obj, calls = self._car_4_authority_fixture([("-36", 4)])
        original_transform = obj.carlaActor.get_transform()
        original_route = list(obj.route)

        simulation.synchronize_clients()

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["segment_id"], connector_id)
        self.assertIsNone(calls[0][1]["lane_index"])
        self.assertIs(obj.carlaActor.get_transform(), original_transform)
        self.assertEqual(obj.carlaActor.transform_calls, [])
        self.assertEqual(obj.route, original_route)
        self.assertIn(obj, simulation.carla_actors)
        self.assertEqual(
            simulation._carla_authoritative_segments[obj], connector_id
        )

    def test_carla_authoritative_segment_does_not_flicker_back_to_source(self):
        connector_id = "cont/-36/-37"
        simulation, obj, calls = self._car_4_authority_fixture(
            [("-36", 4), ("-37", 3), ("-36", 4)]
        )

        simulation.synchronize_clients()
        simulation.synchronize_clients()
        simulation.synchronize_clients()

        self.assertEqual(
            [call[1]["segment_id"] for call in calls],
            [connector_id, "-37", "-37"],
        )
        self.assertEqual(calls[0][1]["lane_index"], None)
        self.assertEqual(calls[1][1]["lane_index"], 3)
        self.assertEqual(calls[2][1]["lane_index"], None)
        self.assertEqual(simulation._carla_authoritative_segments[obj], "-37")

    def test_internal_edge_publishes_canonical_connector_without_route_change(self):
        connector_id = "cont/-36/-37"
        simulation, obj, calls = self._car_4_authority_fixture(
            [(":1146_3", 9)]
        )
        original_route = list(obj.route)

        simulation.synchronize_clients()

        self.assertEqual(calls[0][1]["segment_id"], connector_id)
        self.assertIsNone(calls[0][1]["lane_index"])
        self.assertEqual(calls[0][1]["connector_path_id"], 0)
        self.assertEqual(obj.route, original_route)

    def test_unrelated_outside_road_fails_before_shadow_teleport(self):
        simulation, obj, calls = self._car_4_authority_fixture(
            [("unrelated/outside", 2)]
        )

        with self.assertRaisesRegex(
            SimulationCreationError,
            "outside its METS-R ownership domain.*allowed segments",
        ):
            simulation.synchronize_clients()

        self.assertEqual(calls, [])
        self.assertEqual(obj.carlaActor.transform_calls, [])

    def test_partial_shadow_acknowledgement_still_raises_control_error(self):
        response = metsr_response(
            "teleportCoSimVeh",
            [{"vehicleId": 5, "status": "ok", "transitionPending": True}],
            status="partial",
        )
        simulation, obj, calls = self._car_4_authority_fixture(
            [("-36", 4)], teleport_response=response
        )

        with self.assertRaises(METSRControlError):
            simulation.synchronize_clients()

        self.assertEqual(len(calls), 1)
        self.assertEqual(obj.carlaActor.transform_calls, [])

    def test_all_error_shadow_ack_reports_server_vehicle_detail(self):
        response = metsr_response(
            "teleportCoSimVeh",
            [
                {
                    "vehicleId": 5,
                    "status": "error",
                    "errorCode": "INVALID_CONNECTOR_PATH",
                    "message": "connectorPathId is outside the connector",
                }
            ],
            status="error",
        )
        simulation, obj, calls = self._car_4_authority_fixture(
            [("-36", 4)], teleport_response=response
        )

        with self.assertRaisesRegex(
            METSRControlError,
            "vehicle 5.*INVALID_CONNECTOR_PATH.*outside the connector",
        ) as raised:
            simulation.synchronize_clients()

        self.assertEqual(
            raised.exception.record["errorCode"], "INVALID_CONNECTOR_PATH"
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(obj.carlaActor.transform_calls, [])


    def test_outbound_handoff_accepts_carla_choice_outside_metsr_route(self):
        connector_id = "opaque-A-to-B"
        state = {
            "segmentId": "A",
            "laneIndex": 1,
            "destinationRoadId": "C",
        }
        simulation, obj, calls = self._sync_guard_fixture(
            state,
            "B",
            ["OTHER", "C"],
            observed_lane=2,
            controlled_roads={"A"},
            transition_result={
                "data": [
                    {
                        "segmentId": "B",
                        "laneIndex": 2,
                        "controlMode": "native",
                        "releasedFromCoSim": True,
                    }
                ]
            },
        )
        simulation.carla_control_segments = {"A", ":A_B_0"}
        simulation.metsr_internal_edge_to_connector = {
            ":A_B_0": connector_id
        }
        simulation.metsr_connector_records = {
            connector_id: {
                "connectorId": connector_id,
                "sourceRoadId": "A",
                "targetRoadId": "B",
                "internalEdgeIds": [":A_B_0"],
            }
        }
        simulation.metsr_road_connectors = {"A": {connector_id}}

        simulation.synchronize_clients()

        self.assertEqual(len(calls["teleport"]), 1)
        self.assertEqual(calls["teleport"][0][1]["segment_id"], "B")
        self.assertEqual(calls["teleport"][0][1]["lane_index"], 2)
        self.assertNotIn(obj, simulation.carla_actors)
        self.assertIn(obj, simulation.metsr_actors)
        self.assertNotIn(obj, simulation._carla_authoritative_segments)
        self.assertEqual(simulation.pending_road_entries, {})

    def test_outbound_handoff_requires_explicit_native_release_ack(self):
        connector_id = "opaque-A-to-B"
        simulation, obj, calls = self._sync_guard_fixture(
            {
                "segmentId": "A",
                "laneIndex": 1,
                "destinationRoadId": "C",
            },
            "B",
            ["A", "B", "C"],
            observed_lane=2,
            controlled_roads={"A"},
            transition_result={
                "data": [
                    {
                        "segmentId": "B",
                        "laneIndex": 2,
                        "controlMode": "cosim",
                        "releasedFromCoSim": False,
                    }
                ]
            },
        )
        simulation.carla_control_segments = {"A", ":A_B_0"}
        simulation.metsr_internal_edge_to_connector = {
            ":A_B_0": connector_id
        }
        simulation.metsr_connector_records = {
            connector_id: {
                "connectorId": connector_id,
                "sourceRoadId": "A",
                "targetRoadId": "B",
                "internalEdgeIds": [":A_B_0"],
            }
        }
        simulation.metsr_road_connectors = {"A": {connector_id}}

        with self.assertRaisesRegex(METSRControlError, "direct native handoff"):
            simulation.synchronize_clients()

        self.assertEqual(len(calls["teleport"]), 1)
        self.assertIn(obj, simulation.carla_actors)
        self.assertNotIn(obj, simulation.metsr_actors)

    def test_pcla_source_boundary_uses_canonical_connector(self):
        calls = []
        connector_id = "cont/-36/-37"
        actor = SimpleNamespace(
            get_transform=lambda: carla.Transform(
                carla.Location(), carla.Rotation(yaw=10.0)
            ),
            get_velocity=lambda: SimpleNamespace(x=2.0, y=0.0, z=0.0),
        )
        class Vehicle:
            pass

        obj = Vehicle()
        obj.name = "ego"
        obj.route = ["-37", "destination"]
        obj.carlaActor = actor
        simulation = object.__new__(CosimSimulation)
        simulation.carla_control_roads = {"-37": True}
        simulation.carla_control_segments = {"-37", ":1146_3"}
        simulation.metsr_internal_edge_to_connector = {
            ":1146_3": connector_id
        }
        simulation.metsr_connector_records = {
            connector_id: {
                "connectorId": connector_id,
                "sourceRoadId": "-36",
                "targetRoadId": "-37",
                "internalEdgeIds": [":1146_3"],
            }
        }
        simulation.metsr_road_connectors = {"-37": {connector_id}}
        simulation.road_entry_holds = {}
        simulation.road_entry_hold_diagnostics = {}
        simulation.pending_road_entries = {}
        simulation.pending_route_refreshes = {}
        simulation.pending_lane_reconciliation_verifications = {}
        simulation.same_road_departure_lane_verifications = {}
        simulation._pending_pcla_pose_resets = set()
        simulation.destination_route_cache = {}
        record = {
            "vehicleId": 5,
            "segmentId": connector_id,
            "segmentType": "connector",
            "connectorId": connector_id,
            "internalEdgeIds": [":1146_3"],
            "sourceRoadId": "-36",
            "targetRoadId": "-37",
            "transitionSourceRoadId": "-36",
            "transitionTargetRoadId": "-37",
            "transitionPending": True,
            "destinationRoadId": "destination",
        }
        vehicle_state = {
            "segmentId": "-37",
            "transitionPending": False,
            "destinationRoadId": "destination",
        }

        def teleport(*args, **kwargs):
            calls.append((args, kwargs))
            return {
                "messageType": "teleportCoSimVeh",
                "status": "ok",
                "data": [{"vehicleId": 5, "status": "ok"}],
            }

        simulation.metsr_client = SimpleNamespace(
            teleport_cosim_vehicle=teleport,
            query_route_between_roads=lambda *args: (_ for _ in ()).throw(
                AssertionError("a pending connector must not rebuild the route")
            ),
            update_vehicle_route=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("a pending connector must preserve the route")
            ),
        )

        simulation._synchronize_pcla_ego_shadow(
            obj,
            5,
            vehicle_state,
            record,
            carla.Location(x=181.81, y=136.67, z=0.0),
            connector_id,
            None,
            "-37",
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["segment_id"], connector_id)
        self.assertIsNone(calls[0][1]["lane_index"])

    def test_multi_edge_connector_always_uses_canonical_connector_id(self):
        simulation = object.__new__(CosimSimulation)
        connector_id = "cont/58/-14"
        internal_edges = [":452_4_0", ":452_14_0"]
        simulation.carla_control_roads = {"-14": True}
        simulation.carla_control_segments = {"-14", *internal_edges}
        simulation.metsr_internal_edge_to_connector = {
            edge_id: connector_id for edge_id in internal_edges
        }
        simulation.metsr_connector_records = {
            connector_id: {
                "connectorId": connector_id,
                "sourceRoadId": "58",
                "targetRoadId": "-14",
                "internalEdgeIds": internal_edges,
            }
        }
        simulation.metsr_road_connectors = {"-14": {connector_id}}
        record = {
            "segmentId": connector_id,
            "segmentType": "connector",
            "connectorId": connector_id,
            "internalEdgeIds": internal_edges,
            "sourceRoadId": "58",
            "targetRoadId": "-14",
            "transitionPending": True,
        }

        self.assertEqual(
            simulation._canonical_connector_hint(record, "58"),
            connector_id,
        )
        self.assertEqual(
            simulation._canonical_connector_hint(record, ":452_14_0"),
            connector_id,
        )
        self.assertIsNone(
            simulation._canonical_connector_hint(record, "unrelated")
        )

    def test_freeze_roads_partial_failure_raises_without_retry_or_rollback(self):
        attempts = []
        rollbacks = []
        simulation = self._connector_control_fixture({"A", "B"})

        def set_cosim_road(roads):
            attempts.append(list(roads))
            return {
                "messageType": "setCoSimRoad",
                "status": "partial",
                "data": [
                    {
                        "roadId": "A",
                        "status": "ok",
                        "connectorIds": [],
                        "connectors": [],
                    },
                    {
                        "roadId": "B",
                        "status": "error",
                        "errorCode": "FREEZE_BLOCKED",
                        "retryable": True,
                    },
                ],
            }

        simulation.metsr_client = SimpleNamespace(
            set_cosim_road=set_cosim_road,
            release_cosim_road=lambda roads: (
                rollbacks.append(list(roads))
                or {
                    "messageType": "releaseCoSimRoad",
                    "status": "ok",
                    "data": [{"roadId": "A", "status": "ok"}],
                }
            ),
        )

        with self.assertRaisesRegex(METSRControlError, "FREEZE_BLOCKED"):
            simulation.freeze_roads(["A", "B", "A"])

        self.assertEqual(simulation.carla_control_roads, {})
        self.assertEqual(simulation.carla_control_segments, set())
        self.assertEqual(attempts, [["A", "B"]])
        self.assertEqual(rollbacks, [])

    def test_freeze_rejects_missing_expected_incident_edge_without_rollback(self):
        rollbacks = []
        simulation = self._connector_control_fixture(
            {"-37"}, {":1146_3": [("-36_6", "-37_5")]}
        )
        response = self._takeover_response(
            "-37", "cont/-36/-37", [], "-36"
        )
        simulation.metsr_client = SimpleNamespace(
            set_cosim_road=lambda roads: response,
            release_cosim_road=lambda roads: (
                rollbacks.append(list(roads))
                or {
                    "messageType": "releaseCoSimRoad",
                    "status": "ok",
                    "data": [{"roadId": "-37", "status": "ok"}],
                }
            ),
        )

        with self.assertRaisesRegex(METSRControlError, "expected incident"):
            simulation.freeze_roads(["-37"])

        self.assertEqual(rollbacks, [])
        self.assertEqual(simulation.carla_control_roads, {})

    def test_freeze_roads_propagates_nonretryable_failure(self):
        simulation = self._connector_control_fixture({"A"})

        def reject(roads):
            raise METSRControlError(
                "permanent failure", record={"retryable": False}
            )

        simulation.metsr_client = SimpleNamespace(set_cosim_road=reject)

        with self.assertRaises(METSRControlError):
            simulation.freeze_roads(["A"])

        self.assertEqual(simulation.carla_control_roads, {})

    def test_release_preserves_connector_owned_by_other_endpoint(self):
        simulation = self._connector_control_fixture({"A", "B"})
        simulation.carla_control_roads = {"A": True, "B": True}
        connector_id = "cont/A/B"
        connector = {
            "connectorId": connector_id,
            "sourceRoadId": "A",
            "targetRoadId": "B",
            "internalEdgeIds": [":1146_3"],
            "controlMode": "COSIM",
        }
        simulation.carla_control_segments = {"A", "B", ":1146_3"}
        simulation.metsr_internal_edge_to_connector = {":1146_3": connector_id}
        simulation.metsr_connector_records = {connector_id: connector}
        simulation.metsr_road_connectors = {
            "A": {connector_id},
            "B": {connector_id},
        }
        released = []

        def release(road):
            released.append(road)
            if road == "A":
                record = {
                    "roadId": "A",
                    "status": "ok",
                    "releasedConnectorIds": [],
                    "releasedConnectors": [],
                    "connectorIds": [connector_id],
                    "connectors": [connector],
                }
            else:
                record = {
                    "roadId": "B",
                    "status": "ok",
                    "releasedConnectorIds": [connector_id],
                    "releasedConnectors": [connector],
                    "connectorIds": [],
                    "connectors": [],
                }
            return {
                "messageType": "releaseCoSimRoad",
                "status": "ok",
                "data": [record],
            }

        simulation.metsr_client = SimpleNamespace(
            release_cosim_road=release
        )

        simulation.release_roads(["A"])

        self.assertEqual(released, ["A"])
        self.assertEqual(simulation.carla_control_roads, {"B": True})
        self.assertEqual(
            simulation.metsr_internal_edge_to_connector,
            {":1146_3": connector_id},
        )
        self.assertIn(":1146_3", simulation.carla_control_segments)

        simulation.release_roads(["B"])

        self.assertEqual(released, ["A", "B"])
        self.assertEqual(simulation.carla_control_roads, {})
        self.assertEqual(simulation.metsr_internal_edge_to_connector, {})
        self.assertEqual(simulation.carla_control_segments, set())

    def test_release_failure_is_fatal_and_keeps_local_ownership(self):
        simulation = self._connector_control_fixture({"A"})
        simulation.carla_control_roads = {"A": True}
        simulation.carla_control_segments = {"A"}
        simulation.metsr_client = SimpleNamespace(
            release_cosim_road=lambda road: {
                "messageType": "releaseCoSimRoad",
                "status": "partial",
                "data": [
                    {
                        "roadId": "A",
                        "status": "error",
                        "errorCode": "UNEXPECTED_RELEASE_FAILURE",
                        "retryable": True,
                    }
                ],
            }
        )

        with self.assertRaisesRegex(
            METSRControlError, "UNEXPECTED_RELEASE_FAILURE"
        ):
            simulation.release_roads(["A"])

        self.assertEqual(simulation.carla_control_roads, {"A": True})
        self.assertEqual(simulation.carla_control_segments, {"A"})

    def test_exact_car_4_connector_record_promotes_by_target_road(self):
        class Vehicle:
            pass

        car_4 = Vehicle()
        car_4.name = "car_4"
        car_4.spawn_guard = 0
        car_4.carla_actor_flag = False
        simulation = object.__new__(CosimSimulation)
        simulation.objects = [SimpleNamespace(name="ego"), car_4]
        simulation.bubble_roads = []
        simulation.sim_ticks_per_carla = 1
        simulation.count = 37
        simulation.completed_route = {}
        simulation.pending_road_entries = {}
        simulation.pv_id_map = {}
        simulation.admitted_queue_vehicles = {}
        simulation.carla_control_roads = {"-37": True}
        simulation.road_pop_density = {"-37": 0}
        simulation.carla_actors = []
        simulation.metsr_actors = [car_4]
        record = {
            "vehicleId": 5,
            "segmentId": "cont/-36/-37",
            "segmentType": "connector",
            "connectorId": "cont/-36/-37",
            "internalEdgeIds": [":1146_3"],
            "observedSegmentId": ":1146_3",
            "sourceRoadId": "-36",
            "targetRoadId": "-37",
            "transitionPending": True,
            "x": 181.81,
            "y": -136.67,
        }
        simulation._collect_metsr_vehicle_data = lambda objects: {car_4: record}
        promoted = []
        simulation.createObjectInCarla = lambda obj, update_velocity=True: (
            promoted.append((obj, update_velocity)) or True
        )

        simulation.update_bubble_objects([], [])

        self.assertEqual(promoted, [(car_4, True)])
        self.assertEqual(simulation.carla_actors, [car_4])
        self.assertEqual(simulation.metsr_actors, [])

    def test_outbound_connector_stays_in_cosim_when_only_source_is_controlled(self):
        class Vehicle:
            pass

        vehicle = Vehicle()
        vehicle.name = "outbound"
        vehicle.spawn_guard = 0
        vehicle.carla_actor_flag = False
        simulation = object.__new__(CosimSimulation)
        simulation.objects = [SimpleNamespace(name="ego"), vehicle]
        simulation.bubble_roads = []
        simulation.sim_ticks_per_carla = 1
        simulation.count = 4
        simulation.completed_route = {}
        simulation.pending_road_entries = {}
        simulation.pv_id_map = {}
        simulation.admitted_queue_vehicles = {}
        simulation.carla_control_roads = {"A": True}
        simulation.carla_control_segments = {"A", ":A_B_0"}
        simulation.metsr_connector_records = {
            "cont/A/B": {
                "connectorId": "cont/A/B",
                "sourceRoadId": "A",
                "targetRoadId": "B",
                "internalEdgeIds": [":A_B_0"],
            }
        }
        simulation.metsr_road_connectors = {"A": {"cont/A/B"}}
        simulation.metsr_internal_edge_to_connector = {":A_B_0": "cont/A/B"}
        simulation.road_pop_density = {"B": 0}
        simulation.carla_actors = []
        simulation.metsr_actors = [vehicle]
        record = {
            "vehicleId": 9,
            "segmentId": "cont/A/B",
            "segmentType": "connector",
            "connectorId": "cont/A/B",
            "internalEdgeIds": [":A_B_0"],
            "observedSegmentId": ":A_B_0",
            "sourceRoadId": "A",
            "targetRoadId": "B",
            "transitionPending": False,
            "x": 5.0,
            "y": 6.0,
        }
        simulation._collect_metsr_vehicle_data = lambda objects: {vehicle: record}
        promoted = []
        simulation.createObjectInCarla = lambda obj, update_velocity=True: (
            promoted.append(obj) or True
        )

        simulation.update_bubble_objects([], [])

        self.assertEqual(promoted, [vehicle])
        self.assertEqual(simulation.carla_actors, [vehicle])
        self.assertEqual(simulation.metsr_actors, [])

    def test_bubble_update_never_demotes_from_shadow_road_alone(self):
        class Vehicle:
            pass

        obj = Vehicle()
        obj.name = "npc"
        obj.position = (0.0, 0.0)
        obj.spawn_guard = 0
        obj.carla_actor_flag = True
        simulation = object.__new__(CosimSimulation)
        simulation.objects = [SimpleNamespace(name="ego"), obj]
        simulation.bubble_roads = []
        simulation.sim_ticks_per_carla = 1
        simulation.count = 8
        simulation.completed_route = {}
        simulation.pending_road_entries = {}
        simulation.carla_control_roads = {"A": True}
        simulation.road_pop_density = {"N": 0}
        simulation.pv_id_map = {}
        simulation.admitted_queue_vehicles = {}
        simulation._collect_metsr_vehicle_data = lambda objects: {
            obj: {
                "segmentId": "N",
                "transitionPending": False,
                "x": 1.0,
                "y": 2.0,
            }
        }
        demoted = []
        simulation.remove_bubble_object = lambda target: demoted.append(target)

        simulation.update_bubble_objects([], [], allow_demotion=False)
        self.assertEqual(demoted, [])

        simulation.update_bubble_objects(
            [],
            [],
            allow_promotion=False,
            advance_spawn_guards=False,
            count_density=False,
        )
        self.assertEqual(demoted, [])

    def test_step_synchronizes_before_releasing_old_roads(self):
        simulation = object.__new__(CosimSimulation)
        simulation.valid_metsr_roads = []
        simulation.ego = SimpleNamespace(x=0.0, y=0.0, bubble=None)
        simulation.bubble_size = 25.0
        simulation._get_bubble_roads = lambda region: []
        simulation.classify_bubble_roads = lambda roads: (["new"], ["old"])
        events = []
        simulation.freeze_roads = lambda roads: events.append(("freeze", list(roads)))
        simulation.release_roads = lambda roads: events.append(("release", list(roads)))
        simulation._synchronize_boundary_vehicles = lambda: events.append(("boundary", None))
        simulation._service_metsr_entering_queues = lambda: events.append(("queues", None))
        simulation.get_bubble_intersections = lambda **kwargs: []
        simulation.update_bubble_objects = lambda *args, **kwargs: events.append(("bubble", None))
        simulation.tick_carla = lambda: events.append(("carla", None))
        simulation.synchronize_clients = lambda **kwargs: events.append(("sync", None))
        simulation._report_road_entry_holds = lambda: None
        simulation.tick_metsr = lambda: None
        vehicle_queries = []
        simulation._collect_metsr_vehicle_data = (
            lambda objects: vehicle_queries.append(tuple(objects)) or {}
        )
        simulation.metsr_actors = []
        simulation.carla_actors = []
        simulation.render = False
        simulation.bubble_sizes = []
        simulation.objects = [simulation.ego]
        simulation.frozen_vehicles = set()
        simulation.bubble_spawn_queue = []
        simulation.total_active_vehicles = []
        simulation.completed_route = {}
        simulation.count = 1
        simulation.run_name = None

        simulation.step()

        self.assertEqual(
            events,
            [
                ("freeze", ["new"]), ("boundary", None), ("queues", None),
                ("bubble", None), ("carla", None), ("sync", None),
                ("bubble", None), ("release", ["old"]),
            ],
        )
        self.assertEqual(len(vehicle_queries), 2)

    def test_old_road_is_retained_until_carla_actor_advances(self):
        class Vehicle:
            pass

        obj = Vehicle()
        simulation = object.__new__(CosimSimulation)
        simulation.carla_actors = [obj]
        simulation.carla_control_roads = {"old": True, "new": True}
        simulation.carla_control_segments = {"old", "new"}
        simulation.metsr_internal_edge_to_connector = {}
        simulation.metsr_connector_records = {}
        simulation.metsr_road_connectors = {}
        simulation.pending_road_entries = {}
        simulation._carla_authoritative_segments = {obj: "old"}

        self.assertEqual(simulation._releasable_old_roads(["old"]), [])

        simulation._carla_authoritative_segments[obj] = "new"
        self.assertEqual(simulation._releasable_old_roads(["old"]), ["old"])

    def test_connector_occupancy_retains_both_controlled_endpoints(self):
        class Vehicle:
            pass

        obj = Vehicle()
        connector_id = "opaque/A/B"
        simulation = object.__new__(CosimSimulation)
        simulation.carla_actors = [obj]
        simulation.carla_control_roads = {"A": True, "B": True, "old": True}
        simulation.carla_control_segments = {"A", "B", "old", ":A_B_0"}
        simulation.metsr_internal_edge_to_connector = {":A_B_0": connector_id}
        simulation.metsr_connector_records = {
            connector_id: {
                "connectorId": connector_id,
                "sourceRoadId": "A",
                "targetRoadId": "B",
                "internalEdgeIds": [":A_B_0"],
            }
        }
        simulation.metsr_road_connectors = {
            "A": {connector_id},
            "B": {connector_id},
        }
        simulation.pending_road_entries = {}
        simulation._carla_authoritative_segments = {obj: connector_id}

        self.assertEqual(
            simulation._releasable_old_roads(["A", "B", "old"]),
            ["old"],
        )

    def test_unresolved_carla_actor_blocks_old_road_release(self):
        obj = object()
        simulation = object.__new__(CosimSimulation)
        simulation.carla_actors = [obj]
        simulation.carla_control_roads = {"A": True, "B": True}
        simulation.carla_control_segments = {"A", "B"}
        simulation.metsr_internal_edge_to_connector = {}
        simulation.metsr_connector_records = {}
        simulation.metsr_road_connectors = {}
        simulation.pending_road_entries = {}
        simulation._carla_authoritative_segments = {}

        self.assertEqual(simulation._releasable_old_roads(["A", "B"]), [])

    def test_spawn_ego_calls_takeover_once(self):
        simulation = object.__new__(CosimSimulation)
        simulation.bubble_size = 25.0
        simulation._get_bubble_roads = lambda: []
        simulation.classify_bubble_roads = lambda roads: (["A", "B"], [])
        events = []

        simulation.freeze_roads = lambda roads: events.append(
            ("freeze", set(roads))
        )
        simulation.metsr_client = SimpleNamespace(
            tick=lambda: events.append(("tick", None))
        )
        simulation.createObjectInMetsr = lambda obj: events.append(("metsr", None))
        simulation.createObjectInCarla = (
            lambda obj: events.append(("carla", None)) or True
        )
        obj = SimpleNamespace(position=SimpleNamespace(x=1.0, y=2.0))

        simulation.spawn_ego(obj)

        self.assertEqual(
            events,
            [
                ("freeze", {"A", "B"}),
                ("tick", None),
                ("metsr", None),
                ("tick", None),
                ("carla", None),
            ],
        )

    def test_spawn_ego_immediate_freeze_keeps_one_initial_tick(self):
        simulation = object.__new__(CosimSimulation)
        simulation.bubble_size = 25.0
        simulation._get_bubble_roads = lambda: []
        simulation.classify_bubble_roads = lambda roads: (["A"], [])
        events = []
        simulation.freeze_roads = (
            lambda roads: events.append(("freeze", set(roads))) or set()
        )
        simulation.metsr_client = SimpleNamespace(
            tick=lambda: events.append(("tick", None))
        )
        simulation.createObjectInMetsr = lambda obj: events.append(("metsr", None))
        simulation.createObjectInCarla = (
            lambda obj: events.append(("carla", None)) or True
        )
        obj = SimpleNamespace(position=SimpleNamespace(x=1.0, y=2.0))

        simulation.spawn_ego(obj)

        self.assertEqual(
            events,
            [
                ("freeze", {"A"}),
                ("tick", None),
                ("metsr", None),
                ("tick", None),
                ("carla", None),
            ],
        )

    def test_spawn_ego_propagates_takeover_failure_without_retry(self):
        simulation = object.__new__(CosimSimulation)
        simulation.bubble_size = 25.0
        simulation._get_bubble_roads = lambda: []
        simulation.classify_bubble_roads = lambda roads: (["A"], [])
        freeze_calls = []
        ticks = []
        created = []
        def reject(roads):
            freeze_calls.append(set(roads))
            raise METSRControlError("takeover failed")

        simulation.freeze_roads = reject
        simulation.metsr_client = SimpleNamespace(tick=lambda: ticks.append(True))
        simulation.createObjectInMetsr = lambda obj: created.append("metsr")
        simulation.createObjectInCarla = lambda obj: created.append("carla") or True
        obj = SimpleNamespace(position=SimpleNamespace(x=1.0, y=2.0))

        with self.assertRaisesRegex(METSRControlError, "takeover failed"):
            simulation.spawn_ego(obj)

        self.assertEqual(ticks, [])
        self.assertEqual(freeze_calls, [{"A"}])
        self.assertEqual(created, [])

    def test_initiate_autopilot_uses_configured_tm_and_allows_lane_changes(self):
        events = []

        class Actor:
            def set_autopilot(self, enabled, port):
                events.append(("autopilot", enabled, port))

        actor = Actor()
        simulation = object.__new__(CosimSimulation)
        simulation.tm = SimpleNamespace(
            get_port=lambda: 8123,
            auto_lane_change=lambda target, enabled: events.append(
                ("lane_change", target, enabled)
            ),
        )
        obj = SimpleNamespace(
            carla_actor_flag=True,
            carlaActor=actor,
            active_autopilot=False,
            autopilot_action=False,
        )

        self.assertTrue(simulation.initiate_autopilot(obj))

        self.assertEqual(
            events,
            [("autopilot", True, 8123), ("lane_change", actor, True)],
        )
        self.assertTrue(obj.active_autopilot)
        self.assertTrue(obj.autopilot_action)

    def test_road_entry_hold_preserves_tm_registration_and_custom_path(self):
        events = []
        refreshed_path = [object(), object()]

        class Actor:
            def set_target_velocity(self, velocity):
                events.append(("target_velocity", velocity))

            def enable_constant_velocity(self, velocity):
                events.append(("constant_velocity", True, velocity))

            def disable_constant_velocity(self):
                events.append(("constant_velocity", False))

            def apply_control(self, control):
                events.append(("control", control))

            def set_autopilot(self, enabled, port):
                events.append(("autopilot", enabled, port))

        class Vehicle:
            pass

        actor = Actor()
        obj = Vehicle()
        obj.carlaActor = actor
        obj.active_autopilot = True
        simulation = object.__new__(CosimSimulation)
        simulation.road_entry_holds = {}
        simulation.tm = SimpleNamespace(
            get_port=lambda: 8123,
            auto_lane_change=lambda target, enabled: events.append(
                ("lane_change", target, enabled)
            ),
            set_path=lambda target, path: events.append(
                ("path", target, path)
            ),
        )
        simulation.metsr_trajectory_to_carla = lambda target: (
            events.append(("route", target)) or refreshed_path
        )

        simulation._hold_for_road_entry(obj)
        simulation._release_road_entry_hold(obj)

        self.assertFalse(any(event[0] == "autopilot" for event in events))
        self.assertEqual(
            [event[0] for event in events],
            [
                "target_velocity",
                "control",
                "constant_velocity",
                "route",
                "path",
                "lane_change",
                "constant_velocity",
                "control",
            ],
        )
        self.assertEqual(events[4], ("path", actor, refreshed_path))
        self.assertIs(obj.trajectory, refreshed_path)
        self.assertNotIn(obj, simulation.road_entry_holds)

    def test_road_entry_hold_release_uses_validated_live_route_tail(self):
        events = []
        refreshed_path = [object()]

        class Actor:
            def disable_constant_velocity(self):
                events.append("constant_velocity_off")

            def apply_control(self, control):
                events.append("brake_clear")

        class HashableNamespace(SimpleNamespace):
            __hash__ = object.__hash__

        obj = HashableNamespace(
            carlaActor=Actor(),
            active_autopilot=True,
            # This immutable route is intentionally stale.
            route=["OLD", "A", "B", "C"],
        )
        simulation = object.__new__(CosimSimulation)
        simulation.count = 8
        simulation.road_entry_holds = {obj: True}
        simulation.road_entry_hold_diagnostics = {}
        simulation.metsr_trajectory_to_carla = lambda target: (_ for _ in ()).throw(
            AssertionError("release must not query/fall back to the stale route")
        )

        def generate(route, target):
            events.append(("generate", list(route), target))
            return refreshed_path

        simulation.generate_carla_trajectory = generate
        simulation.tm = SimpleNamespace(
            set_path=lambda actor, path: events.append(("path", actor, path)),
            auto_lane_change=lambda actor, enabled: events.append(
                ("lane_change", actor, enabled)
            ),
        )

        simulation._release_road_entry_hold(
            obj, remaining_route=("B", "C")
        )

        self.assertEqual(events[0], ("generate", ["B", "C"], obj))
        self.assertEqual(events[1], ("path", obj.carlaActor, refreshed_path))
        self.assertEqual(events[2], ("lane_change", obj.carlaActor, False))
        self.assertEqual(events[3:], ["constant_velocity_off", "brake_clear"])
        self.assertIs(obj.trajectory, refreshed_path)
        self.assertNotIn(obj, simulation.road_entry_holds)

    def test_road_entry_hold_diagnostics_report_start_wait_and_release(self):
        class Actor:
            def set_target_velocity(self, velocity):
                pass

            def enable_constant_velocity(self, velocity):
                pass

            def disable_constant_velocity(self):
                pass

            def apply_control(self, control):
                pass

        class Vehicle:
            name = "car_77"

        obj = Vehicle()
        obj.carlaActor = Actor()
        obj.active_autopilot = False
        simulation = object.__new__(CosimSimulation)
        simulation.count = 12
        simulation.road_entry_holds = {}
        simulation.road_entry_hold_diagnostics = {}

        output = StringIO()
        with redirect_stdout(output):
            simulation._hold_for_road_entry(
                obj, source="-50", target="-51", reason="ENTRY_BLOCKED"
            )
            simulation.count = 112
            simulation._hold_for_road_entry(
                obj, source="-50", target="-51",
                reason="TARGET_LANE_RESERVED"
            )
            simulation._report_road_entry_holds()
            simulation._report_road_entry_holds()
            simulation.count = 212
            simulation._hold_for_road_entry(
                obj, source=None, target=None, reason=None
            )
            simulation._report_road_entry_holds()
            simulation.count = 237
            simulation._release_road_entry_hold(obj)

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn(
            "hold started: car_77 at step 12; edge -50->-51; "
            "reason=ENTRY_BLOCKED; retries=0",
            lines[0],
        )
        self.assertIn(
            "hold waiting: car_77 at step 112; held_ticks=100; retries=1",
            lines[1],
        )
        self.assertIn(
            "hold waiting: car_77 at step 212; held_ticks=200; retries=2",
            lines[2],
        )
        self.assertIn("reason=TARGET_LANE_RESERVED", lines[2])
        self.assertIn(
            "hold released: car_77 at step 237; held_ticks=225; retries=2",
            lines[3],
        )
        self.assertNotIn(obj, simulation.road_entry_holds)
        self.assertNotIn(obj, simulation.road_entry_hold_diagnostics)

    def test_removing_object_clears_road_entry_hold_diagnostics(self):
        class Vehicle:
            pass

        obj = Vehicle()
        obj.autopilot_action = False
        obj.active_autopilot = False
        obj.carla_actor_flag = True
        simulation = object.__new__(CosimSimulation)
        simulation.carla_actors = [obj]
        simulation.metsr_actors = []
        simulation.pending_road_entries = {obj: {"target": "B"}}
        simulation.road_entry_holds = {obj: True}
        simulation.road_entry_hold_diagnostics = {obj: {"start_step": 3}}
        simulation.pending_route_refreshes = {obj: "B"}
        simulation._carla_authoritative_segments = {obj: "A"}

        simulation.remove_bubble_object(obj, destroy=False)

        self.assertNotIn(obj, simulation.road_entry_hold_diagnostics)
        self.assertNotIn(obj, simulation._carla_authoritative_segments)

    def test_failed_carla_spawn_clears_authoritative_segment(self):
        class Vehicle:
            pass

        class Blueprint:
            @staticmethod
            def has_attribute(name):
                return False

        obj = Vehicle()
        obj.name = "queued_car"
        obj.blueprint = "vehicle.test"
        obj.rolename = None
        obj.position = SimpleNamespace(x=1.0, y=2.0, z=0.0)
        obj.orientation = object()
        obj.snapToGround = False
        obj.color = None
        simulation = object.__new__(CosimSimulation)
        simulation._carla_authoritative_segments = {obj: "A"}
        simulation.bubble_spawn_queue = {obj}
        simulation.blueprintLib = SimpleNamespace(find=lambda name: Blueprint())
        simulation.carla_world = SimpleNamespace(
            try_spawn_actor=lambda blueprint, transform: None
        )

        with patch(
            "scenic.simulators.cosim.simulator.utils.scenicToCarlaLocation",
            return_value=carla.Location(x=1.0, y=2.0, z=0.0),
        ), patch(
            "scenic.simulators.cosim.simulator.utils.scenicToCarlaRotation",
            return_value=carla.Rotation(),
        ):
            self.assertFalse(simulation.createObjectInCarla(obj))

        self.assertNotIn(obj, simulation._carla_authoritative_segments)

    def test_road_entry_hold_route_failure_keeps_actor_stopped(self):
        events = []

        class Actor:
            def disable_constant_velocity(self):
                events.append("released")

            def apply_control(self, control):
                events.append("control")

        class Vehicle:
            pass

        actor = Actor()
        obj = Vehicle()
        obj.carlaActor = actor
        simulation = object.__new__(CosimSimulation)
        simulation.count = 17
        simulation.road_entry_holds = {obj: True}
        simulation.road_entry_hold_diagnostics = {
            obj: {
                "start_step": 10,
                "retries": 2,
                "source": "A",
                "target": "B",
                "latest_reason": "ENTRY_BLOCKED",
                "next_report_step": 110,
            }
        }
        simulation.tm = SimpleNamespace(
            set_path=lambda target, path: events.append("path"),
            auto_lane_change=lambda target, enabled: events.append(
                "lane_change"
            ),
        )

        def fail_route(target):
            raise SimulationCreationError("route refresh failed")

        simulation.metsr_trajectory_to_carla = fail_route

        with self.assertRaisesRegex(SimulationCreationError, "route refresh failed"):
            simulation._release_road_entry_hold(obj)

        self.assertEqual(events, [])
        self.assertIn(obj, simulation.road_entry_holds)
        self.assertIn(obj, simulation.road_entry_hold_diagnostics)

    def test_default_trajectory_queries_private_vehicle_destination(self):
        class Vehicle:
            route = ["STALE"]

        simulation = object.__new__(CosimSimulation)
        obj = Vehicle()
        calls = []
        simulation.getMetsrPrivateVehId = lambda target: 87

        def query_vehicle(vehicle_id, **kwargs):
            calls.append((vehicle_id, kwargs))
            return metsr_response("vehicle", [{
                "vehicleId": 87, "segmentId": "A", "destinationRoadId": "N",
            }])

        simulation.metsr_client = SimpleNamespace(query_vehicle=query_vehicle)
        path = [object()]
        simulation.generate_carla_destination_trajectory = (
            lambda destination, target: calls.append((destination, target)) or path
        )
        self.assertIs(simulation.metsr_trajectory_to_carla(obj), path)
        self.assertEqual(calls, [
            (87, {"private_veh": True, "transform_coords": True}), ("N", obj),
        ])
        self.assertEqual(simulation._carla_destination_plans[obj], ("N", path))

    def test_default_trajectory_requires_queried_destination(self):
        simulation = object.__new__(CosimSimulation)
        obj = SimpleNamespace(route=["STALE", "Z"])
        simulation.getMetsrPrivateVehId = lambda target: 87
        simulation.metsr_client = SimpleNamespace(
            query_vehicle=lambda *args, **kwargs: metsr_response(
                "vehicle", [{"vehicleId": 87, "segmentId": "A"}]
            ),
        )
        with self.assertRaisesRegex(SimulationCreationError, "omitted destinationRoadId"):
            simulation.metsr_trajectory_to_carla(obj)

    def test_default_trajectory_ignores_remaining_route_and_pending_fields(self):
        class Vehicle:
            route = ["STALE", "WRONG"]

        simulation = object.__new__(CosimSimulation)
        obj = Vehicle()
        simulation.getMetsrPrivateVehId = lambda target: 87
        simulation.metsr_client = SimpleNamespace(
            query_vehicle=lambda *args, **kwargs: metsr_response("vehicle", [{
                "vehicleId": 87, "destinationRoadId": "N",
                "routeRoadIds": ["T"], "transitionPending": True,
                "transitionTargetRoadId": "WRONG", "transitionTargetLaneIndex": 0,
            }]),
        )
        calls = []
        expected = [object()]
        simulation.generate_carla_destination_trajectory = (
            lambda destination, target: calls.append((destination, target)) or expected
        )
        self.assertIs(simulation.metsr_trajectory_to_carla(obj), expected)
        self.assertEqual(calls, [("N", obj)])

    def _predecessor_anchor_fixture(self, terminal_lane="T_0"):
        simulation = object.__new__(CosimSimulation)
        source = self._route_waypoint(1, -1, 9.0, 0.0)
        target = self._route_waypoint(2, -1, 0.1, 1.0)
        alternate = self._route_waypoint(2, -2, 0.1, 1.0)
        terminal = target if terminal_lane == "T_0" else alternate
        simulation.map = SimpleNamespace(get_waypoint=lambda *args, **kwargs: source)
        simulation.grp = SimpleNamespace(
            _sampling_resolution=2.0,
            trace_route=lambda start, end: [(terminal, None)],
        )
        simulation.scenic_to_metsr_map = {
            "1_-1": ["P_0"],
            "2_-1": ["T_0"],
            "2_-2": ["T_1"],
        }
        simulation.metsr_lane_indices = {"P_0": 0, "T_0": 0, "T_1": 1}
        simulation.metsr_lane_connections = {
            ("P", "T"): [("P_0", "T_0")]
        }
        simulation._index_carla_waypoints = lambda: None
        simulation._lane_anchor_waypoint = lambda lane: target
        simulation._validate_carla_route_trace = lambda *args, **kwargs: None
        simulation.metsr_road_cache = {
            ("T", 0): [(1.0, 0.0, 0.0), (1.85, 0.0, 0.0)]
        }
        return simulation, source, target

    def test_internal_connector_uses_exact_target_lane_anchor(self):
        simulation, source, target = self._predecessor_anchor_fixture()
        internal = self._route_waypoint(680, -4, 0.0, 0.0)
        internal.is_junction = True
        simulation.map = SimpleNamespace(
            get_waypoint=lambda *args, **kwargs: internal
        )
        simulation.scenic_to_metsr_map = {
            "680_-4": [":J_5_1"],
            "2_-1": ["T_0"],
            "2_-2": ["T_1"],
        }
        simulation.metsr_internal_lane_connections = {
            ":J_5_1": [("P_0", "T_0")]
        }

        anchor, trace = simulation._forward_lane_anchor_trace(
            "T_0", internal.transform.location, "T", 1
        )

        self.assertIs(anchor, target)
        self.assertIs(trace[-1][0], target)

    def test_short_target_uses_exact_predecessor_lane_anchor(self):
        simulation, source, target = self._predecessor_anchor_fixture()

        anchor, trace = simulation._forward_lane_anchor_trace(
            "T_0", source.transform.location, "T", 1
        )

        self.assertIs(anchor, target)
        self.assertIs(trace[-1][0], target)

    def test_predecessor_anchor_rejects_alternate_terminal_lane(self):
        simulation, source, _ = self._predecessor_anchor_fixture("T_1")

        with self.assertRaisesRegex(
            SimulationCreationError, "did not reach exact SUMO target lane T_0"
        ):
            simulation._forward_lane_anchor_trace(
                "T_0", source.transform.location, "T", 1
            )
    def test_synchronization_publishes_without_live_private_record(self):
        state = {
            "segmentId": "A",
            "destinationRoadId": "Z",
            "transitionPending": False,
            "laneIndex": 1,
        }
        simulation, _, calls = self._sync_guard_fixture(
            state, "A", ["A", "B"], cosim_records=[]
        )

        simulation.synchronize_clients()

        self.assertEqual(len(calls["teleport"]), 1)
        self.assertEqual(calls["teleport"][0][1]["segment_id"], "A")

    def test_synchronization_rejects_duplicate_live_private_records(self):
        state = {
            "segmentId": "A",
            "destinationRoadId": "Z",
            "transitionPending": False,
            "laneIndex": 1,
        }
        records = [
            {"vehicleId": 7, "isPrivate": True, "routeRoadIds": ["A", "B"]},
            {"vehicleId": 7, "isPrivate": True, "routeRoadIds": ["A", "C"]},
        ]
        simulation, _, calls = self._sync_guard_fixture(
            state, "A", ["A", "B"], cosim_records=records
        )

        with self.assertRaisesRegex(
            SimulationCreationError, "during synchronization; found 2"
        ):
            simulation.synchronize_clients()

        self.assertEqual(calls["teleport"], [])

    def test_synchronization_queries_missing_live_route_from_destination(self):
        state = {
            "segmentId": "A",
            "destinationRoadId": "Z",
            "transitionPending": False,
            "laneIndex": 1,
        }
        simulation, _, calls = self._sync_guard_fixture(
            state,
            "A",
            ["A", "B"],
            cosim_records=[{"vehicleId": 7, "isPrivate": True, "routeRoadIds": []}],
        )

        simulation.synchronize_clients()

        self.assertEqual(calls["route_query"], [])
        self.assertEqual(len(calls["teleport"]), 1)

    def test_completed_vehicle_does_not_require_nonempty_live_route(self):
        state = {
            "segmentId": "Z",
            "destinationRoadId": "Z",
            "transitionPending": False,
        }
        simulation, obj, calls = self._sync_guard_fixture(
            state, "Z", ["Z"]
        )

        simulation.synchronize_clients()

        self.assertTrue(simulation.completed_route[obj])
        self.assertEqual(len(calls["teleport"]), 1)

    def test_hold_control_failure_is_not_recorded_as_safe(self):
        class Actor:
            def set_target_velocity(self, velocity):
                pass

            def apply_control(self, control):
                pass

            def enable_constant_velocity(self, velocity):
                raise RuntimeError("CARLA refused constant velocity")

        class Vehicle:
            pass

        obj = Vehicle()
        obj.name = "unsafe-hold"
        obj.carlaActor = Actor()
        obj.active_autopilot = True
        simulation = object.__new__(CosimSimulation)
        simulation.count = 8
        simulation.road_entry_holds = {}
        simulation.road_entry_hold_diagnostics = {}

        with self.assertRaisesRegex(
            SimulationCreationError, "Unable to place.*safety hold"
        ):
            simulation._hold_for_road_entry(obj, "A", "B", "ENTRY_BLOCKED")

        self.assertIn(obj, simulation.road_entry_holds)
        self.assertIn(obj, simulation.road_entry_hold_diagnostics)

    def test_release_control_failure_keeps_hold_and_diagnostic(self):
        class Actor:
            def disable_constant_velocity(self):
                raise RuntimeError("CARLA refused release")

            def apply_control(self, control):
                self.fail("brake must not be cleared after disable failure")

        class Vehicle:
            pass

        obj = Vehicle()
        obj.name = "unsafe-release"
        obj.carlaActor = Actor()
        simulation = object.__new__(CosimSimulation)
        simulation.count = 12
        simulation.road_entry_holds = {obj: False}
        simulation.road_entry_hold_diagnostics = {
            obj: {
                "start_step": 10,
                "retries": 1,
                "source": "A",
                "target": "B",
                "latest_reason": "ENTRY_BLOCKED",
                "next_report_step": 110,
            }
        }

        with self.assertRaisesRegex(
            SimulationCreationError, "Unable to physically release"
        ):
            simulation._release_road_entry_hold(obj)

        self.assertIn(obj, simulation.road_entry_holds)
        self.assertIn(obj, simulation.road_entry_hold_diagnostics)

    def test_carla_tick_reasserts_hold_before_every_physics_step(self):
        events = []

        class Actor:
            def __init__(self):
                self.control = carla.VehicleControl(
                    manual_gear_shift=True, gear=1
                )

            def get_control(self):
                return self.control

            def set_target_velocity(self, velocity):
                events.append(("velocity", velocity.x, velocity.y, velocity.z))

            def set_target_angular_velocity(self, velocity):
                events.append(("angular", velocity.x, velocity.y, velocity.z))

            def apply_control(self, control):
                self.control = control
                events.append(
                    (
                        "control",
                        control.throttle,
                        control.steer,
                        control.brake,
                        control.hand_brake,
                        control.reverse,
                        control.manual_gear_shift,
                        control.gear,
                    )
                )

            def enable_constant_velocity(self, velocity):
                events.append(("constant", velocity.x, velocity.y, velocity.z))

        class Vehicle:
            pass

        obj = Vehicle()
        obj.carlaActor = Actor()
        simulation = object.__new__(CosimSimulation)
        simulation.road_entry_holds = {obj: False}
        simulation.sim_ticks_per_carla = 2
        simulation.carla_world = SimpleNamespace(
            tick=lambda: events.append(("tick",))
        )

        simulation.tick_carla()

        hold = [
            ("velocity", 0.0, 0.0, 0.0),
            ("angular", 0.0, 0.0, 0.0),
            ("control", 0.0, 0.0, 1.0, True, False, True, 1),
            ("constant", 0.0, 0.0, 0.0),
        ]
        self.assertEqual(events, hold + [("tick",)] + hold + [("tick",)])

    def test_hold_and_release_notify_pcla_of_controls_applied_to_carla(self):
        notified_controls = []

        class Actor:
            def __init__(self):
                self.control = carla.VehicleControl(
                    reverse=False,
                    manual_gear_shift=True,
                    gear=1,
                )

            def get_control(self):
                return self.control

            def set_target_velocity(self, velocity):
                pass

            def set_target_angular_velocity(self, velocity):
                pass

            def apply_control(self, control):
                self.control = control

            def enable_constant_velocity(self, velocity):
                pass

            def disable_constant_velocity(self):
                pass

        class PCLA:
            def notify_control_applied(self, control):
                notified_controls.append(control)

        class Vehicle:
            pass

        obj = Vehicle()
        obj.name = "ego"
        obj.carlaActor = Actor()
        obj.active_autopilot = False
        obj.pcla = PCLA()
        simulation = object.__new__(CosimSimulation)
        simulation.count = 14
        simulation.road_entry_holds = {}
        simulation.road_entry_hold_diagnostics = {}

        with redirect_stdout(StringIO()):
            simulation._hold_for_road_entry(obj, "-29", "27", "PREPARATION")
            simulation._release_road_entry_hold(obj)

        self.assertEqual(len(notified_controls), 2)
        hold_control, release_control = notified_controls
        self.assertEqual(
            (
                hold_control.throttle,
                hold_control.steer,
                hold_control.brake,
                hold_control.hand_brake,
                hold_control.manual_gear_shift,
                hold_control.gear,
            ),
            (0.0, 0.0, 1.0, True, True, 1),
        )
        self.assertEqual(
            (
                release_control.throttle,
                release_control.brake,
                release_control.hand_brake,
                release_control.manual_gear_shift,
                release_control.gear,
            ),
            (0.0, 0.0, False, True, 1),
        )

    def test_pose_reset_is_deferred_until_verified_hold_release(self):
        pose_resets = []

        class Actor:
            def __init__(self):
                self.control = carla.VehicleControl(
                    manual_gear_shift=True, gear=1
                )

            def get_control(self):
                return self.control

            def apply_control(self, control):
                self.control = control

            def disable_constant_velocity(self):
                pass

        class PCLA:
            def notify_pose_changed(self):
                pose_resets.append("fresh-frame-reset")

        class Vehicle:
            pass

        obj = Vehicle()
        obj.name = "ego"
        obj.carlaActor = Actor()
        obj.pcla = PCLA()
        simulation = object.__new__(CosimSimulation)
        simulation.count = 30
        simulation.road_entry_holds = {obj: False}
        simulation.road_entry_hold_diagnostics = {}
        simulation._pending_pcla_pose_resets = set()

        simulation._defer_pcla_pose_reset(obj)

        self.assertEqual(pose_resets, [])
        self.assertIn(obj, simulation._pending_pcla_pose_resets)

        with redirect_stdout(StringIO()):
            simulation._release_road_entry_hold(obj)

        self.assertEqual(pose_resets, ["fresh-frame-reset"])
        self.assertNotIn(obj, simulation._pending_pcla_pose_resets)

    def test_pose_change_resets_legacy_pcla_localization_state(self):
        agent = SimpleNamespace(
            filter_initialized=True,
            state_log=["old-filter-state"],
        )
        obj = SimpleNamespace(pcla=SimpleNamespace(agent_instance=agent))

        CosimSimulation._notify_pcla_pose_changed(obj)

        self.assertFalse(agent.filter_initialized)
        self.assertEqual(agent.state_log, [])

    def test_held_pcla_action_is_discarded_until_fresh_action_after_release(self):
        events = []

        class Actor:
            def __init__(self, name):
                self.name = name

            def apply_control(self, control):
                events.append(("applied", self.name, control))

        class Vehicle:
            pass

        class ControlAction:
            def __init__(self, name):
                self.name = name

            def applyTo(self, obj, simulation):
                events.append(("action", self.name))
                obj._control = self.name

        held = Vehicle()
        held.carlaActor = Actor("held")
        held.carla_actor_flag = True
        held.autopilot_action = False
        held.active_autopilot = False
        held._control = "stale-control"

        free = Vehicle()
        free.carlaActor = Actor("free")
        free.carla_actor_flag = True
        free.autopilot_action = False
        free.active_autopilot = False
        free._control = None
        free.pcla = SimpleNamespace(
            notify_control_applied=lambda control: events.append(
                ("notified", "free", control)
            )
        )

        simulation = object.__new__(CosimSimulation)
        simulation.agents = [held, free]
        simulation.road_entry_holds = {held: False}

        simulation.executeActions(
            {
                held: (ControlAction("held-during-hold"),),
                free: (ControlAction("free-control"),),
            }
        )

        self.assertEqual(
            events,
            [
                ("action", "free-control"),
                ("applied", "free", "free-control"),
                ("notified", "free", "free-control"),
            ],
        )
        self.assertIsNone(held._control)

        simulation.road_entry_holds.clear()
        simulation.executeActions(
            {
                held: (ControlAction("held-after-release"),),
                free: (),
            }
        )

        self.assertEqual(
            events[-2:],
            [
                ("action", "held-after-release"),
                ("applied", "held", "held-after-release"),
            ],
        )
        self.assertNotIn("held-during-hold", [event[-1] for event in events])

    def test_ordinary_carla_updates_use_one_batched_teleport(self):
        class Vehicle:
            pass

        first = Vehicle()
        second = Vehicle()
        calls = []
        simulation = object.__new__(CosimSimulation)
        simulation.carla_control_roads = {"A": True, "B": True}
        simulation.completed_route = {}
        simulation.count = 4

        def teleport(*args, **kwargs):
            calls.append((args, kwargs))
            return metsr_response(
                "teleportCoSimVeh",
                [
                    {"vehicleId": 12, "segmentId": "B", "laneIndex": 2},
                    {"vehicleId": 11, "segmentId": "A", "laneIndex": 1},
                ],
            )

        simulation.metsr_client = SimpleNamespace(
            teleport_cosim_vehicle=teleport
        )
        updates = [
            {
                "object": first,
                "vehicle_id": 11,
                "x": 1.0,
                "y": 2.0,
                "z": 0.0,
                "bearing": 10.0,
                "speed": 3.0,
                "private_veh": True,
                "transform_coords": True,
                "authoritative_segment": "A",
                "authoritative_lane": 1,
                "logical_road": "A",
                "vehicle_state": {"destinationRoadId": "Z"},
                "cosim_record": {},
            },
            {
                "object": second,
                "vehicle_id": 12,
                "x": 4.0,
                "y": 5.0,
                "z": 0.0,
                "bearing": 20.0,
                "speed": 6.0,
                "private_veh": True,
                "transform_coords": True,
                "authoritative_segment": "B",
                "authoritative_lane": 2,
                "logical_road": "B",
                "vehicle_state": {"destinationRoadId": "Z"},
                "cosim_record": {},
            },
        ]

        simulation._publish_carla_vehicle_updates(updates)

        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(args, ([11, 12], [1.0, 4.0], [2.0, 5.0]))
        self.assertEqual(kwargs["segment_id"], ["A", "B"])
        self.assertEqual(kwargs["lane_index"], [1, 2])
        self.assertEqual(kwargs["speed"], [3.0, 6.0])

    def test_pcla_ego_same_road_sync_only_updates_metsr_shadow(self):
        teleport_calls = []

        class Actor:
            def __init__(self):
                self.location = carla.Location(x=10.0, y=-2.0, z=0.5)
                self.transform = carla.Transform(
                    self.location, carla.Rotation(yaw=15.0)
                )

            def get_location(self):
                return self.location

            def get_transform(self):
                return self.transform

            def get_velocity(self):
                return SimpleNamespace(x=3.0, y=4.0, z=0.0)

            def set_transform(self, transform):
                raise AssertionError("Scenic must not move a PCLA ego")

            def destroy(self):
                raise AssertionError("Scenic must not destroy a PCLA ego in sync")

        class Vehicle:
            pass

        obj = Vehicle()
        obj.name = "ego"
        obj.route = ["-29", "27", "Z"]
        obj.carlaActor = Actor()
        obj.pcla = object()
        vehicle_state = {
            "segmentId": "-29",
            "destinationRoadId": "Z",
            "laneIndex": 3,
            "transitionPending": False,
        }
        cosim_record = {
            "vehicleId": 7,
            "isPrivate": True,
            # Deliberately omit routeRoadIds: the PCLA path does not need it.
            "destinationRoadId": "Z",
        }
        simulation = object.__new__(CosimSimulation)
        simulation.ego = obj
        simulation.carla_actors = [obj]
        simulation.carla_control_roads = {"-29": True}
        simulation.carla_control_segments = {"-29"}
        simulation.pending_road_entries = {}
        simulation.pending_route_refreshes = {}
        simulation.pending_lane_reconciliation_verifications = {}
        simulation.same_road_departure_lane_verifications = {}
        simulation.road_entry_holds = {}
        simulation.road_entry_hold_diagnostics = {}
        simulation.completed_route = {}
        simulation.count = 1
        simulation._collect_metsr_vehicle_data = lambda objects: {
            obj: vehicle_state
        }
        simulation.getMetsrPrivateVehId = lambda target: 7
        simulation._mapped_carla_observation = lambda location: ("-29", 3)
        simulation.check_client_synchronization = lambda: None

        def teleport(*args, **kwargs):
            teleport_calls.append((args, kwargs))
            return {
                "messageType": "teleportCoSimVeh",
                "status": "ok",
                "data": [
                    {
                        "vehicleId": 7,
                        "segmentId": "-29",
                        "laneIndex": 3,
                    }
                ],
            }

        simulation.metsr_client = SimpleNamespace(
            query_cosim_vehicle=lambda: {
                "messageType": "coSimVehicle",
                "status": "ok",
                "data": [cosim_record],
            },
            teleport_cosim_vehicle=teleport,
            query_route_between_roads=lambda *args: (_ for _ in ()).throw(
                AssertionError("same-road PCLA sync must not query a route")
            ),
            update_vehicle_route=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("same-road PCLA sync must not replace the route")
            ),
        )

        simulation.synchronize_clients()

        self.assertEqual(len(teleport_calls), 1)
        args, kwargs = teleport_calls[0]
        self.assertEqual(args[:3], (7, 10.0, 2.0))
        self.assertEqual(kwargs["z"], 0.5)
        self.assertEqual(
            kwargs["bearing"], cosim_utils.get_metsr_rotation(15.0)
        )
        self.assertEqual(kwargs["segment_id"], "-29")
        self.assertEqual(kwargs["lane_index"], 3)
        self.assertEqual(kwargs["speed"], 5.0)
        self.assertIs(kwargs["private_veh"], True)
        self.assertIs(kwargs["transform_coords"], True)
        self.assertEqual(simulation.road_entry_holds, {})
        self.assertEqual(simulation.pending_road_entries, {})

    def test_pcla_shadow_accepts_new_road_without_route_queries_or_updates(self):
        simulation, obj, calls = self._sync_guard_fixture(
            {"segmentId": "A", "destinationRoadId": "Z", "laneIndex": 0},
            "B", ["D", "Z"], controlled_roads={"A", "B"},
            lane_connections={("A", "B"): [("A_0", "B_0")]},
            observed_lane=0,
        )
        simulation.ego = obj
        obj.pcla = object()
        simulation.metsr_client.query_route_between_roads = (
            lambda *args: self.fail("METS-R replans its own route on handoff")
        )
        simulation.metsr_client.update_vehicle_route = (
            lambda *args, **kwargs: self.fail("CARLA shadow needs no route rewrite")
        )
        simulation.synchronize_clients()
        self.assertEqual(len(calls["teleport"]), 1)
        self.assertEqual(calls["teleport"][0][1]["segment_id"], "B")
        self.assertEqual(obj.route, ["D", "Z"])
        self.assertEqual(calls["path"], [])

    def test_pcla_ego_cannot_be_destroyed_or_demoted_during_sync(self):
        class Actor:
            def __init__(self):
                self.destroy_calls = 0

            def destroy(self):
                self.destroy_calls += 1

        actor = Actor()
        obj = SimpleNamespace(
            name="ego",
            carlaActor=actor,
            pcla=object(),
            carla_actor_flag=True,
            autopilot_action=False,
            active_autopilot=False,
        )
        simulation = object.__new__(CosimSimulation)
        simulation.ego = obj
        simulation.carla_actors = [obj]
        simulation.metsr_actors = []

        with self.assertRaisesRegex(
            SimulationCreationError, "cannot be destroyed during ordinary"
        ):
            simulation.destroy_carla_obj(obj)
        with self.assertRaisesRegex(
            SimulationCreationError, "cannot be demoted"
        ):
            simulation.remove_bubble_object(obj)

        self.assertEqual(actor.destroy_calls, 0)
        self.assertEqual(simulation.carla_actors, [obj])
        self.assertEqual(simulation.metsr_actors, [])


class CosimCleanupTests(unittest.TestCase):
    class Actor:
        def __init__(self):
            self.id = 73
            self.alive = True
            self.destroy_calls = 0

        @property
        def is_alive(self):
            return self.alive

        def destroy(self):
            if not self.alive:
                raise RuntimeError("trying to operate on a destroyed actor")
            self.destroy_calls += 1
            self.alive = False

    @staticmethod
    def _simulation(obj):
        recorder_calls = []
        simulation = object.__new__(CosimSimulation)
        simulation.run_name = None
        simulation.metsr_client = SimpleNamespace(verbose=False)
        simulation.carla_actors = [obj]
        simulation.render = False
        simulation.cameraManager = None
        simulation.carla_client = SimpleNamespace(
            stop_recorder=lambda: recorder_calls.append(True)
        )
        return simulation, recorder_calls

    def test_camera_closes_before_pcla_and_stale_sensor_sweep_is_skipped(self):
        events = []
        actor = self.Actor()
        pcla = SimpleNamespace(cleanup=lambda **kwargs: events.append("pcla"))
        obj = SimpleNamespace(carlaActor=actor, pcla=pcla)
        simulation, _ = self._simulation(obj)
        simulation.render = True
        simulation.cameraManager = SimpleNamespace(
            destroy_sensor=lambda: events.append("camera")
        )
        simulation.add_pre_actor_teardown_callback(lambda: events.append("dashboard"))
        simulation._destroy_attached_carla_sensors = lambda actor: self.fail(
            "The last CARLA snapshot still lists sensors which PCLA already destroyed"
        )
        with patch.object(carla, "Vehicle", self.Actor):
            simulation.tm = SimpleNamespace(get_port=lambda: 8000)
            actor.set_autopilot = lambda *args: None
            simulation.destroy()
        self.assertEqual(events, ["camera", "dashboard", "pcla"])
        self.assertEqual(actor.destroy_calls, 1)

    def test_destroy_stops_attached_sensors_before_destroying_actor(self):
        actor = self.Actor()

        class Sensor:
            id = 80
            parent = actor

            def __init__(self):
                self.alive = True
                self.listening = True
                self.stop_calls = 0
                self.destroy_calls = 0

            @property
            def is_alive(self):
                return self.alive

            def is_listening(self):
                return self.listening

            def stop(self):
                self.stop_calls += 1
                self.listening = False

            def destroy(self):
                self.destroy_calls += 1
                self.alive = False

        sensor = Sensor()

        class Actors(list):
            def filter(self, pattern):
                return [sensor] if sensor.alive else []

        class World:
            def get_actors(self):
                return Actors([actor, sensor])

            def get_actor(self, actor_id):
                if actor_id == actor.id and actor.alive:
                    return actor
                if actor_id == sensor.id and sensor.alive:
                    return sensor
                return None

        obj = SimpleNamespace(carlaActor=actor, pcla=None)
        simulation, _ = self._simulation(obj)
        simulation.carla_world = World()
        callback_observations = []
        simulation.add_pre_actor_teardown_callback(
            lambda: callback_observations.append(actor.is_alive)
        )

        with redirect_stdout(StringIO()):
            simulation.destroy()

        self.assertEqual(callback_observations, [True])
        self.assertEqual(sensor.stop_calls, 1)
        self.assertEqual(sensor.destroy_calls, 1)
        self.assertFalse(sensor.is_alive)
        self.assertEqual(actor.destroy_calls, 1)

    def test_pcla_non_owning_cleanup_leaves_vehicle_for_scenic(self):
        actor = self.Actor()
        cleanup_arguments = []

        class PCLA:
            def cleanup(self, *, destroy_vehicle=True):
                cleanup_arguments.append(destroy_vehicle)
                if destroy_vehicle:
                    actor.destroy()

        obj = SimpleNamespace(carlaActor=actor, pcla=PCLA())
        simulation, recorder_calls = self._simulation(obj)

        with redirect_stdout(StringIO()):
            simulation.destroy()
            simulation.destroy()

        self.assertEqual(cleanup_arguments, [False])
        self.assertEqual(actor.destroy_calls, 1)
        self.assertIsNone(obj.carlaActor)
        self.assertEqual(recorder_calls, [True])

    def test_legacy_pcla_cleanup_cannot_destroy_scenic_vehicle(self):
        actor = self.Actor()
        cleanup_observations = []

        class LegacyPCLA:
            def __init__(self):
                self.vehicle = actor

            def cleanup(self):
                cleanup_observations.append(
                    (self.vehicle.id, self.vehicle.is_alive)
                )
                if self.vehicle.is_alive:
                    self.vehicle.destroy()
                self.vehicle = None

        pcla = LegacyPCLA()
        obj = SimpleNamespace(carlaActor=actor, pcla=pcla)
        simulation, _ = self._simulation(obj)

        with redirect_stdout(StringIO()):
            simulation.destroy()

        self.assertEqual(cleanup_observations, [(actor.id, False)])
        self.assertEqual(actor.destroy_calls, 1)
        self.assertIsNone(obj.carlaActor)
        self.assertIsNone(pcla.vehicle)

    def test_legacy_pcla_provider_cleanup_cannot_destroy_scenic_vehicle(self):
        actor = self.Actor()
        cleanup_pool_observations = []

        class CarlaDataProvider:
            _carla_actor_pool = {actor.id: actor}

        class LegacyPCLA:
            def __init__(self):
                self.vehicle = actor

            def cleanup(self):
                provider = globals()["CarlaDataProvider"]
                cleanup_pool_observations.append(
                    tuple(provider._carla_actor_pool)
                )
                for pooled_actor in provider._carla_actor_pool.values():
                    pooled_actor.destroy()
                provider._carla_actor_pool.clear()
                if self.vehicle.is_alive:
                    self.vehicle.destroy()
                self.vehicle = None

        pcla = LegacyPCLA()
        obj = SimpleNamespace(carlaActor=actor, pcla=pcla)
        simulation, _ = self._simulation(obj)

        with (
            patch.dict(
                LegacyPCLA.cleanup.__globals__,
                {"CarlaDataProvider": CarlaDataProvider},
            ),
            redirect_stdout(StringIO()),
        ):
            simulation.destroy()

        self.assertEqual(cleanup_pool_observations, [()])
        self.assertEqual(actor.destroy_calls, 1)
        self.assertIsNone(obj.carlaActor)
        self.assertIsNone(pcla.vehicle)

    def test_teardown_does_not_destroy_server_missing_stale_actor(self):
        server_state = {"registered": True}

        class StaleActor:
            id = 74
            is_alive = True

            def __init__(self):
                self.destroy_calls = 0

            def destroy(self):
                self.destroy_calls += 1
                if not server_state["registered"]:
                    raise RuntimeError("unable to destroy actor: not found")

        actor = StaleActor()

        class PCLA:
            def cleanup(self, *, destroy_vehicle=True):
                self.destroy_vehicle = destroy_vehicle
                # Model an external PCLA component removing the server-side actor
                # while this Python proxy still reports is_alive=True.
                server_state["registered"] = False

        pcla = PCLA()
        obj = SimpleNamespace(carlaActor=actor, pcla=pcla)
        simulation, _ = self._simulation(obj)
        simulation.carla_world = SimpleNamespace(
            get_actor=lambda actor_id: actor if server_state["registered"] else None
        )

        with redirect_stdout(StringIO()):
            simulation.destroy()

        self.assertFalse(pcla.destroy_vehicle)
        self.assertEqual(actor.destroy_calls, 0)
        self.assertIsNone(obj.carlaActor)

    def test_pcla_cleanup_failure_does_not_mask_simulation_failure(self):
        actor = self.Actor()

        class FailingPCLA:
            def cleanup(self):
                raise RuntimeError("sensor teardown failed")

        obj = SimpleNamespace(carlaActor=actor, pcla=FailingPCLA())
        simulation, recorder_calls = self._simulation(obj)

        with redirect_stdout(StringIO()), self.assertWarnsRegex(
            RuntimeWarning, "sensor teardown failed"
        ):
            simulation.destroy()

        self.assertEqual(actor.destroy_calls, 1)
        self.assertIsNone(obj.carlaActor)
        self.assertEqual(recorder_calls, [True])

    def test_destroy_releases_every_remaining_cosim_road(self):
        actor = self.Actor()
        obj = SimpleNamespace(carlaActor=actor, pcla=None)
        simulation, recorder_calls = self._simulation(obj)
        simulation.carla_control_roads = {"B": True, "A": True}
        releases = []
        simulation.release_roads = lambda roads: releases.append(set(roads))

        with redirect_stdout(StringIO()):
            simulation.destroy()

        self.assertEqual(releases, [{"A", "B"}])
        self.assertEqual(actor.destroy_calls, 1)
        self.assertEqual(recorder_calls, [True])

    def test_destroy_cleans_boundary_actors_even_when_road_release_fails(self):
        actor = self.Actor()
        boundary = self.Actor()
        boundary.id = 74
        simulation, _ = self._simulation(SimpleNamespace(carlaActor=actor, pcla=None))
        simulation.boundary_actors = {(False, "12"): boundary}

        def reject_release(roads):
            raise METSRControlError("release contract violated")

        simulation.release_roads = reject_release
        with redirect_stdout(StringIO()), self.assertRaises(METSRControlError):
            simulation.destroy()
        simulation.destroy()

        self.assertEqual(actor.destroy_calls, 1)
        self.assertEqual(boundary.destroy_calls, 1)
        self.assertEqual(simulation.boundary_actors, {})

    def test_teardown_clears_all_authoritative_segments(self):
        class HashableObject(SimpleNamespace):
            __hash__ = object.__hash__

        actor = self.Actor()
        obj = HashableObject(carlaActor=actor, pcla=None)
        simulation, _ = self._simulation(obj)
        simulation.carla_control_roads = {}
        simulation._carla_authoritative_segments = {obj: "A"}

        with redirect_stdout(StringIO()):
            simulation.destroy()

        self.assertEqual(simulation._carla_authoritative_segments, {})
        self.assertIsNone(obj.carlaActor)

    def test_destroy_propagates_road_release_failure_after_local_cleanup(self):
        actor = self.Actor()
        obj = SimpleNamespace(carlaActor=actor, pcla=None)
        simulation, recorder_calls = self._simulation(obj)
        simulation.carla_control_roads = {"A": True}

        def reject_release(roads):
            raise METSRControlError("release contract violated")

        simulation.release_roads = reject_release

        with redirect_stdout(StringIO()), self.assertRaisesRegex(
            METSRControlError, "release contract violated"
        ):
            simulation.destroy()

        self.assertEqual(actor.destroy_calls, 1)
        self.assertEqual(recorder_calls, [True])


if __name__ == "__main__":
    unittest.main()
