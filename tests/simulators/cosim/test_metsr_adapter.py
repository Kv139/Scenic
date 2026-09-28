from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scenic.core.simulators import SimulationCreationError
from scenic.simulators.carla.utils import routing as carla_routing
from scenic.simulators.cosim import (
    cosim_routing_utils as bubble_routing,
    cosim_utils as runtime,
)
from scenic.simulators.cosim.cosim_utils import (
    render_step_interval as _render_step_interval,
)
from scenic.simulators.cosim.simulator import carla
from scenic.simulators.cosim.utils import utils as cosim_utils
from scenic.simulators.metsr import (
    cosim_routing_utils as routing,
    cosim_utils as native,
    network as metsr_network,
)
from scenic.simulators.metsr.cosim_utils import (
    COSIM_ADMISSION_SPAWN_MAX_TICKS,
    METSRControlError,
    METSRCosimState,
)
from scenic.simulators.metsr.network import NetworkMetadata
from tests.simulators.cosim.helpers import Vehicle, make_simulation, metsr_response


class NetworkMappingTests(unittest.TestCase):
    def test_default_render_interval_is_one_simulated_second(self):
        self.assertEqual(_render_step_interval(0.1, None), 10)
        self.assertEqual(_render_step_interval(0.05, None), 20)
        self.assertEqual(_render_step_interval(0.1, 4), 4)

    def test_lane_suffix_removal_preserves_internal_road_id(self):
        state = METSRCosimState(
            metadata=NetworkMetadata(lane_mapping={"1034_-1": [":1034_0_0"]})
        )
        self.assertEqual(state.metadata.roads_by_scenic["1034"], {":1034_0"})

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
            indices = metsr_network.load_network_metadata(path).lane_indices

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
            internal = metsr_network.load_network_metadata(path).internal_lane_connections

        self.assertEqual(internal, {":J_0_0": (("A_1", "B_2"),)})

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
            edges = metsr_network.load_network_metadata(path).internal_edge_connections

        self.assertEqual(
            edges,
            {":452_4": (("src_under/%:road_0", "dst_under/%:road_0"),)},
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

    def test_remote_mapped_lane_distance_is_detected_before_teleport(self):
        simulation = make_simulation()
        simulation.metsr_state.road_cache = {("A", 0): [(0.0, 0.0), (10.0, 0.0)]}

        distance = routing.distance_to_mapped_road(
            simulation.metsr_state, simulation.metsr_client, (100.0, 100.0), ["A_0"], "A"
        )

        self.assertGreater(distance, 100.0)

    def test_projection_guard_translates_raw_sumo_lane_suffix(self):
        calls = []
        simulation = make_simulation()
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, lane_indices={"-29_6": 4}
        )
        simulation.metsr_state.road_cache = {}
        simulation.metsr_client = SimpleNamespace(
            query_centerline=lambda road, lane_index, transform_coords: (
                calls.append((road, lane_index, transform_coords))
                or metsr_response(
                    "centerLine",
                    [{"centerline": [(0.0, 0.0), (10.0, 0.0)]}],
                )
            )
        )

        distance = routing.distance_to_mapped_road(
            simulation.metsr_state, simulation.metsr_client, (5.0, 1.0), ["-29_6"], "-29"
        )

        self.assertEqual(distance, 1.0)
        self.assertEqual(calls, [("-29", 4, True)])

    def test_nearest_mapped_lane_returns_exact_compact_index(self):
        simulation = make_simulation()
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, lane_indices={"4_5": 3, "4_6": 4}
        )
        simulation.metsr_state.road_cache = {
            ("4", 3): [(0.0, 3.5), (10.0, 3.5)],
            ("4", 4): [(0.0, 0.0), (10.0, 0.0)],
        }

        lane_index, distance = routing.nearest_mapped_lane(
            simulation.metsr_state,
            simulation.metsr_client,
            (5.0, 0.5),
            ["4_5", "4_6"],
            "4",
        )

        self.assertEqual(lane_index, 4)
        self.assertEqual(distance, 0.5)

    def test_internal_mapped_lane_uses_connector_path_centerline(self):
        calls = []
        connector_id = "cont/6/5"
        simulation = make_simulation()
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata,
            internal_lane_connections={":332_0_4": [("6_4", "5_4")]},
        )
        simulation.metsr_state.internal_edge_to_connector = {":332_0": connector_id}
        simulation.metsr_state.connector_records = {
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
        simulation.metsr_state.connector_path_cache = {}
        simulation.metsr_state.connector_centerline_cache = {}
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

        resolved = routing.nearest_mapped_connector_path_projection(
            simulation.metsr_state, simulation.metsr_client, (5.0, 1.0), [":332_0_4"]
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
        origin_waypoint = SimpleNamespace(road_id=333, lane_id=5, lane_width=3.5)
        destination_waypoint = SimpleNamespace(road_id=100, lane_id=1)

        simulation = make_simulation()
        simulation.network_helper = SimpleNamespace(
            _nearest_lane=lambda target: origin_lane
        )
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata,
            lane_mapping={"333_5": [":332_0_4"], "100_1": ["D_0"]},
        )
        simulation.metsr_state.valid_roads = {"5", "6", "D"}
        simulation.map = SimpleNamespace(
            get_waypoint=lambda location, *args, **kwargs: (
                destination_waypoint if location.x > 50 else origin_waypoint
            )
        )
        simulation.spawn_points = [
            SimpleNamespace(location=carla.Location(x=100.0, y=0.0, z=0.0))
        ]
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, lane_indices={"D_0": 0}
        )
        simulation.metsr_state.road_cache = {("D", 0): [(90.0, 0.0), (110.0, 0.0)]}
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata,
            internal_lane_connections={":332_0_4": [("6_4", "5_4")]},
        )
        simulation.metsr_state.internal_edge_to_connector = {":332_0": connector_id}
        simulation.metsr_state.connector_records = {
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
        simulation.metsr_state.connector_path_cache = {}
        simulation.metsr_state.connector_centerline_cache = {
            (connector_id, 4): [(0.0, 0.0), (10.0, 0.0)]
        }
        simulation.metsr_state.controlled_roads = {"5": True, "6": True}
        simulation.metsr_state.controlled_segments = {"5", "6", ":332_0"}
        simulation.metsr_state.road_connectors = {
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
                "vehicle",
                [
                    {
                        "x": 5.0,
                        "y": 1.0,
                        "segmentId": connector_id,
                        "onConnector": True,
                        "sourceRoadId": "6",
                        "targetRoadId": "5",
                        "destinationRoadId": "D",
                    }
                ],
            ),
            update_vehicle_route=lambda vehicle_id, route, private_veh: (
                route_calls.append((vehicle_id, route, private_veh))
                # Native METS-R rejects a physical-road route while the
                # vehicle still occupies cont/6/5 (the seed-37 failure).
                or metsr_response(
                    "updateVehicleRoute", [{"vehicleId": vehicle_id, "status": "error"}]
                )
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
            width=2.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            parentOrientation=ParentOrientation(),
        )
        calls = {"initialize": [], "route": []}
        simulation = make_simulation()
        simulation.network_helper = SimpleNamespace(
            _nearest_lane=lambda target: SimpleNamespace(road=SimpleNamespace(id=1), id=1)
        )
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, lane_mapping={"1_1": ["A_0"], "2_1": ["D_0"]}
        )
        simulation._nearest_mapped_lane_projection = lambda *args: (0, 0.0, False)
        helper_patch = patch.object(
            routing,
            "nearest_mapped_lane_projection",
            lambda _simulation, *args, **kwargs: simulation._nearest_mapped_lane_projection(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        simulation._source_projection_error_limit = lambda *args: 3.0
        helper_patch = patch.object(
            carla_routing,
            "source_projection_error_limit",
            lambda _simulation, *args, **kwargs: simulation._source_projection_error_limit(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        simulation.identify_nearest_road = lambda *args: "D"
        helper_patch = patch.object(
            routing,
            "identify_nearest_road",
            lambda _simulation, *args, **kwargs: simulation.identify_nearest_road(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        simulation.map = SimpleNamespace(
            get_waypoint=lambda *args: SimpleNamespace(road_id=2, lane_id=1)
        )
        simulation.spawn_points = [
            SimpleNamespace(location=carla.Location(x=100.0, y=0.0, z=0.0))
        ]
        simulation.metsr_state.controlled_roads = {"A": True}
        simulation.getMetsrPrivateVehId = lambda target: 7
        simulation.metsr_client = SimpleNamespace(
            query_route_between_roads=lambda source, target: metsr_response(
                "routesBwRoads", [{"roadIds": [source, target]}]
            ),
            initialize_cosim_vehicle=lambda **kwargs: (
                calls["initialize"].append(kwargs)
                or metsr_response(
                    "initializeCoSimVeh",
                    [
                        {
                            "vehicleId": 7,
                            "status": "ok",
                            "segmentId": "A",
                            "laneIndex": 0,
                        }
                    ],
                )
            ),
            query_vehicle=lambda *args, **kwargs: metsr_response(
                "vehicle", [{"x": 5.0, "y": 1.0, "segmentId": "A"}]
            ),
            update_vehicle_route=lambda vehicle_id, route, private_veh: (
                calls["route"].append((vehicle_id, route, private_veh))
                or metsr_response(
                    "updateVehicleRoute", [{"vehicleId": vehicle_id, "status": "error"}]
                )
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
            "initializeCoSimVeh",
            [
                {
                    "vehicleId": 7,
                    "status": "error",
                    "errorCode": "INITIALIZATION_FAILED",
                    "message": "No route from the matched road to the destination",
                }
            ],
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

        simulation = make_simulation()
        simulation.network_helper = SimpleNamespace(
            _nearest_lane=lambda target: origin_lane
        )
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata,
            lane_mapping={"333_5": ["A_0"], "100_1": ["D_0"]},
        )
        simulation.metsr_state.valid_roads = {"A", "D"}
        simulation.map = SimpleNamespace(
            get_waypoint=lambda location, *args, **kwargs: (
                destination_waypoint if location.x > 50 else origin_waypoint
            )
        )
        simulation.spawn_points = [
            SimpleNamespace(location=carla.Location(x=100.0, y=0.0, z=0.0))
        ]
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, lane_indices={"A_0": 0, "D_0": 0}
        )
        simulation.metsr_state.road_cache = {
            ("A", 0): [(0.0, 0.0), (10.0, 0.0)],
            ("D", 0): [(90.0, 0.0), (110.0, 0.0)],
        }
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, internal_lane_connections={}
        )
        simulation.metsr_state.internal_edge_to_connector = {}
        simulation.metsr_state.connector_records = {}
        simulation.metsr_state.connector_path_cache = {}
        simulation.metsr_state.connector_centerline_cache = {}
        simulation.metsr_state.road_connectors = {}
        simulation.metsr_state.controlled_roads = {}
        simulation.metsr_state.controlled_segments = set()

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
        self.assertEqual(generated, [{"vehID": 7, "origin": "A", "destination": "D"}])

    def _town06_short_lane_spawn_fixture(self):
        simulation, obj, generated = self._native_road_spawn_fixture()
        # Reported demo2 seed-34 spawn, and Town06.net.xml lane -64_4 with
        # netOffset removed. SUMO truncates this driving lane to four metres.
        obj.position = type(obj.position)(-5.303203, -378.247755)
        obj.width = 2.1
        simulation.network_helper._nearest_lane = lambda target: SimpleNamespace(
            road=SimpleNamespace(id=64), id=-3
        )
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata,
            lane_mapping={
                "64_-3": ["-64_4"],
                "100_1": ["D_0"],
            },
        )
        simulation.metsr_state.valid_roads = {"-64", "D"}
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, lane_indices={"-64_4": 2, "D_0": 0}
        )
        simulation.metsr_state.road_cache[("-64", 2)] = [
            (-3.19, -360.49),
            (-3.24, -361.51),
            (-3.31, -362.58),
            (-3.39, -363.64),
            (-3.47, -364.48),
        ]
        simulation.map.get_waypoint = lambda location, *args, **kwargs: (
            SimpleNamespace(road_id=100, lane_id=1)
            if location.x > 50
            else SimpleNamespace(road_id=64, lane_id=-3, lane_width=3.5)
        )
        return simulation, obj, generated

    def test_native_spawn_beyond_short_lane_endpoint_uses_road_queue(self):
        simulation, obj, generated = self._town06_short_lane_spawn_fixture()
        with redirect_stdout(StringIO()):
            simulation.createObjectInMetsr(obj)

        self.assertEqual(obj.route, ["-64", "D"])
        self.assertEqual(generated, [{"vehID": 7, "origin": "-64", "destination": "D"}])

    def test_controlled_spawn_beyond_short_lane_endpoint_is_rejected(self):
        simulation, obj, generated = self._town06_short_lane_spawn_fixture()
        simulation.metsr_state.controlled_roads = {"-64": True}
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
        simulation = make_simulation()
        simulation.objects = [SimpleNamespace(name="ego"), obj]
        simulation.bubble_roads = []
        simulation.sim_ticks_per_carla = 1
        simulation.count = 4
        simulation.completed_route = {}
        simulation.pending_road_entries = {}
        simulation.metsr_state.private_vehicle_ids = {obj: 7}
        simulation.metsr_state.admitted_vehicles = {}
        simulation.metsr_state.controlled_roads = {}
        simulation.metsr_state.controlled_segments = set()
        simulation.metsr_state.connector_records = {}
        simulation.metsr_state.road_connectors = {}
        simulation.metsr_state.internal_edge_to_connector = {}
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

        simulation.update_bubble_objects(vehicle_data={obj: record})

        self.assertNotIn(obj, simulation.completed_route)
        self.assertEqual(simulation.carla_actors, [])
        self.assertEqual(simulation.metsr_actors, [obj])

    def test_polyline_projection_distinguishes_terminal_from_interior(self):
        polyline = [(0.0, 0.0), (0.0, 0.0), (10.0, 0.0)]

        terminal_distance, terminal = carla_routing.point_to_polyline_projection(
            (-4.0, 1.0), polyline
        )
        (
            interior_distance,
            interior_terminal,
        ) = carla_routing.point_to_polyline_projection((5.0, 4.0), polyline)

        self.assertAlmostEqual(terminal_distance, math.sqrt(17.0))
        self.assertTrue(terminal)
        self.assertEqual(interior_distance, 4.0)
        self.assertFalse(interior_terminal)

    def test_terminal_allowance_requires_matching_carla_lane(self):
        simulation = make_simulation()
        waypoint = SimpleNamespace(road_id=52, lane_id=-6, lane_width=3.5)
        simulation.map = SimpleNamespace(get_waypoint=lambda *args, **kwargs: waypoint)
        obj = SimpleNamespace(position=SimpleNamespace(x=-32.487, y=18.810, z=0.0))

        self.assertEqual(
            carla_routing.source_projection_error_limit(
                simulation.map, obj.position, "52_-6", 3.0, at_terminal=True
            ),
            6.5,
        )
        self.assertEqual(
            carla_routing.source_projection_error_limit(
                simulation.map, obj.position, "52_-5", 3.0, at_terminal=True
            ),
            3.0,
        )
        self.assertEqual(
            carla_routing.source_projection_error_limit(
                simulation.map, obj.position, "52_-6", 3.0, at_terminal=False
            ),
            3.0,
        )

    def test_authoritative_lane_map_rejects_an_unknown_lane(self):
        simulation = make_simulation()
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, lane_indices={}
        )
        simulation.metsr_state.valid_roads = ["-29"]

        self.assertEqual(
            metsr_network.mapped_lane_query(simulation.metsr_state, "-29_6"),
            ("-29", None),
        )

    def test_road_only_mapping_preserves_underscores(self):
        simulation = make_simulation()
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, lane_indices={}
        )
        simulation.metsr_state.valid_roads = [":1034_0"]

        self.assertEqual(
            metsr_network.mapped_lane_query(simulation.metsr_state, ":1034_0"),
            (":1034_0", -1),
        )

    def test_malformed_centerline_response_fails_closed(self):
        simulation = make_simulation()
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, lane_indices={"-29_6": 4}
        )
        simulation.metsr_state.road_cache = {}
        simulation.metsr_client = SimpleNamespace(
            query_centerline=lambda *args, **kwargs: metsr_response("centerLine")
        )

        distance = routing.distance_to_mapped_road(
            simulation.metsr_state, simulation.metsr_client, (5.0, 1.0), ["-29_6"], "-29"
        )

        self.assertTrue(math.isinf(distance))

    def test_client_sync_diagnostic_uses_200_step_cadence(self):
        due = runtime.client_sync_diagnostic_due

        self.assertTrue(due(0, True))
        self.assertFalse(due(1, True))
        self.assertFalse(due(199, True))
        self.assertTrue(due(200, True))
        self.assertFalse(due(200, False))

    def _sync_guard_fixture(
        self,
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
            is_alive = True

            def __init__(self):
                self.id = 99
                self.type_id = "vehicle.test"
                self.location = carla.Location(x=10.0, y=-2.0, z=0.0)
                self.transform = carla.Transform(self.location, carla.Rotation(yaw=0.0))
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
                record.setdefault("segmentType", "road")
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
            calls["generate"].append((list(route), obj, required_first_lane_id))
            calls["generate_start"].append(target_start)
            return trajectory

        simulation = make_simulation()
        simulation.carla_actors = [obj]
        simulation.metsr_actors = []
        simulation.metsr_state.authoritative_segments = {}
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
        simulation.metsr_state.controlled_roads = {
            str(road_id): True for road_id in controlled_roads
        }
        simulation.metsr_state.controlled_segments = set(
            simulation.metsr_state.controlled_roads
        )
        simulation.metsr_state.internal_edge_to_connector = {}
        simulation.metsr_state.connector_records = {}
        simulation.metsr_state.road_connectors = {}
        simulation.count = 1
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, lane_indices=dict(lane_indices or {})
        )
        simulation.metsr_state.road_cache = {}
        simulation.carla_world = SimpleNamespace(get_actors=lambda: [obj.carlaActor])
        simulation._collect_metsr_vehicle_data = lambda objects: {obj: state}
        helper_patch = patch.object(
            native,
            "collect_metsr_vehicle_data",
            lambda _simulation, *args, **kwargs: simulation._collect_metsr_vehicle_data(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        simulation.getMetsrPrivateVehId = lambda target: 7
        simulation._mapped_carla_observation = lambda location: (
            ("B", 2)
            if abs(float(location.y) + 5.0) < 1e-6
            else (observed_road, observed_lane)
        )
        simulation._native_handoff_target = lambda target, location: None
        helper_patch = patch.object(
            routing,
            "native_handoff_target",
            lambda _simulation, *args, **kwargs: simulation._native_handoff_target(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
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
            query_cosim_vehicle=lambda: metsr_response("coSimVehicle", cosim_records),
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
        simulation._pending_lane_reconciliation_target = lambda target, road, lane: (
            "B_8",
            old_transform,
            target_transform,
        )
        simulation._assert_pending_lane_reconciliation_clear = (
            lambda actor, old, target: None
        )

        def apply_transform(actor, transform, context):
            calls["transform"].append((actor, transform, context))
            actor.transform = transform
            actor.location = transform.location

        simulation._apply_carla_transform_without_tick = apply_transform
        simulation._install_carla_destination_path = lambda target: (_ for _ in ()).throw(
            AssertionError("the constrained recovery path must not be overwritten")
        )
        simulation.tm = SimpleNamespace(
            set_path=lambda actor, path: calls["path"].append((actor, path)),
            auto_lane_change=lambda actor, enabled: calls["lane_change"].append(
                (actor, enabled)
            ),
        )
        simulation.check_client_synchronization = lambda: None
        helper_patch = patch.object(
            runtime,
            "check_client_synchronization",
            lambda _simulation, *args, **kwargs: simulation.check_client_synchronization(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
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
        self.assertNotIn("segment_id", calls["teleport"][0][1])
        self.assertNotIn("lane_index", calls["teleport"][0][1])

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

        simulation.synchronize_clients()

        self.assertEqual(len(calls["teleport"]), 1)
        self.assertNotIn("segment_id", calls["teleport"][0][1])
        self.assertNotIn("lane_index", calls["teleport"][0][1])

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

        simulation.synchronize_clients()

        self.assertEqual(len(calls["teleport"]), 1)
        self.assertNotIn("segment_id", calls["teleport"][0][1])
        self.assertEqual(calls["transform"], [])

    def test_queue_service_ignores_old_roads_and_unmanaged_vehicles(self):
        class ManagedObject:
            pass

        managed = ManagedObject()
        admitted = []
        simulation = make_simulation()
        simulation.metsr_state.controlled_roads = {"A": True, "OLD": True}
        simulation.metsr_state.active_bubble_roads = {"A"}
        simulation.metsr_state.private_vehicle_ids = {managed: 11}
        simulation.metsr_state.admitted_vehicles = {}
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
                ],
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
            simulation.metsr_state.admitted_vehicles,
            {11: {"roadId": "A", "admittedAt": 7}},
        )

    def test_queue_service_batches_ready_managed_vehicles(self):
        class ManagedObject:
            pass

        first = ManagedObject()
        second = ManagedObject()
        requests_seen = []
        simulation = make_simulation()
        simulation.metsr_state.controlled_roads = {"A": True}
        simulation.metsr_state.active_bubble_roads = {"A"}
        simulation.metsr_state.private_vehicle_ids = {first: 11, second: 12}
        simulation.metsr_state.admitted_vehicles = {}
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
            simulation.metsr_state.admitted_vehicles,
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
        simulation = make_simulation()
        simulation.metsr_state.controlled_roads = {"A": True}
        simulation.metsr_state.active_bubble_roads = {"A"}
        simulation.metsr_state.private_vehicle_ids = {managed: 11}
        simulation.metsr_state.admitted_vehicles = {}
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
        simulation = make_simulation()
        simulation.metsr_state.private_vehicle_ids = {managed: 11}
        simulation.metsr_state.admitted_vehicles = {11: {"roadId": "A", "admittedAt": 2}}
        simulation.count = 2 + COSIM_ADMISSION_SPAWN_MAX_TICKS

        with self.assertRaisesRegex(SimulationCreationError, "did not spawn it within"):
            native.check_queue_admission_spawn_deadlines(
                simulation.metsr_state, simulation.count, set()
            )

    def _connector_control_fixture(self, represented_roads, edge_connections=None):
        simulation = make_simulation()
        simulation.metsr_state.controlled_roads = {}
        simulation.metsr_state.controlled_segments = set()
        simulation.metsr_state.internal_edge_to_connector = {}
        simulation.metsr_state.connector_records = {}
        simulation.metsr_state.road_connectors = {}
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata,
            internal_edge_connections=dict(edge_connections or {}),
        )
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, internal_lane_connections={}
        )
        lane_ids = {
            str(lane)
            for pairs in simulation.metsr_state.metadata.internal_edge_connections.values()
            for pair in pairs
            for lane in pair
        }
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata,
            lane_indices={lane: index for index, lane in enumerate(sorted(lane_ids))},
        )
        simulation.network_helper = SimpleNamespace(
            metsr_represented_roads=set(map(str, represented_roads))
        )
        simulation.metsr_state.valid_roads = list(map(str, represented_roads))
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
                or self._takeover_response("-37", connector_id, [":1146_3"], "-36")
            )
        )

        self.assertIsNone(simulation.freeze_roads(["-37"]))

        self.assertEqual(calls, [["-37"]])
        self.assertEqual(
            simulation.metsr_state.internal_edge_to_connector,
            {":1146_3": connector_id},
        )
        self.assertEqual(simulation.metsr_state.controlled_segments, {"-37", ":1146_3"})
        self.assertNotIn(connector_id, simulation.metsr_state.controlled_segments)

    def test_freeze_accepts_outbound_connector_touching_controlled_source(self):
        connector_id = "cont/A/B"
        simulation = self._connector_control_fixture({"A"}, {":A_B": [("A_0", "B_0")]})
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

        self.assertEqual(simulation.metsr_state.road_connectors, {"A": {connector_id}})
        self.assertEqual(
            simulation.metsr_state.internal_edge_to_connector,
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
            simulation.metsr_state.controlled_segments,
            {"-14_2", ":452_4", ":452_14"},
        )
        self.assertEqual(
            simulation.metsr_state.internal_edge_to_connector,
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
                    simulation.metsr_state.internal_edge_to_connector[edge_id],
                    connector_id,
                )
                self.assertNotIn(connector_id, simulation.metsr_state.controlled_segments)

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
        simulation = make_simulation()
        waypoint = SimpleNamespace(
            road_id=334,
            lane_id=4,
            is_junction=True,
            transform=carla.Transform(
                carla.Location(x=location.x + 1.292, y=location.y, z=0.0),
                carla.Rotation(),
            ),
        )
        simulation.map = SimpleNamespace(get_waypoint=lambda *args, **kwargs: waypoint)
        simulation.network_helper = SimpleNamespace()
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, lane_mapping={"334_4": [":332_8_2"]}
        )
        simulation.metsr_state.valid_roads = {"6", "5", "-38"}
        simulation.metsr_state.controlled_roads = {"6": True, "5": True, "-38": True}
        simulation.metsr_state.controlled_segments = {
            "6",
            "5",
            "-38",
            ":332_0",
            ":332_8",
        }
        simulation.metsr_state.internal_edge_to_connector = {
            ":332_0": previous,
            ":332_8": other,
        }
        simulation.metsr_state.connector_records = {
            previous: {
                "connectorId": previous,
                "sourceRoadId": "6",
                "targetRoadId": "5",
                "intersectionId": 14,
                "internalEdgeIds": [":332_0"],
                "paths": [{"connectorPathId": 2, "viaLaneIds": [":332_0_4"]}],
            },
            other: {
                "connectorId": other,
                "sourceRoadId": "-38",
                "targetRoadId": "5",
                "intersectionId": 14,
                "internalEdgeIds": [":332_8"],
                "paths": [{"connectorPathId": 2, "viaLaneIds": [":332_8_2"]}],
            },
        }
        simulation.metsr_state.road_connectors = {"5": {previous, other}}
        simulation.metsr_state.connector_centerline_cache = {
            (previous, 2): [(664.8, -168.58, 0.0), (665.32, -120.35, 0.0)]
        }
        simulation.metsr_state.authoritative_segments = {obj: previous}
        state = {"segmentId": previous, "connectorId": previous, "connectorPathId": 2}
        return simulation, obj, location, waypoint, state

    def _car_4_authority_fixture(self, raw_observations, teleport_response=None):
        connector_id = "cont/-36/-37"

        class Vehicle:
            pass

        class Actor:
            is_alive = True

            def __init__(self):
                self.location = carla.Location(x=181.81, y=136.67, z=0.0)
                self.transform = carla.Transform(self.location, carla.Rotation(yaw=7.0))
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
        simulation = make_simulation()
        simulation.carla_actors = [obj]
        simulation.metsr_actors = []
        simulation.metsr_state.authoritative_segments = {}
        simulation.metsr_state.controlled_roads = {"-37": True}
        simulation.metsr_state.controlled_segments = {"-37", ":1146_3"}
        simulation.metsr_state.internal_edge_to_connector = {":1146_3": connector_id}
        simulation.metsr_state.connector_records = {
            connector_id: {
                "connectorId": connector_id,
                "sourceRoadId": "-36",
                "targetRoadId": "-37",
                "internalEdgeIds": [":1146_3"],
            }
        }
        simulation.metsr_state.road_connectors = {"-37": {connector_id}}
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, lane_indices={}
        )
        simulation.metsr_state.road_cache = {}
        simulation.pending_road_entries = {}
        simulation.pending_route_refreshes = {}
        simulation.pending_lane_reconciliation_verifications = {}
        simulation.same_road_departure_lane_verifications = {}
        simulation.road_entry_holds = {}
        simulation.road_entry_hold_diagnostics = {}
        simulation.completed_route = {}
        simulation.count = 36
        simulation.ego = None
        simulation._collect_metsr_vehicle_data = lambda objects: {obj: vehicle_state}
        helper_patch = patch.object(
            native,
            "collect_metsr_vehicle_data",
            lambda _simulation, *args, **kwargs: simulation._collect_metsr_vehicle_data(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
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
            query_cosim_vehicle=lambda: metsr_response("coSimVehicle", [cosim_record]),
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
        simulation._apply_carla_transform_without_tick = lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(AssertionError("a shadow response must not transform CARLA"))
        simulation.check_client_synchronization = lambda: None
        helper_patch = patch.object(
            runtime,
            "check_client_synchronization",
            lambda _simulation, *args, **kwargs: simulation.check_client_synchronization(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        return simulation, obj, calls

    def test_pose_publication_does_not_depend_on_flickering_source_membership(self):
        simulation, obj, calls = self._car_4_authority_fixture(
            [("-36", 4), ("-37", 3), ("-36", 4)]
        )

        simulation.synchronize_clients()
        simulation.synchronize_clients()
        simulation.synchronize_clients()

        self.assertTrue(all("pose_only" not in call[1] for call in calls))
        self.assertNotIn("lane_index", calls[0][1])
        self.assertNotIn("lane_index", calls[1][1])
        self.assertNotIn("lane_index", calls[2][1])
        self.assertTrue(all("segment_id" not in call[1] for call in calls))

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

        self.assertEqual(raised.exception.record["errorCode"], "INVALID_CONNECTOR_PATH")
        self.assertEqual(len(calls), 1)
        self.assertEqual(obj.carlaActor.transform_calls, [])

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

        self.assertEqual(simulation.metsr_state.controlled_roads, {})
        self.assertEqual(simulation.metsr_state.controlled_segments, set())
        self.assertEqual(attempts, [["A", "B"]])
        self.assertEqual(rollbacks, [])

    def test_freeze_rejects_missing_expected_incident_edge_without_rollback(self):
        rollbacks = []
        simulation = self._connector_control_fixture(
            {"-37"}, {":1146_3": [("-36_6", "-37_5")]}
        )
        response = self._takeover_response("-37", "cont/-36/-37", [], "-36")
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
        self.assertEqual(simulation.metsr_state.controlled_roads, {})

    def test_freeze_roads_propagates_nonretryable_failure(self):
        simulation = self._connector_control_fixture({"A"})

        def reject(roads):
            raise METSRControlError("permanent failure", record={"retryable": False})

        simulation.metsr_client = SimpleNamespace(set_cosim_road=reject)

        with self.assertRaises(METSRControlError):
            simulation.freeze_roads(["A"])

        self.assertEqual(simulation.metsr_state.controlled_roads, {})

    def test_release_preserves_connector_owned_by_other_endpoint(self):
        simulation = self._connector_control_fixture({"A", "B"})
        simulation.metsr_state.controlled_roads = {"A": True, "B": True}
        connector_id = "cont/A/B"
        connector = {
            "connectorId": connector_id,
            "sourceRoadId": "A",
            "targetRoadId": "B",
            "internalEdgeIds": [":1146_3"],
            "controlMode": "COSIM",
        }
        simulation.metsr_state.controlled_segments = {"A", "B", ":1146_3"}
        simulation.metsr_state.internal_edge_to_connector = {":1146_3": connector_id}
        simulation.metsr_state.connector_records = {connector_id: connector}
        simulation.metsr_state.road_connectors = {
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

        simulation.metsr_client = SimpleNamespace(release_cosim_road=release)

        simulation.release_roads(["A"])

        self.assertEqual(released, ["A"])
        self.assertEqual(simulation.metsr_state.controlled_roads, {"B": True})
        self.assertEqual(
            simulation.metsr_state.internal_edge_to_connector,
            {":1146_3": connector_id},
        )
        self.assertIn(":1146_3", simulation.metsr_state.controlled_segments)

        simulation.release_roads(["B"])

        self.assertEqual(released, ["A", "B"])
        self.assertEqual(simulation.metsr_state.controlled_roads, {})
        self.assertEqual(simulation.metsr_state.internal_edge_to_connector, {})
        self.assertEqual(simulation.metsr_state.controlled_segments, set())

    def test_release_failure_is_fatal_and_keeps_local_ownership(self):
        simulation = self._connector_control_fixture({"A"})
        simulation.metsr_state.controlled_roads = {"A": True}
        simulation.metsr_state.controlled_segments = {"A"}
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

        with self.assertRaisesRegex(METSRControlError, "UNEXPECTED_RELEASE_FAILURE"):
            simulation.release_roads(["A"])

        self.assertEqual(simulation.metsr_state.controlled_roads, {"A": True})
        self.assertEqual(simulation.metsr_state.controlled_segments, {"A"})

    def test_exact_car_4_connector_record_promotes_by_target_road(self):
        class Vehicle:
            pass

        car_4 = Vehicle()
        car_4.name = "car_4"
        car_4.spawn_guard = 0
        car_4.carla_actor_flag = False
        simulation = make_simulation()
        simulation.objects = [SimpleNamespace(name="ego"), car_4]
        simulation.bubble_roads = []
        simulation.sim_ticks_per_carla = 1
        simulation.count = 37
        simulation.completed_route = {}
        simulation.pending_road_entries = {}
        simulation.metsr_state.private_vehicle_ids = {}
        simulation.metsr_state.admitted_vehicles = {}
        simulation.metsr_state.controlled_roads = {"-37": True}
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
        helper_patch = patch.object(
            native,
            "collect_metsr_vehicle_data",
            lambda _simulation, *args, **kwargs: simulation._collect_metsr_vehicle_data(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        promoted = []
        simulation.createObjectInCarla = (
            lambda obj, update_velocity=True, native_state=None: (
                promoted.append((obj, update_velocity, native_state)) or True
            )
        )

        simulation.update_bubble_objects()

        self.assertEqual(promoted, [(car_4, True, record)])
        self.assertEqual(simulation.carla_actors, [car_4])
        self.assertEqual(simulation.metsr_actors, [])

    def test_outbound_connector_stays_in_cosim_when_only_source_is_controlled(self):
        class Vehicle:
            pass

        vehicle = Vehicle()
        vehicle.name = "outbound"
        vehicle.spawn_guard = 0
        vehicle.carla_actor_flag = False
        simulation = make_simulation()
        simulation.objects = [SimpleNamespace(name="ego"), vehicle]
        simulation.bubble_roads = []
        simulation.sim_ticks_per_carla = 1
        simulation.count = 4
        simulation.completed_route = {}
        simulation.pending_road_entries = {}
        simulation.metsr_state.private_vehicle_ids = {}
        simulation.metsr_state.admitted_vehicles = {}
        simulation.metsr_state.controlled_roads = {"A": True}
        simulation.metsr_state.controlled_segments = {"A", ":A_B_0"}
        simulation.metsr_state.connector_records = {
            "cont/A/B": {
                "connectorId": "cont/A/B",
                "sourceRoadId": "A",
                "targetRoadId": "B",
                "internalEdgeIds": [":A_B_0"],
            }
        }
        simulation.metsr_state.road_connectors = {"A": {"cont/A/B"}}
        simulation.metsr_state.internal_edge_to_connector = {":A_B_0": "cont/A/B"}
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
        helper_patch = patch.object(
            native,
            "collect_metsr_vehicle_data",
            lambda _simulation, *args, **kwargs: simulation._collect_metsr_vehicle_data(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        promoted = []
        simulation.createObjectInCarla = (
            lambda obj, update_velocity=True, native_state=None: (
                promoted.append((obj, native_state)) or True
            )
        )

        simulation.update_bubble_objects()

        self.assertEqual(promoted, [(vehicle, record)])
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
        simulation = make_simulation()
        simulation.carla_actors = [obj]
        simulation.objects = [SimpleNamespace(name="ego"), obj]
        simulation.bubble_roads = []
        simulation.sim_ticks_per_carla = 1
        simulation.count = 8
        simulation.completed_route = {}
        simulation.pending_road_entries = {}
        simulation.metsr_state.controlled_roads = {"A": True}
        simulation.road_pop_density = {"N": 0}
        simulation.metsr_state.private_vehicle_ids = {}
        simulation.metsr_state.admitted_vehicles = {}
        simulation._collect_metsr_vehicle_data = lambda objects: {
            obj: {
                "segmentId": "N",
                "transitionPending": False,
                "x": 1.0,
                "y": 2.0,
            }
        }
        helper_patch = patch.object(
            native,
            "collect_metsr_vehicle_data",
            lambda _simulation, *args, **kwargs: simulation._collect_metsr_vehicle_data(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        demoted = []
        simulation.remove_bubble_object = lambda target: demoted.append(target)

        simulation.update_bubble_objects()
        self.assertEqual(demoted, [])

        simulation._remove_completed_carla_objects()
        self.assertEqual(demoted, [])

    def test_step_synchronizes_before_releasing_old_roads(self):
        simulation = make_simulation()
        simulation.metsr_state.valid_roads = []
        simulation.ego = SimpleNamespace(x=0.0, y=0.0, bubble=None)
        simulation.bubble_size = 25.0
        simulation._get_bubble_roads = lambda region: []
        helper_patch = patch.object(
            bubble_routing,
            "get_bubble_roads",
            lambda _simulation, *args, **kwargs: simulation._get_bubble_roads(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        simulation._select_bubble_roads = lambda roads: routing.BubbleRoadSelection(
            tuple(roads), frozenset(["new"]), ("new",), ("old",)
        )
        events = []
        simulation.freeze_roads = lambda roads: events.append(("freeze", list(roads)))
        simulation.release_roads = lambda roads, **kwargs: events.append(
            ("release", list(roads))
        )
        simulation._synchronize_boundary_vehicles = lambda: events.append(
            ("boundary", None)
        )
        simulation._service_metsr_entering_queues = lambda: events.append(
            ("queues", None)
        )
        simulation.get_bubble_intersections = lambda **kwargs: []
        helper_patch = patch.object(
            bubble_routing,
            "get_bubble_intersections",
            lambda _simulation, *args, **kwargs: simulation.get_bubble_intersections(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        simulation.update_bubble_objects = lambda *args, **kwargs: events.append(
            ("bubble", None)
        )
        simulation.tick_carla = lambda: events.append(("carla", None))
        simulation.synchronize_clients = lambda **kwargs: events.append(("sync", None))
        simulation._report_road_entry_holds = lambda: None
        simulation.tick_metsr = lambda: None
        vehicle_queries = []
        simulation._collect_metsr_vehicle_data = (
            lambda objects: vehicle_queries.append(tuple(objects)) or {}
        )
        helper_patch = patch.object(
            native,
            "collect_metsr_vehicle_data",
            lambda _simulation, *args, **kwargs: simulation._collect_metsr_vehicle_data(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        simulation.metsr_actors = []
        simulation.carla_actors = []
        simulation.render = False
        simulation.bubble_sizes = []
        simulation.objects = [simulation.ego]
        simulation.frozen_vehicles = set()
        simulation.bubble_spawn_queue = set()
        simulation.total_active_vehicles = []
        simulation.completed_route = {}
        simulation.count = 1
        simulation.run_name = None

        simulation.step()

        self.assertEqual(
            events,
            [
                ("freeze", ["new"]),
                ("boundary", None),
                ("queues", None),
                ("bubble", None),
                ("carla", None),
                ("sync", None),
                ("release", ["old"]),
            ],
        )
        self.assertEqual(len(vehicle_queries), 2)

    def test_old_road_release_waits_for_last_acknowledged_occupant(self):
        class Vehicle:
            pass

        obj = Vehicle()
        simulation = make_simulation()
        simulation.carla_actors = [obj]
        simulation.metsr_state.controlled_roads = {"old": True, "new": True}
        simulation.metsr_state.controlled_segments = {"old", "new"}
        simulation.metsr_state.internal_edge_to_connector = {}
        simulation.metsr_state.connector_records = {}
        simulation.metsr_state.road_connectors = {}
        simulation.pending_road_entries = {}
        simulation.metsr_state.authoritative_segments = {obj: "old"}

        self.assertEqual(
            native.releasable_old_roads(
                simulation.metsr_state, ["old"], simulation.carla_actors
            ),
            [],
        )

        simulation.metsr_state.authoritative_segments[obj] = "new"
        self.assertEqual(
            native.releasable_old_roads(
                simulation.metsr_state, ["old"], simulation.carla_actors
            ),
            ["old"],
        )

    def test_occupied_connector_keeps_its_endpoint_roads_controlled(self):
        class Vehicle:
            pass

        obj = Vehicle()
        connector_id = "opaque/A/B"
        simulation = make_simulation()
        simulation.carla_actors = [obj]
        simulation.metsr_state.controlled_roads = {"A": True, "B": True, "old": True}
        simulation.metsr_state.controlled_segments = {"A", "B", "old", ":A_B_0"}
        simulation.metsr_state.internal_edge_to_connector = {":A_B_0": connector_id}
        simulation.metsr_state.connector_records = {
            connector_id: {
                "connectorId": connector_id,
                "sourceRoadId": "A",
                "targetRoadId": "B",
                "internalEdgeIds": [":A_B_0"],
            }
        }
        simulation.metsr_state.road_connectors = {
            "A": {connector_id},
            "B": {connector_id},
        }
        simulation.pending_road_entries = {}
        simulation.metsr_state.authoritative_segments = {obj: connector_id}

        self.assertEqual(
            native.releasable_old_roads(
                simulation.metsr_state, ["A", "B", "old"], simulation.carla_actors
            ),
            ["old"],
        )

    def test_unacknowledged_carla_actor_defers_road_release(self):
        obj = object()
        simulation = make_simulation()
        simulation.carla_actors = [obj]
        simulation.metsr_state.controlled_roads = {"A": True, "B": True}
        simulation.metsr_state.controlled_segments = {"A", "B"}
        simulation.metsr_state.internal_edge_to_connector = {}
        simulation.metsr_state.connector_records = {}
        simulation.metsr_state.road_connectors = {}
        simulation.pending_road_entries = {}
        simulation.metsr_state.authoritative_segments = {}

        self.assertEqual(
            native.releasable_old_roads(
                simulation.metsr_state, ["A", "B"], simulation.carla_actors
            ),
            [],
        )

    def test_spawn_ego_calls_takeover_once(self):
        simulation = make_simulation()
        simulation.bubble_size = 25.0
        simulation._get_bubble_roads = lambda: []
        helper_patch = patch.object(
            bubble_routing,
            "get_bubble_roads",
            lambda _simulation, *args, **kwargs: simulation._get_bubble_roads(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        simulation._select_bubble_roads = lambda roads: routing.BubbleRoadSelection(
            tuple(roads), frozenset(["A", "B"]), ("A", "B"), ()
        )
        events = []

        simulation.freeze_roads = lambda roads: events.append(("freeze", set(roads)))
        simulation.metsr_client = SimpleNamespace(
            tick=lambda: events.append(("tick", None))
        )
        simulation.createObjectInMetsr = lambda obj: events.append(("metsr", None))
        simulation.createObjectInCarla = (
            lambda obj: events.append(("carla", None)) or True
        )
        obj = Vehicle(position=SimpleNamespace(x=1.0, y=2.0))

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
        simulation = make_simulation()
        simulation.bubble_size = 25.0
        simulation._get_bubble_roads = lambda: []
        helper_patch = patch.object(
            bubble_routing,
            "get_bubble_roads",
            lambda _simulation, *args, **kwargs: simulation._get_bubble_roads(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        simulation._select_bubble_roads = lambda roads: routing.BubbleRoadSelection(
            tuple(roads), frozenset(["A"]), ("A",), ()
        )
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
        obj = Vehicle(position=SimpleNamespace(x=1.0, y=2.0))

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
        simulation = make_simulation()
        simulation.bubble_size = 25.0
        simulation._get_bubble_roads = lambda: []
        helper_patch = patch.object(
            bubble_routing,
            "get_bubble_roads",
            lambda _simulation, *args, **kwargs: simulation._get_bubble_roads(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        simulation._select_bubble_roads = lambda roads: routing.BubbleRoadSelection(
            tuple(roads), frozenset(["A"]), ("A",), ()
        )
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
        obj = Vehicle(position=SimpleNamespace(x=1.0, y=2.0))

        with self.assertRaisesRegex(METSRControlError, "takeover failed"):
            simulation.spawn_ego(obj)

        self.assertEqual(ticks, [])
        self.assertEqual(freeze_calls, [{"A"}])
        self.assertEqual(created, [])

    def test_initiate_autopilot_uses_configured_tm_and_allows_lane_changes(self):
        events = []

        class Actor:
            is_alive = True

            def set_autopilot(self, enabled, port):
                events.append(("autopilot", enabled, port))

        actor = Actor()
        simulation = make_simulation()
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
        simulation = make_simulation()
        simulation.metsr_state.authoritative_segments = {obj: "A"}
        simulation.bubble_spawn_queue = {obj}
        simulation.blueprintLib = SimpleNamespace(find=lambda name: Blueprint())
        simulation.carla_world = SimpleNamespace(
            try_spawn_actor=lambda blueprint, transform: None
        )

        with patch(
            "scenic.simulators.cosim.cosim_utils.utils.scenicToCarlaLocation",
            return_value=carla.Location(x=1.0, y=2.0, z=0.0),
        ), patch(
            "scenic.simulators.cosim.cosim_utils.utils.scenicToCarlaRotation",
            return_value=carla.Rotation(),
        ):
            self.assertFalse(simulation.createObjectInCarla(obj))

        self.assertNotIn(obj, simulation.metsr_state.authoritative_segments)

    def test_default_trajectory_queries_private_vehicle_destination(self):
        class Vehicle:
            route = ["STALE"]
            carlaActor = object()

        simulation = make_simulation()
        obj = Vehicle()
        calls = []
        simulation.getMetsrPrivateVehId = lambda target: 87

        def query_vehicle(vehicle_id, private_veh, transform_coords):
            calls.append(
                (
                    vehicle_id,
                    {"private_veh": private_veh, "transform_coords": transform_coords},
                )
            )
            return metsr_response(
                "vehicle",
                [
                    {
                        "vehicleId": 87,
                        "segmentId": "A",
                        "destinationRoadId": "N",
                    }
                ],
            )

        simulation.metsr_client = SimpleNamespace(query_vehicle=query_vehicle)
        path = [object()]
        simulation.generate_carla_destination_trajectory = (
            lambda destination, target, *, target_start=None: calls.append(
                (destination, target)
            )
            or path
        )
        helper_patch = patch.object(
            routing,
            "generate_carla_destination_trajectory",
            lambda _simulation, *args, **kwargs: simulation.generate_carla_destination_trajectory(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        self.assertIs(simulation._install_carla_destination_path(obj), path)
        self.assertEqual(
            calls,
            [
                ([87], {"private_veh": True, "transform_coords": True}),
                ("N", obj),
            ],
        )
        self.assertEqual(simulation._carla_destination_plans[obj], ("N", path))

    def test_default_trajectory_requires_queried_destination(self):
        simulation = make_simulation()
        obj = Vehicle(route=["STALE", "Z"])
        simulation.getMetsrPrivateVehId = lambda target: 87
        simulation.metsr_client = SimpleNamespace(
            query_vehicle=lambda *args, **kwargs: metsr_response(
                "vehicle", [{"vehicleId": 87, "segmentId": "A"}]
            ),
        )
        with self.assertRaisesRegex(SimulationCreationError, "omitted destinationRoadId"):
            simulation._install_carla_destination_path(obj)

    def test_default_trajectory_ignores_remaining_route_and_pending_fields(self):
        class Vehicle:
            route = ["STALE", "WRONG"]
            carlaActor = object()

        simulation = make_simulation()
        obj = Vehicle()
        simulation.getMetsrPrivateVehId = lambda target: 87
        simulation.metsr_client = SimpleNamespace(
            query_vehicle=lambda *args, **kwargs: metsr_response(
                "vehicle",
                [
                    {
                        "vehicleId": 87,
                        "destinationRoadId": "N",
                        "routeRoadIds": ["T"],
                        "transitionPending": True,
                        "transitionTargetRoadId": "WRONG",
                        "transitionTargetLaneIndex": 0,
                    }
                ],
            ),
        )
        calls = []
        expected = [object()]
        simulation.generate_carla_destination_trajectory = (
            lambda destination, target, *, target_start=None: calls.append(
                (destination, target)
            )
            or expected
        )
        helper_patch = patch.object(
            routing,
            "generate_carla_destination_trajectory",
            lambda _simulation, *args, **kwargs: simulation.generate_carla_destination_trajectory(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        self.assertIs(simulation._install_carla_destination_path(obj), expected)
        self.assertEqual(calls, [("N", obj)])

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
        self.assertNotIn("segment_id", calls["teleport"][0][1])

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
        simulation, obj, calls = self._sync_guard_fixture(state, "Z", ["Z"])
        retirements = []

        def reach_dest(vehicle_id, *, private_veh):
            retirements.append((vehicle_id, private_veh))
            return {
                "messageType": "reachDest",
                "status": "ok",
                "data": [{"status": "ok"}],
            }

        simulation.metsr_client.reach_dest = reach_dest

        simulation.synchronize_clients()

        self.assertTrue(simulation.completed_route[obj])
        self.assertEqual(len(calls["teleport"]), 1)
        self.assertEqual(retirements, [(7, True)])

    def test_ordinary_carla_updates_use_one_batched_teleport(self):
        class Vehicle:
            pass

        first = Vehicle()
        second = Vehicle()
        calls = []
        simulation = make_simulation()
        simulation.metsr_state.controlled_roads = {"A": True, "B": True}
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

        simulation.metsr_client = SimpleNamespace(teleport_cosim_vehicle=teleport)
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
        self.assertNotIn("segment_id", kwargs)
        self.assertNotIn("lane_index", kwargs)
        self.assertEqual(kwargs["speed"], [3.0, 6.0])

    def test_pcla_ego_same_road_sync_only_updates_metsr_shadow(self):
        teleport_calls = []

        class Actor:
            is_alive = True

            def __init__(self):
                self.location = carla.Location(x=10.0, y=-2.0, z=0.5)
                self.transform = carla.Transform(self.location, carla.Rotation(yaw=15.0))

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
        simulation = make_simulation()
        simulation.ego = obj
        simulation.carla_actors = [obj]
        simulation.metsr_state.controlled_roads = {"-29": True}
        simulation.metsr_state.controlled_segments = {"-29"}
        simulation.pending_road_entries = {}
        simulation.pending_route_refreshes = {}
        simulation.pending_lane_reconciliation_verifications = {}
        simulation.same_road_departure_lane_verifications = {}
        simulation.road_entry_holds = {}
        simulation.road_entry_hold_diagnostics = {}
        simulation.completed_route = {}
        simulation.count = 1
        simulation._collect_metsr_vehicle_data = lambda objects: {obj: vehicle_state}
        helper_patch = patch.object(
            native,
            "collect_metsr_vehicle_data",
            lambda _simulation, *args, **kwargs: simulation._collect_metsr_vehicle_data(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
        simulation.getMetsrPrivateVehId = lambda target: 7
        simulation._mapped_carla_observation = lambda location: ("-29", 3)
        simulation.check_client_synchronization = lambda: None
        helper_patch = patch.object(
            runtime,
            "check_client_synchronization",
            lambda _simulation, *args, **kwargs: simulation.check_client_synchronization(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()

        def teleport(*args, **kwargs):
            teleport_calls.append((args, kwargs))
            return {
                "messageType": "teleportCoSimVeh",
                "status": "ok",
                "data": [
                    {
                        "vehicleId": 7,
                        "controlMode": "cosim",
                        "releasedFromCoSim": False,
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
        self.assertEqual(kwargs["bearing"], cosim_utils.get_metsr_rotation(15.0))
        self.assertNotIn("segment_id", kwargs)
        self.assertNotIn("lane_index", kwargs)
        self.assertEqual(kwargs["speed"], 5.0)
        self.assertIs(kwargs["private_veh"], True)
        self.assertIs(kwargs["transform_coords"], True)
        self.assertEqual(simulation.road_entry_holds, {})
        self.assertEqual(simulation.pending_road_entries, {})

    def test_pcla_shadow_accepts_new_road_without_route_queries_or_updates(self):
        simulation, obj, calls = self._sync_guard_fixture(
            {"segmentId": "A", "destinationRoadId": "Z", "laneIndex": 0},
            "B",
            ["D", "Z"],
            controlled_roads={"A", "B"},
            lane_connections={("A", "B"): [("A_0", "B_0")]},
            observed_lane=0,
        )
        simulation.ego = obj
        obj.pcla = object()
        simulation.metsr_client.query_route_between_roads = lambda *args: self.fail(
            "METS-R replans its own route on handoff"
        )
        simulation.metsr_client.update_vehicle_route = lambda *args, **kwargs: self.fail(
            "CARLA shadow needs no route rewrite"
        )
        simulation.synchronize_clients()
        self.assertEqual(len(calls["teleport"]), 1)
        self.assertNotIn("segment_id", calls["teleport"][0][1])
        self.assertEqual(obj.route, ["D", "Z"])
        self.assertEqual(calls["path"], [])

    def test_pcla_ego_cannot_be_destroyed_or_demoted_during_sync(self):
        class Actor:
            is_alive = True

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
        simulation = make_simulation()
        simulation.ego = obj
        simulation.carla_actors = [obj]
        simulation.metsr_actors = []

        with self.assertRaisesRegex(
            SimulationCreationError, "cannot be destroyed during ordinary"
        ):
            simulation.destroy_carla_obj(obj)
        with self.assertRaisesRegex(SimulationCreationError, "cannot be demoted"):
            simulation.remove_bubble_object(obj)

        self.assertEqual(actor.destroy_calls, 0)
        self.assertEqual(simulation.carla_actors, [obj])
        self.assertEqual(simulation.metsr_actors, [])

    def test_native_actions_use_per_vehicle_routes_and_requested_acceleration(self):
        class Vehicle(SimpleNamespace):
            __hash__ = object.__hash__

        vehicles = [
            Vehicle(
                carla_actor_flag=False,
                autopilot_action=False,
                target_acceleration=acc,
                trajectory=[Vehicle(road=SimpleNamespace(id=str(index)), id=-1)],
                trajectory_is_active=False,
            )
            for index, acc in enumerate((1.5, -1.0))
        ]
        simulation = make_simulation()
        simulation.agents = vehicles
        controls, routes = [], []
        simulation.metsr_client = SimpleNamespace(
            control_vehicle=lambda vehicle, acc, **kwargs: controls.append((vehicle, acc))
            or metsr_response("controlVeh"),
            update_vehicle_route=lambda vehicle, route, **kwargs: routes.append(
                (vehicle, route)
            )
            or metsr_response("updateVehicleRoute"),
        )
        simulation.metsr_state = METSRCosimState(
            metadata=NetworkMetadata(lane_mapping={"0_-1": ["0_0"], "1_-1": ["1_0"]}),
            valid_roads=["0", "1"],
        )
        with patch(
            "scenic.simulators.cosim.simulator.DrivingSimulation.executeActions"
        ), patch.object(
            metsr_network,
            "generate_metsr_trajectory",
            wraps=metsr_network.generate_metsr_trajectory,
        ) as generate:
            simulation.executeActions({obj: () for obj in vehicles})
            simulation.executeActions({obj: () for obj in vehicles})
        self.assertEqual(controls, [(0, 1.5), (1, -1.0)] * 2)
        self.assertEqual(routes, [(0, ["0"]), (1, ["1"])])
        self.assertEqual(
            [call.args for call in generate.call_args_list],
            [(simulation.metsr_state, obj.trajectory) for obj in vehicles],
        )
        self.assertTrue(all(obj.trajectory_is_active for obj in vehicles))
        self.assertEqual(
            simulation.metsr_state.metadata.lane_mapping,
            {"0_-1": ("0_0",), "1_-1": ("1_0",)},
        )


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
        simulation = make_simulation()
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
        helper_patch = patch.object(
            runtime,
            "destroy_attached_carla_sensors",
            lambda _simulation, *args, **kwargs: simulation._destroy_attached_carla_sensors(
                *args, **kwargs
            ),
        )
        self.addCleanup(helper_patch.stop)
        helper_patch.start()
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
                cleanup_observations.append((self.vehicle.id, self.vehicle.is_alive))
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
                cleanup_pool_observations.append(tuple(provider._carla_actor_pool))
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
        simulation.metsr_state.controlled_roads = {"B": True, "A": True}
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

        simulation.metsr_client.reset = lambda: None
        simulation.release_roads = reject_release
        with redirect_stdout(StringIO()):
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
        simulation.metsr_state.controlled_roads = {}
        simulation.metsr_state.authoritative_segments = {obj: "A"}

        with redirect_stdout(StringIO()):
            simulation.destroy()

        self.assertEqual(simulation.metsr_state.authoritative_segments, {})
        self.assertIsNone(obj.carlaActor)

    def test_destroy_propagates_failed_reset_after_local_cleanup(self):
        actor = self.Actor()
        obj = SimpleNamespace(carlaActor=actor, pcla=None)
        simulation, recorder_calls = self._simulation(obj)
        simulation.metsr_state.controlled_roads = {"A": True}

        def reject_release(roads):
            raise METSRControlError("release contract violated")

        simulation.release_roads = reject_release

        def reject_reset():
            raise METSRControlError("reset failed")

        simulation.metsr_client.reset = reject_reset

        with redirect_stdout(StringIO()), self.assertRaisesRegex(
            METSRControlError, "reset failed"
        ):
            simulation.destroy()

        self.assertEqual(actor.destroy_calls, 1)
        self.assertEqual(recorder_calls, [True])


if __name__ == "__main__":
    unittest.main()
