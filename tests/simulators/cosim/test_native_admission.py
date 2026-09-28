"""CARLA admission must use the current native pose, including after queue entry."""

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

from scenic.simulators.cosim.simulator import carla
from tests.simulators.cosim.helpers import make_admission_simulation


class NativeAdmissionTests(unittest.TestCase):

    def test_admission_uses_one_current_pose_and_velocity_snapshot(self):
        simulation, obj, state, events = make_admission_simulation()
        self.assertTrue(
            simulation.createObjectInCarla(obj, update_velocity=True, native_state=state)
        )
        transform, velocity = events[0][1], events[1][1]
        self.assertEqual(
            (transform.location.x, transform.location.y, transform.location.z),
            (12.5, 23.5, 10.0),
        )
        self.assertAlmostEqual(abs(transform.rotation.yaw), 180.0, places=5)
        self.assertAlmostEqual(velocity.x, -8.0, places=5)
        self.assertAlmostEqual(velocity.y, 0.0, places=5)
        self.assertAlmostEqual(velocity.z, 0.0, places=5)
        self.assertEqual(tuple(obj.position), (-999, 999, 0))

    def test_ground_snap_queries_the_current_native_elevation(self):
        simulation, obj, state, events = make_admission_simulation(snap=True)
        self.assertTrue(
            simulation.createObjectInCarla(obj, update_velocity=True, native_state=state)
        )
        self.assertEqual(events[0], ("ground_query", (12.5, 23.5, 10.0)))
        self.assertAlmostEqual(events[1][1].location.z, 10.5)

    def test_all_compass_headings_use_global_orientation_and_velocity(self):
        for bearing in (0.0, 90.0, 180.0, 270.0, 45.0):
            with self.subTest(bearing=bearing):
                simulation, obj, state, events = make_admission_simulation()
                state["bearing"] = bearing
                simulation.createObjectInCarla(
                    obj, update_velocity=True, native_state=state
                )
                transform, velocity = events[0][1], events[1][1]
                yaw = math.radians(transform.rotation.yaw)
                self.assertAlmostEqual(velocity.x, 8.0 * math.cos(yaw), places=5)
                self.assertAlmostEqual(velocity.y, 8.0 * math.sin(yaw), places=5)
                self.assertAlmostEqual(
                    (transform.rotation.yaw + 90) % 360, bearing, places=5
                )

    def degenerate_connector_fixture(self):
        simulation, obj, state, events = make_admission_simulation()
        state.update(
            x=7.089999999999976,
            y=200.48000000000002,
            z=0.02,
            bearing=179.6725991095069,
            speed=0.28945161722131957,
            connectorId="opaque/continuation",
            connectorPathId=2,
            sourceRoadId="-36",
            targetRoadId="-37",
            connectorPathViaLaneIds=[":37.36_0_5"],
        )
        simulation.metsr_state.controlled_roads = {"-37": True}
        simulation.metsr_state.connector_records = {
            "opaque/continuation": {"sourceRoadId": "-36", "targetRoadId": "-37"}
        }
        simulation.metsr_state.road_connectors = {"-37": {"opaque/continuation"}}
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata,
            internal_lane_connections={":37.36_0_5": [("-36_5", "-37_5")]},
        )
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, lane_mapping={"37_-1": ["-37_5"]}
        )
        simulation.metsr_state.connector_centerline_cache = {
            ("opaque/continuation", 2): [(7.09, 200.48, 0.02), (7.09, 200.48, 0.02)]
        }
        waypoint = SimpleNamespace(
            road_id=37,
            lane_id=-1,
            transform=carla.Transform(
                carla.Location(
                    7.0856804847717285, -200.48301696777344, 0.02132006548345089
                ),
                carla.Rotation(yaw=179.757080078125),
            ),
        )
        simulation.map = SimpleNamespace(get_waypoint=lambda *a, **k: waypoint)
        return simulation, obj, state, events, waypoint

    def test_captured_zero_length_connector_uses_its_mapped_lane_direction(self):
        simulation, obj, state, events, waypoint = self.degenerate_connector_fixture()
        simulation.createObjectInCarla(obj, update_velocity=True, native_state=state)
        transform, velocity = events[0][1], events[1][1]
        self.assertAlmostEqual(
            (transform.rotation.yaw - waypoint.transform.rotation.yaw + 180) % 360,
            180.0,
            places=5,
        )
        forward = transform.get_forward_vector()
        self.assertAlmostEqual(velocity.x, state["speed"] * forward.x, places=5)
        self.assertAlmostEqual(velocity.y, state["speed"] * forward.y, places=5)
        self.assertAlmostEqual(transform.location.x, state["x"], places=5)
        self.assertAlmostEqual(transform.location.y, -state["y"], places=5)

    def test_valid_connector_geometry_retains_native_heading(self):
        simulation, obj, state, events, _ = self.degenerate_connector_fixture()
        simulation.metsr_state.connector_centerline_cache[("opaque/continuation", 2)][
            1
        ] = (
            6.09,
            200.48,
            0.02,
        )
        simulation.createObjectInCarla(obj, update_velocity=True, native_state=state)
        self.assertAlmostEqual(events[0][1].rotation.yaw, state["bearing"] - 90, places=5)

    def test_wrong_mapped_lane_cannot_supply_connector_direction(self):
        simulation, obj, state, events, _ = self.degenerate_connector_fixture()
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, lane_mapping={"37_-1": ["-37_4"]}
        )
        simulation.createObjectInCarla(obj, update_velocity=True, native_state=state)
        self.assertAlmostEqual(events[0][1].rotation.yaw, state["bearing"] - 90, places=5)

    def test_other_road_elevation_cannot_supply_connector_direction(self):
        simulation, obj, state, events, waypoint = self.degenerate_connector_fixture()
        waypoint.transform.location = carla.Location(7.08568, -200.48302, 8.02)
        simulation.createObjectInCarla(obj, update_velocity=True, native_state=state)
        self.assertAlmostEqual(events[0][1].rotation.yaw, state["bearing"] - 90, places=5)

    def test_missing_geometry_cannot_supply_connector_direction(self):
        simulation, obj, state, events, _ = self.degenerate_connector_fixture()
        simulation.metsr_state.connector_centerline_cache[("opaque/continuation", 2)] = []
        simulation.createObjectInCarla(obj, update_velocity=True, native_state=state)
        self.assertAlmostEqual(events[0][1].rotation.yaw, state["bearing"] - 90, places=5)

    def test_missing_topology_cannot_supply_connector_direction(self):
        simulation, obj, state, events, _ = self.degenerate_connector_fixture()
        simulation.metsr_state.metadata = replace(
            simulation.metsr_state.metadata, internal_lane_connections={}
        )
        simulation.createObjectInCarla(obj, update_velocity=True, native_state=state)
        self.assertAlmostEqual(events[0][1].rotation.yaw, state["bearing"] - 90, places=5)

    def test_scenic_spawn_without_native_snapshot_keeps_sampled_pose(self):
        simulation, obj, state, events = make_admission_simulation()
        self.assertTrue(simulation.createObjectInCarla(obj))
        transform = events[0][1]
        self.assertEqual(
            (transform.location.x, transform.location.y, transform.location.z),
            (-999, -999, 0),
        )
        self.assertAlmostEqual(transform.rotation.yaw % 360, 90.0, places=5)
        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
