"""A completed CARLA trip must retire its native shadow and never respawn."""

from collections import defaultdict
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scenic.simulators.metsr.cosim_utils import METSRControlError, METSRCosimState
from tests.simulators.cosim.helpers import make_simulation


class CompletedTripRetirementTests(unittest.TestCase):
    def fixture(self, retirement=None):
        obj = type("Vehicle", (), {})()
        obj.name = "car_87"
        obj.carla_actor_flag = True
        obj.carlaActor = object()
        obj.spawn_guard = 0
        simulation = make_simulation()
        simulation.metsr_state = METSRCosimState()
        simulation.ego = object()
        simulation.objects = [simulation.ego, obj]
        simulation.count = 250
        simulation.sim_ticks_per_carla = 1
        simulation.carla_actors = [obj]
        simulation.metsr_actors = []
        simulation.metsr_state.private_vehicle_ids = {obj: 88}
        simulation.completed_route = {}
        simulation.metsr_state.controlled_roads = {"Z": True, "A": True}
        simulation.metsr_state.controlled_segments = {"Z", "A"}
        simulation.bubble_roads = []
        simulation.road_pop_density = defaultdict(int)
        simulation.metsr_state.admitted_vehicles = {88: {"roadId": "Z"}}
        events = []
        simulation.accepted_segment = "Z"
        simulation.accepted_segment_type = "road"

        def teleport(vehicle_id, *args, **kwargs):
            events.append("pose accepted")
            return {
                "messageType": "teleportCoSimVeh",
                "status": "ok",
                "data": [
                    {
                        "vehicleId": vehicle_id,
                        "status": "ok",
                        "controlMode": "cosim",
                        "releasedFromCoSim": False,
                        "segmentId": simulation.accepted_segment,
                        "segmentType": simulation.accepted_segment_type,
                    }
                ],
            }

        def retire(vehicle_id, *, private_veh):
            self.assertTrue(obj.carla_actor_flag)
            self.assertNotIn(obj, simulation.completed_route)
            self.assertEqual(events, ["pose accepted"])
            self.assertEqual((vehicle_id, private_veh), (88, True))
            events.append("native retired")
            return (
                retirement
                if retirement is not None
                else {
                    "messageType": "reachDest",
                    "status": "ok",
                    "data": [{"status": "ok"}],
                }
            )

        def remove(target):
            self.assertIs(target, obj)
            self.assertIn(obj, simulation.completed_route)
            events.append("actor removed")
            obj.carla_actor_flag = False
            simulation.carla_actors.remove(obj)
            simulation.metsr_actors.append(obj)

        simulation.metsr_client = SimpleNamespace(
            teleport_cosim_vehicle=Mock(side_effect=teleport),
            reach_dest=Mock(side_effect=retire),
        )
        simulation.remove_bubble_object = Mock(side_effect=remove)
        simulation.createObjectInCarla = Mock(
            side_effect=AssertionError("completed actor respawned")
        )
        state = {
            "onRoad": True,
            "segmentId": "Z",
            "destinationRoadId": "Z",
            "controlMode": "cosim",
            "laneIndex": 0,
            "x": 1,
            "y": 2,
            "z": 0.5,
        }
        update = {
            "object": obj,
            "vehicle_id": 88,
            "x": 1,
            "y": 2,
            "z": 0.5,
            "bearing": 90,
            "speed": 8,
            "private_veh": True,
            "transform_coords": True,
            "authoritative_segment": "Z",
            "authoritative_lane": 0,
            "authoritative_connector_path": None,
            "logical_road": "Z",
            "vehicle_state": state,
            "cosim_record": {},
        }
        return simulation, obj, update, state, events

    def test_final_pose_is_accepted_then_native_retired_then_completion_recorded(self):
        simulation, obj, update, state, events = self.fixture()
        simulation._publish_carla_vehicle_updates([update])
        self.assertEqual(events, ["pose accepted", "native retired"])
        self.assertTrue(simulation.completed_route[obj])
        self.assertEqual(obj.finished_route, 250)
        self.assertTrue(obj.carla_actor_flag)
        simulation._remove_completed_carla_objects()
        self.assertEqual(events, ["pose accepted", "native retired", "actor removed"])

    def test_rejected_final_pose_does_not_retire_or_complete_vehicle(self):
        simulation, obj, update, _, _ = self.fixture()
        simulation.metsr_client.teleport_cosim_vehicle = Mock(
            return_value={
                "messageType": "teleportCoSimVeh",
                "status": "partial",
                "data": [
                    {"vehicleId": 88, "status": "error", "errorCode": "INVALID_POSE"}
                ],
            }
        )
        with self.assertRaises(METSRControlError):
            simulation._publish_carla_vehicle_updates([update])
        simulation.metsr_client.reach_dest.assert_not_called()
        self.assertNotIn(obj, simulation.completed_route)
        self.assertTrue(obj.carla_actor_flag)
        simulation.remove_bubble_object.assert_not_called()

    def test_rejected_or_ambiguous_retirement_keeps_actor_and_does_not_complete(self):
        invalid = [
            {"messageType": "reachDest", "status": "error", "data": []},
            {
                "messageType": "reachDest",
                "status": "partial",
                "data": [{"status": "ok"}],
            },
            {
                "messageType": "reachDest",
                "status": "ok",
                "data": [{"vehicleId": 88, "status": "error"}],
            },
            {"messageType": "reachDest", "status": "ok", "data": []},
            {"messageType": "reachDest", "status": "ok", "data": None},
            {"messageType": "reachDest", "status": "ok", "data": {"status": "ok"}},
            {"messageType": "reachDest", "status": "ok", "data": "ok"},
            {"messageType": "reachDest", "status": "ok", "data": [{}]},
            {
                "messageType": "reachDest",
                "status": "ok",
                "data": [{"vehicleId": 89, "status": "ok"}],
            },
            {
                "messageType": "reachDest",
                "status": "ok",
                "data": [{"status": "ok"}, {"status": "ok"}],
            },
            {"messageType": "reachDest", "status": "ok", "data": ["ok"]},
            {"messageType": "other", "status": "ok", "data": [{"status": "ok"}]},
        ]
        for response in invalid:
            with self.subTest(response=response):
                simulation, obj, update, _, _ = self.fixture(retirement=response)
                original_actor = obj.carlaActor
                with self.assertRaises(METSRControlError):
                    simulation._publish_carla_vehicle_updates([update])
                self.assertNotIn(obj, simulation.completed_route)
                self.assertFalse(hasattr(obj, "finished_route"))
                self.assertTrue(obj.carla_actor_flag)
                self.assertIs(obj.carlaActor, original_actor)
                self.assertEqual(simulation.carla_actors, [obj])
                simulation.remove_bubble_object.assert_not_called()

    def test_modern_acknowledgement_with_matching_vehicle_id_is_accepted(self):
        simulation, obj, update, _, _ = self.fixture(
            retirement={
                "messageType": "reachDest",
                "status": "ok",
                "data": [{"vehicleId": "88", "status": "ok"}],
            }
        )
        simulation._publish_carla_vehicle_updates([update])
        self.assertTrue(simulation.completed_route[obj])

    def test_completed_trip_cannot_respawn_from_stale_data_after_cleanup(self):
        simulation, obj, update, stale_state, events = self.fixture()
        simulation._publish_carla_vehicle_updates([update])
        # Promotion never removes actors; completion cleanup removes it once.
        # Later steps receive the same stale CoSim record.
        simulation.update_bubble_objects(vehicle_data={obj: stale_state})
        self.assertTrue(obj.carla_actor_flag)
        simulation._remove_completed_carla_objects()
        for _ in range(3):
            simulation.update_bubble_objects(vehicle_data={obj: stale_state})
            simulation._remove_completed_carla_objects()
        simulation.metsr_client.reach_dest.assert_called_once_with(88, private_veh=True)
        simulation.remove_bubble_object.assert_called_once_with(obj)
        simulation.createObjectInCarla.assert_not_called()
        self.assertEqual(events, ["pose accepted", "native retired", "actor removed"])
        self.assertNotIn(88, simulation.metsr_state.admitted_vehicles)

    def test_queued_new_native_vehicle_is_not_prematurely_completed(self):
        simulation, obj, _, _, _ = self.fixture()
        obj.carla_actor_flag = False
        queued = {"onRoad": False, "segmentId": None, "queuedRoadId": "Z"}
        simulation.update_bubble_objects(vehicle_data={obj: queued})
        self.assertNotIn(obj, simulation.completed_route)
        simulation.metsr_client.reach_dest.assert_not_called()
        simulation.createObjectInCarla.assert_not_called()
        simulation.remove_bubble_object.assert_not_called()

    def test_non_destination_and_destination_connector_do_not_retire(self):
        for segment, logical_road, path_id in (("A", "A", None), ("cont/A/Z", "Z", 0)):
            with self.subTest(segment=segment):
                simulation, obj, update, _, _ = self.fixture()
                simulation.accepted_segment = segment
                simulation.accepted_segment_type = (
                    "road" if path_id is None else "connector"
                )
                simulation._publish_carla_vehicle_updates([update])
                self.assertNotIn(obj, simulation.completed_route)
                simulation.metsr_client.reach_dest.assert_not_called()

    def test_ego_completion_preserves_live_authoritative_shadow(self):
        simulation, obj, update, _, _ = self.fixture()
        simulation.ego = obj
        simulation._publish_carla_vehicle_updates([update])
        self.assertTrue(simulation.completed_route[obj])
        self.assertTrue(obj.carla_actor_flag)
        simulation.metsr_client.reach_dest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
