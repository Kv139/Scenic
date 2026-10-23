"""Passive native commuters do not force full-fleet Scenic property refreshes."""

import csv
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scenic.core.simulators import Simulation
from scenic.simulators.cosim.simulator import CosimSimulation
from scenic.simulators.cosim import runtime_utils
from scenic.simulators.cosim.tracking import (
    BackgroundRefreshSchedule, NeighborhoodTracker, neighborhood_keys,
)
from scenic.simulators.metsr import bridge_utils
from scenic.simulators.metsr.bridge_utils import METSRControlError, METSRCosimState


class Vehicle(SimpleNamespace):
    __hash__ = object.__hash__


def response(kind, ids):
    return {
        "messageType": kind, "status": "ok",
        "data": [{"vehicleId": value, "isPrivate": private} for private, value in ids],
    }


def simulation(background_count=5, *, initialized=True, hz=1.0):
    sim = object.__new__(CosimSimulation)
    sim._neighborhood_tracker = NeighborhoodTracker()
    sim.currentTime = 0
    sim.count = 0
    sim.timestep = 0.05
    sim._background_refresh = BackgroundRefreshSchedule(hz, sim.timestep)
    sim._pending_background_observation = None
    sim.metsr_state = METSRCosimState()
    sim.ego = Vehicle(name="ego", cosim_background=False, position=(0, 0))
    hazard = Vehicle(name="hazard", cosim_background=False, position=(1, 0))
    sim.objects = [sim.ego, hazard] + [
        Vehicle(name=f"background_{i}", cosim_background=True, position=(i + 2, 0))
        for i in range(background_count)
    ]
    sim.carla_actors = [sim.ego]
    sim.metsr_actors = sim.objects[1:]
    sim.completed_route = {}
    sim._boundary_vehicle_keys = set()
    for obj in sim.objects:
        sim.getMetsrPrivateVehId(obj)
    if initialized:
        sim._neighborhood_tracker.mark_updated(sim.objects, 0.0)
        sim._background_refresh.commit(0.0)
    sim.metsr_client = SimpleNamespace(
        query_cosim_vehicle=Mock(return_value=response("coSimVehicle", [])),
        query_boundary_vehicle=Mock(return_value=response("boundaryVehicle", [])),
    )
    return sim


def identity(sim, obj):
    return True, sim.metsr_state.private_vehicle_ids[obj]


def test_selection_preserves_identity_and_creation_order_across_departure_return():
    sim = simulation()
    tracker = sim._neighborhood_tracker
    a, b = sim.objects[2:4]
    keys = {(private, str(value)) for private, value in [identity(sim, b), identity(sim, a)]}
    assert tracker.select(keys) == tuple(sim.objects[:4])
    assert tracker.select(set()) == tuple(sim.objects[:2])
    assert tracker.select({next(key for key in keys if key[1] == str(identity(sim, a)[1]))}) == tuple(sim.objects[:3])
    assert len(tracker.objects_by_key) == len(sim.objects)
    assert sim.completed_route == {}


def test_public_id_does_not_activate_private_scenic_object():
    sim = simulation()
    obj = sim.objects[2]
    selected = sim._neighborhood_tracker.select({(False, str(identity(sim, obj)[1]))})
    assert obj not in selected


def test_owned_and_pending_remain_selected_without_index_membership():
    sim = simulation()
    owned, pending, retired = sim.objects[2:5]
    sim.completed_route[retired] = True
    keys = {(True, str(identity(sim, retired)[1]))}
    selected = sim._neighborhood_tracker.select(
        keys, owned=[owned], pending_ids=[identity(sim, pending)[1]], completed=sim.completed_route
    )
    assert selected == tuple(sim.objects[:4])
    assert retired not in selected


def test_ego_and_sensor_bearing_objects_cannot_be_made_dormant():
    tracker = NeighborhoodTracker()
    ego = Vehicle(cosim_background=True)
    sensor = Vehicle(cosim_background=True, sensors={"camera": object()})
    passive = Vehicle(cosim_background=True)
    for i, obj in enumerate((ego, sensor, passive)):
        tracker.register(obj, i, ego=obj is ego)
    tracker.mark_updated((ego, sensor, passive), 0.0)
    assert tracker.select(set()) == (ego, sensor)


@pytest.mark.parametrize("record", [
    {}, {"vehicleId": 5}, {"vehicleId": 5, "isPrivate": 1}, None,
    {"vehicleId": True, "isPrivate": True},
    {"vehicleId": 5.5, "isPrivate": True},
    {"vehicleId": " ", "isPrivate": True},
])
def test_malformed_neighborhood_identity_is_not_treated_as_empty(record):
    value = {"messageType": "coSimVehicle", "status": "ok", "data": [record]}
    with pytest.raises(METSRControlError):
        neighborhood_keys(value, "coSimVehicle")


def test_snapshot_reuses_pre_boundary_and_refreshes_both_indexes_after_tick(monkeypatch):
    sim = simulation()
    near, boundary, arriving = sim.objects[2:5]
    sim._boundary_vehicle_keys = {(True, str(identity(sim, boundary)[1]))}
    sim.metsr_client.query_cosim_vehicle.side_effect = [
        response("coSimVehicle", [identity(sim, near)]),
        response("coSimVehicle", [identity(sim, arriving)]),
    ]
    sim.metsr_client.query_boundary_vehicle.return_value = response("boundaryVehicle", [])
    snapshots = []

    def collect(client, objects):
        snapshots.append(tuple(objects))
        return {obj: {"sample": len(snapshots)} for obj in objects}

    monkeypatch.setattr(bridge_utils, "collect_metsr_vehicle_data", collect)
    first = sim._collect_tracked_vehicle_data(before_step=True)
    assert tuple(first) == tuple(sim.objects[:4])
    sim.metsr_client.query_boundary_vehicle.assert_not_called()
    second = sim._collect_tracked_vehicle_data()
    assert tuple(second) == (sim.ego, sim.objects[1], arriving)
    assert sim.objectsToUpdate() == tuple(second)
    assert sim.tracked_objects == tuple(sim.objects)
    assert sim.currentState() == tuple(obj.position for obj in sim.objects)
    assert all(record["sample"] == 2 for record in second.values())
    assert sim.completed_route == {}
    assert len(sim.objects) == 7


def test_opt_out_preserves_legacy_batch_and_default_core_iteration(monkeypatch):
    sim = simulation(background_count=0)
    calls = []
    monkeypatch.setattr(bridge_utils, "collect_metsr_vehicle_data", lambda client, objects: calls.append(tuple(objects)) or {})
    sim._collect_tracked_vehicle_data(before_step=True)
    sim._collect_tracked_vehicle_data()
    assert calls == [(sim.objects[1], sim.ego), (sim.objects[1], sim.ego)]
    sim.metsr_client.query_cosim_vehicle.assert_not_called()
    sim.metsr_client.query_boundary_vehicle.assert_not_called()
    assert sim.tracked_objects == tuple(sim.objects)
    assert Simulation.objectsToUpdate(sim) is sim.objects


def test_missing_from_neighborhood_does_not_retire_native_vehicle():
    sim = simulation()
    sim.road_pop_density = {}
    sim.update_bubble_objects(vehicle_data={})
    assert sim.completed_route == {}
    assert len(sim.metsr_actors) == 6


@pytest.mark.parametrize("fleet_size", [10, 1000, 10000])
def test_query_and_dynamic_property_work_scales_with_neighborhood(monkeypatch, fleet_size):
    sim = simulation(background_count=fleet_size)
    near = sim.objects[2:6]
    sim.metsr_client.query_cosim_vehicle.return_value = response("coSimVehicle", [identity(sim, obj) for obj in near])
    queries = []
    monkeypatch.setattr(bridge_utils, "collect_metsr_vehicle_data", lambda client, objects: queries.append(tuple(objects)) or {})
    sim._collect_tracked_vehicle_data()
    updates = []
    for obj in sim.objects:
        obj._simulatorProvidedProperties = {"speed": float}
        obj._recomputeDynamicFinals = Mock()
        obj._clearCaches = Mock()
    sim.getProperties = lambda obj, properties: updates.append(obj) or {"speed": 2.0}
    sim._replayOut = None
    sim.replayCanContinue = lambda: False
    Simulation.updateObjects(sim)
    expected = tuple(sim.objects[:2] + near)
    assert queries == [expected]
    assert tuple(updates) == expected
    assert len(queries[0]) == 6
    for obj in sim.objects[6:]:
        obj._recomputeDynamicFinals.assert_not_called()
        obj._clearCaches.assert_not_called()
    assert len(sim.objects) == fleet_size + 2


def test_changing_selection_preserves_matching_divergence_replay_order():
    sim = simulation()
    near = sim.objects[2:4]
    selections = [near, [], list(reversed(near))]
    for i, obj in enumerate(sim.objects):
        obj._simulatorProvidedProperties = {"speed": float}
        obj._recomputeDynamicFinals = lambda: None
        obj._clearCaches = lambda: None
        obj.speed = float(i)
    sim.getProperties = lambda obj, properties: {"speed": obj.speed}
    sim._replayOut = SimpleNamespace(writeValue=Mock())
    sim._writeDivergenceData = True
    sim.replayCanContinue = lambda: False
    for selection in selections:
        sim._neighborhood_tracker.select({(True, str(identity(sim, obj)[1])) for obj in selection})
        Simulation.updateObjects(sim)
    expected = [0., 1., 2., 3., 0., 1., 0., 1., 2., 3.]
    recorded = [call.args[0] for call in sim._replayOut.writeValue.call_args_list]
    assert recorded == expected
    sim._replayOut = None
    sim._replayIn = SimpleNamespace(readValue=lambda ty: recorded.pop(0))
    sim._checkDivergence = True
    sim.replayCanContinue = lambda: True
    for selection in selections:
        sim._neighborhood_tracker.select({(True, str(identity(sim, obj)[1])) for obj in selection})
        Simulation.updateObjects(sim)
    assert recorded == []


def test_local_statistics_label_scope_and_omit_unobserved_completions(tmp_path):
    sim = simulation()
    sim.bubble_sizes = []
    sim.total_active_vehicles = []
    sim.bubble_spawn_queue = set()
    sim.count = 0
    sim.run_name = str(tmp_path / "tracking")
    sim.road_pop_density = {"road_A": 2}
    for obj in sim.objects:
        obj.finished_route = False
        obj.trip_start = 2
    observed = sim.objects[2]
    observed.finished_route = 7
    sim.completed_route[observed] = True
    runtime_utils.record_step_statistics(sim)
    with open(sim.run_name + "_veh_data.csv", newline="") as stream:
        record = next(csv.DictReader(stream))
    assert record["road_population_scope"] == "pre_step_observed_scenic_objects"
    assert record["completed_routes_scope"] == "observed_scenic_objects"
    assert record["vehicle_count_scope"] == "registered_scenic_objects_minus_queue"
    runtime_utils.log_trip_times(sim)
    with open(sim.run_name + "_trip_logs.csv", newline="") as stream:
        trips = next(csv.DictReader(stream))
    assert float(trips[observed.name]) == 5.0
    assert trips[sim.objects[3].name] == ""


@pytest.mark.parametrize("hz", [0, -2, -0.01, float("inf"), float("nan"), True, "1", 20.1])
def test_background_frequency_rejects_invalid_or_faster_than_simulation(hz):
    with pytest.raises(ValueError, match="background_tracking_hz"):
        BackgroundRefreshSchedule(hz, 0.05)


def test_fractional_frequency_is_anchored_to_simulation_time_without_drift():
    schedule = BackgroundRefreshSchedule(3.0, 0.05)
    times = []
    for tick in range(41):
        time_s = tick * 0.05
        if schedule.due(time_s):
            times.append(round(time_s, 2))
            schedule.commit(time_s)
    assert times == [0.0, 0.35, 0.7, 1.0, 1.35, 1.7, 2.0]


def install_property_updates(sim, monkeypatch):
    queries = []
    for obj in sim.objects:
        obj._simulatorProvidedProperties = {"position": tuple, "speed": float}
        obj._recomputeDynamicFinals = lambda: None
        obj._clearCaches = lambda: None
        obj.velocity = (0.0, 0.0)
    sim._replayOut = None
    sim.replayCanContinue = lambda: False
    sim.bubble_spawn_queue = set()
    sim.records = {}
    sim.trajectory = [sim.currentState()]

    def collect(client, mapping):
        queries.append(tuple(mapping))
        return {obj: {
            "vehicleId": vehicle_id, "x": float(sim.currentTime), "y": float(vehicle_id),
            "speed": float(sim.currentTime), "segmentId": "A",
        } for obj, vehicle_id in mapping.items()}

    monkeypatch.setattr(bridge_utils, "collect_metsr_vehicle_data", collect)
    sim.getProperties = lambda obj, properties: {
        "position": (sim.obj_data_cache[obj]["x"], sim.obj_data_cache[obj]["y"]),
        "speed": sim.obj_data_cache[obj]["speed"],
    }
    return queries


def refresh(sim, time_s, **kwargs):
    sim.currentTime = round(time_s / sim.timestep)
    sim.obj_data_cache = sim._collect_tracked_vehicle_data(observation_time_s=time_s, **kwargs)
    sim.updateObjects()


def test_full_initial_periodic_and_nearby_updates_keep_all_trajectory_entries(monkeypatch):
    sim = simulation(initialized=False)
    queries = install_property_updates(sim, monkeypatch)
    near, far = sim.objects[2:4]
    sim.metsr_client.query_cosim_vehicle.return_value = response("coSimVehicle", [identity(sim, near)])
    assert all(value is None for value in sim.background_observation_times_s.values())
    refresh(sim, 0.0)
    assert queries[-1] == tuple(sim.objects)
    assert all(value == 0.0 for value in sim.background_observation_times_s.values())
    initial_far_position = far.position
    refresh(sim, 0.05)
    assert queries[-1] == (sim.ego, sim.objects[1], near)
    assert far.position == initial_far_position
    assert sim.background_observation_times_s[near] == 0.05
    assert sim.background_observation_ages_s[far] == pytest.approx(0.05)
    assert len(sim.currentState()) == len(sim.objects)
    refresh(sim, 0.95)
    assert queries[-1] == (sim.ego, sim.objects[1], near)
    refresh(sim, 1.0)
    assert queries[-1] == tuple(sim.objects)
    assert far.position != initial_far_position
    assert all(age == 0.0 for age in sim.background_observation_ages_s.values())


def test_full_step_frequency_refreshes_all_vehicles_each_step(monkeypatch):
    sim = simulation(hz=20.0)
    queries = install_property_updates(sim, monkeypatch)
    for time_s in (0.05, 0.1, 0.15):
        refresh(sim, time_s)
        assert queries[-1] == tuple(sim.objects)
        assert all(value == time_s for value in sim.background_observation_times_s.values())


def test_new_distant_vehicle_gets_first_observation_before_next_periodic_deadline(monkeypatch):
    sim = simulation()
    new = Vehicle(name="new", cosim_background=True, position=(9, 9))
    sim.objects.append(new)
    sim.metsr_actors.append(new)
    sim.getMetsrPrivateVehId(new)
    queries = install_property_updates(sim, monkeypatch)
    assert sim.background_observation_times_s[new] is None
    refresh(sim, 0.05)
    assert new in queries[-1]
    assert sim.background_observation_times_s[new] == 0.05
    refresh(sim, 0.1)
    assert new not in queries[-1]
    assert sim.background_observation_ages_s[new] == pytest.approx(0.05)


def test_failed_batch_does_not_advance_full_refresh_deadline_or_freshness(monkeypatch):
    sim = simulation()
    before = sim.background_observation_times_s
    sim.currentTime = 20
    monkeypatch.setattr(bridge_utils, "collect_metsr_vehicle_data", Mock(side_effect=METSRControlError("partial response")))
    with pytest.raises(METSRControlError, match="partial"):
        sim._collect_tracked_vehicle_data(observation_time_s=1.0)
    assert sim.background_observation_times_s == before
    assert sim._background_refresh.due(1.0)
    assert sim._pending_background_observation is None


def test_failed_property_update_does_not_mark_partially_updated_batch_fresh(monkeypatch):
    sim = simulation()
    install_property_updates(sim, monkeypatch)
    sim.currentTime = 20
    sim.obj_data_cache = sim._collect_tracked_vehicle_data(observation_time_s=1.0)
    sim.getProperties = Mock(side_effect=RuntimeError("bad property"))
    with pytest.raises(RuntimeError, match="bad property"):
        sim.updateObjects()
    assert all(time_s == 0.0 for time_s in sim.background_observation_times_s.values())
    assert sim._background_refresh.due(1.0)


def test_final_snapshot_refreshes_all_without_rewriting_held_time_series_rows(monkeypatch):
    sim = simulation()
    queries = install_property_updates(sim, monkeypatch)
    refresh(sim, 0.15)
    held = sim.currentState()
    sim.records["positions"] = [(3, held)]
    sim.trajectory = [held]
    sim._record_final_background_observation()
    final_tick, final = sim.records["background_final_observation"][0]
    assert final_tick == 3
    assert queries[-1] == tuple(sim.objects)
    assert final["time_s"] == pytest.approx(0.15)
    assert final["full_refresh"] is True
    assert all(age == 0 for age in final["last_update_ages_s"].values())
    assert len(final["positions"]) == len(sim.objects)
    assert all(value == pytest.approx(0.15) for value in final["last_update_times_s"].values())
    assert sim.records["positions"] == [(3, held)]
    assert sim.trajectory[-1] == sim.currentState()
    assert sim.trajectory[-1] != held


@pytest.mark.parametrize("error", [RuntimeError("failure"), KeyboardInterrupt()])
def test_failed_or_cancelled_run_does_not_query_final_snapshot(monkeypatch, error):
    sim = simulation()
    monkeypatch.setattr(Simulation, "_run", Mock(side_effect=error))
    sim._record_final_background_observation = Mock()
    with pytest.raises(type(error)) as caught:
        sim._run(object(), 10)
    assert caught.value is error
    sim._record_final_background_observation.assert_not_called()


def test_successful_run_saves_final_snapshot_after_core_returns(monkeypatch):
    sim = simulation()
    events = []
    monkeypatch.setattr(Simulation, "_run", lambda *args: events.append("core") or ("limit", "done"))
    sim._record_final_background_observation = lambda: events.append("final")
    assert sim._run(object(), 10) == ("limit", "done")
    assert events == ["core", "final"]


def test_disabled_schedule_never_becomes_due_even_after_initial_commit():
    schedule = BackgroundRefreshSchedule(-1, 0.05)
    assert schedule.enabled is False
    assert not schedule.due(0.0)
    schedule.commit(0.0)
    assert schedule.last_refresh_s == 0.0
    assert not schedule.due(100.0)
    assert not schedule.due(1000000.0)


def test_disabled_distant_refresh_preserves_initial_near_owned_pending_and_new(monkeypatch):
    sim = simulation(hz=-1, initialized=False)
    queries = install_property_updates(sim, monkeypatch)
    near, far, owned, pending = sim.objects[2:6]
    sim.metsr_client.query_cosim_vehicle.return_value = response("coSimVehicle", [identity(sim, near)])
    refresh(sim, 0.0, force_all=True)
    assert queries[-1] == tuple(sim.objects)
    initial_far = far.position
    sim.carla_actors.append(owned)
    sim.metsr_state.admitted_vehicles[identity(sim, pending)[1]] = {"admittedAt": 0}
    refresh(sim, 100.0)
    assert queries[-1] == (sim.ego, sim.objects[1], near, owned, pending)
    assert far.position == initial_far
    assert sim.background_observation_times_s[far] == 0.0
    assert sim.background_observation_ages_s[far] == 100.0
    assert sim.background_observation_times_s[near] == 100.0
    new = Vehicle(name="new", cosim_background=True, position=(9, 9))
    sim.objects.append(new)
    sim.metsr_actors.append(new)
    sim.getMetsrPrivateVehId(new)
    new._simulatorProvidedProperties = {"position": tuple, "speed": float}
    new._recomputeDynamicFinals = lambda: None
    new._clearCaches = lambda: None
    new.velocity = (0.0, 0.0)
    refresh(sim, 100.05)
    assert new in queries[-1] and far not in queries[-1]
    assert sim.background_observation_times_s[new] == 100.05
    assert len(sim.currentState()) == len(sim.objects)


def test_disabled_final_snapshot_has_held_values_and_ages_without_query(monkeypatch):
    sim = simulation(hz=-1)
    install_property_updates(sim, monkeypatch)
    refresh(sim, 0.15)
    held = sim.currentState()
    sim.trajectory = [held]
    sim._collect_tracked_vehicle_data = Mock(side_effect=AssertionError("unexpected final query"))
    sim.updateObjects = Mock(side_effect=AssertionError("unexpected final property update"))
    sim._record_final_background_observation()
    final = sim.records["background_final_observation"][0][1]
    assert final["background_tracking_hz"] == -1
    assert final["full_refresh"] is False
    far_id = str(identity(sim, sim.objects[2])[1])
    assert final["last_update_times_s"][far_id] == 0.0
    assert final["last_update_ages_s"][far_id] == pytest.approx(0.15)
    assert final["last_update_ages_s"][str(identity(sim, sim.ego)[1])] == pytest.approx(0.0)
    assert sim.trajectory == [held]
    sim._collect_tracked_vehicle_data.assert_not_called()
    sim.updateObjects.assert_not_called()


def test_sync_diagnostic_queries_only_current_owned_poses_not_held_scenic_values(monkeypatch, capsys):
    sim = simulation(hz=-1)
    sim.ego.position = (-999, -999)
    sim.ego.carlaActor = SimpleNamespace(
        is_alive=True, get_location=lambda: SimpleNamespace(x=10.0, y=20.0, z=0.0)
    )
    queried = []

    def collect(client, mapping):
        queried.append(tuple(mapping))
        return {sim.ego: {"x": 10.0, "y": -20.0, "segmentId": "A"}}

    monkeypatch.setattr(bridge_utils, "collect_metsr_vehicle_data", collect)
    runtime_utils.check_client_synchronization(sim)
    assert queried == [(sim.ego,)]
    assert capsys.readouterr().out == ""
    # Explicit held native requests cannot expand the diagnostic to the fleet.
    runtime_utils.check_client_synchronization(sim, objs=sim.objects)
    assert queried[-1] == (sim.ego,)
    assert capsys.readouterr().out == ""


def test_sync_diagnostic_still_reports_real_current_owner_disagreement(monkeypatch, capsys):
    sim = simulation(hz=-1)
    sim.ego.carlaActor = SimpleNamespace(
        is_alive=True, get_location=lambda: SimpleNamespace(x=10.0, y=20.0, z=0.0)
    )
    monkeypatch.setattr(bridge_utils, "collect_metsr_vehicle_data", lambda *args: {
        sim.ego: {"x": 15.0, "y": -20.0, "segmentId": "A"}
    })
    runtime_utils.check_client_synchronization(sim)
    output = capsys.readouterr().out
    assert "native/CARLA synchronization discrepancy" in output
    assert "(15.0, 10.0)" in output


def test_sync_diagnostic_skips_retired_and_unowned_actors_without_query(monkeypatch):
    sim = simulation(hz=-1)
    sim.completed_route[sim.ego] = True
    collect = Mock(side_effect=AssertionError("unexpected fleet query"))
    monkeypatch.setattr(bridge_utils, "collect_metsr_vehicle_data", collect)
    runtime_utils.check_client_synchronization(sim)
    runtime_utils.check_client_synchronization(sim, objs=sim.objects[1:])
    collect.assert_not_called()
