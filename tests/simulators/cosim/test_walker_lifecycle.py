"""Walker removal shares mechanics while retaining co-simulation cleanup policy."""

from types import SimpleNamespace

import pytest

carla = pytest.importorskip("carla")

from scenic.simulators.cosim import runtime_utils as runtime
from scenic.simulators.cosim.simulator import CosimSimulation
from scenic.simulators.metsr.bridge_utils import METSRCosimState


def test_ordinary_walker_removal_stops_controller_and_destroys_actor(monkeypatch):
    events = []

    class Walker:
        def destroy(self):
            events.append("actor")

    class Object:
        pass

    monkeypatch.setattr(carla, "Walker", Walker)
    monkeypatch.setattr(
        runtime,
        "destroy_attached_carla_sensors",
        lambda simulation, actor: events.append("sensors"),
    )
    obj = Object()
    obj.carlaActor = Walker()
    obj.carlaController = SimpleNamespace(
        stop=lambda: events.append("controller stop"),
        destroy=lambda: events.append("controller destroy"),
    )
    simulation = object.__new__(CosimSimulation)
    simulation.ego = None
    simulation.metsr_state = METSRCosimState()
    simulation._carla_destination_plans = {}
    simulation._carla_spawn_property_seeds = {}

    simulation.destroy_carla_obj(obj)

    assert events == ["controller stop", "controller destroy", "sensors", "actor"]
    assert obj.carlaActor is None
