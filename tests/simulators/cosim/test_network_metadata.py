"""The scenario and simulator share one loaded network description."""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("carla")

from scenic.simulators.cosim import simulator as cosim
from scenic.simulators.metsr.network import NetworkMetadata


@pytest.mark.parametrize("metadata_supplied", [True, False])
def test_simulation_creation_reuses_loaded_network_metadata(
    monkeypatch, metadata_supplied
):
    metadata = NetworkMetadata(lane_mapping={"1_-1": ["road_A_0"]})
    loader = MagicMock(return_value=metadata)
    monkeypatch.setattr(cosim.metsr_network, "load_network_metadata", loader)
    monkeypatch.setattr(cosim.carla, "Client", MagicMock())
    monkeypatch.setattr(cosim, "METSRClient", MagicMock())
    simulation_factory = MagicMock()
    monkeypatch.setattr(cosim, "CosimSimulation", simulation_factory)

    simulator = cosim.CosimSimulator(
        carla_map="Town05",
        map_path="Town05.xodr",
        xml_map="Town05.net.xml",
        timestep=0.05,
        sim_timestep=0.05,
        render=False,
        network_metadata=metadata if metadata_supplied else None,
    )
    try:
        for _ in range(2):
            simulator.createSimulation(object(), timestep=None)
        assert simulator.network_metadata is metadata
        assert all(
            call.kwargs["network_metadata"] is metadata
            for call in simulation_factory.call_args_list
        )
        if metadata_supplied:
            loader.assert_not_called()
        else:
            loader.assert_called_once_with("Town05.net.xml")
    finally:
        simulator.destroy()
