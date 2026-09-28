"""The scenario and simulator share one loaded network description."""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("carla")

from scenic.simulators.cosim import simulator as cosim
from scenic.simulators.metsr.network_utils import NetworkMetadata


@pytest.mark.parametrize("metadata_supplied", [True, False])
@pytest.mark.parametrize(
    "render,record,run_name", [(False, "", None), (True, "recordings", "review")]
)
def test_simulation_creation_reuses_loaded_network_metadata(
    monkeypatch, metadata_supplied, render, record, run_name
):
    metadata = NetworkMetadata(lane_mapping={"1_-1": ["road_A_0"]})
    loader = MagicMock(return_value=metadata)
    monkeypatch.setattr(cosim.network_utils, "load_network_metadata", loader)
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
        render=render,
        record=record,
        run_name=run_name,
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
        for call in simulation_factory.call_args_list:
            assert call.kwargs["render"] is render
            assert call.kwargs["record"] == record
            assert call.kwargs["run_name"] == run_name
        if render:
            simulator.metsr_client.start_viz.assert_called_once()
        else:
            simulator.metsr_client.start_viz.assert_not_called()
        if metadata_supplied:
            loader.assert_not_called()
        else:
            loader.assert_called_once_with("Town05.net.xml")
    finally:
        simulator.destroy()
