"""Native metadata can be reused without XML or co-simulation imports."""

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scenic.simulators.metsr import network
from scenic.simulators.metsr.cosim_utils import METSRCosimState


def test_one_xml_load_serves_spawn_region_and_multiple_simulation_states(
    tmp_path, monkeypatch
):
    path = tmp_path / "network.xml"
    path.write_text(
        """<net>
      <edge id="source_opaque" type="highway.primary">
        <lane id="source_opaque_0" index="0" type="shoulder">
          <param key="origId" value="10_-2"/>
        </lane>
        <lane id="source_opaque_2" index="2" type="driving">
          <param key="origId" value="10_-1 11_-1"/>
        </lane>
      </edge>
      <edge id="target" type="highway.primary">
        <lane id="target_1" index="1" type="driving">
          <param key="origId" value="20_-1"/>
        </lane>
      </edge>
      <edge id=":opaque_connector" function="internal">
        <lane id=":opaque_connector_0" index="0"/>
      </edge>
      <connection from="source_opaque" to="target" fromLane="2" toLane="1" via=":opaque_connector_0"/>
      <tlLogic id="signal">
        <param key="linkSignalID:3" value="100 101"/>
      </tlLogic>
    </net>""",
        encoding="utf-8",
    )
    parse = Mock(wraps=network.ET.parse)
    monkeypatch.setattr(network.ET, "parse", parse)

    metadata = network.load_network_metadata(path)
    path.unlink()
    lanes = [
        SimpleNamespace(road=SimpleNamespace(id=10), id=-1),
        SimpleNamespace(road=SimpleNamespace(id=10), id=-2),
    ]
    region = object()
    union = Mock(return_value=region)
    monkeypatch.setattr(network.PolygonalRegion, "unionAll", union)
    keys, actual_region = network.mapped_driving_region(
        SimpleNamespace(lanes=lanes), metadata
    )
    states = [METSRCosimState(metadata=metadata) for _ in range(2)]

    parse.assert_called_once_with(path)
    assert keys == {"10_-1", "11_-1", "20_-1"}
    assert actual_region is region
    union.assert_called_once_with((lanes[0],))
    assert metadata.internal_lane_connections == {
        ":opaque_connector_0": (("source_opaque_2", "target_1"),)
    }
    assert metadata.internal_edge_connections == {
        ":opaque_connector": (("source_opaque_2", "target_1"),)
    }
    assert metadata.signal_mapping == {"signal_3": ("100", "101")}
    assert all(
        network.mapped_lane_query(state, "source_opaque_2") == ("source_opaque", 0)
        for state in states
    )
    assert all(
        state.metadata.roads_by_scenic["11"] == {"source_opaque"} for state in states
    )


def test_native_routing_import_does_not_require_cosim_package():
    pytest.importorskip("carla")
    code = f"""import sys
sys.path = {sys.path!r}
sys.modules['scenic.simulators.cosim'] = None
import scenic.simulators.metsr.cosim_routing_utils
assert not any(name.startswith('scenic.simulators.cosim.') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_metadata_and_state_import_without_carla_or_cosim():
    code = f"""import sys
sys.path = {sys.path!r}
sys.modules['carla'] = None
sys.modules['scenic.simulators.cosim'] = None
import scenic.simulators.metsr.network
import scenic.simulators.metsr.cosim_utils
assert not any(name.startswith('scenic.simulators.cosim.') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
