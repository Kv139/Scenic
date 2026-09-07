from types import SimpleNamespace

from scenic.simulators.metsr.util import modify_property_file


def read_properties(path):
    properties = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key.strip()] = value.strip()
    return properties


def test_property_writer_hardens_old_template_for_synchronized_sim(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "Data.properties.Legacy").write_text(
        "NETWORK_LISTEN_PORT = 0\n"
        "SYNCHRONIZED = false\n"
        "ENABLE_NETWORK = false\n"
    )
    options = SimpleNamespace(
        no_lane_changing_length=3.5,
        max_stuck_time=750,
        intersection_collision_avoidance=True,
    )

    modify_property_file(
        options,
        str(source),
        str(destination),
        port=4100,
        instance=0,
        template="Legacy",
    )

    properties = read_properties(destination / "Data.properties")
    assert properties["NETWORK_LISTEN_PORT"] == "4100"
    assert properties["SYNCHRONIZED"] == "true"
    assert properties["ENABLE_NETWORK"] == "true"
    assert properties["ENABLE_INTERSECTION_SWEPT_COLLISION_CHECK"] == "true"
    assert properties["NO_LANECHANGING_LENGTH"] == "3.5"
    assert properties["LANE_CHANGE_LATERAL_SPEED"] == "1.0"
    assert properties["LANE_CHANGE_MIN_DURATION"] == "1.0"
    assert properties["MAX_ROAD_TRAVERSAL_PATIENCE"] == "750"
