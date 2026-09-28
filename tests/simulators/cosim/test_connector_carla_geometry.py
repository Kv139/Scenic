"""Connector placement geometry includes elevation and rejects missing curves."""
import math
import unittest

from scenic.simulators.carla.utils import routing as carla_routing

NATIVE_TURN_POINTS = [
    (-173.67, 4.02, 0),
    (-178.44, 4.69, 0.02),
    (-178.62, 4.8, 0.02),
    (-181.85, 6.73, 0.03),
    (-183.89, 10.13, 0.04),
    (-184.58, 14.9, 0.06),
]
CAPTURED_POSITION = (-187.240966796875, 7.059037208557129, 0.5407504439353943)


class ConnectorGeometryTests(unittest.TestCase):
    def test_captured_pose_distance_to_native_curve(self):
        self.assertAlmostEqual(
            carla_routing.point_to_polyline_distance_3d(
                CAPTURED_POSITION, NATIVE_TURN_POINTS
            ),
            4.4817482786,
            places=6,
        )

    def test_geometry_distance_includes_elevation(self):
        x, y, z = CAPTURED_POSITION
        self.assertGreater(
            carla_routing.point_to_polyline_distance_3d(
                (x, y, z + 8), NATIVE_TURN_POINTS
            ),
            8,
        )

    def test_nonfinite_geometry_has_no_valid_projection(self):
        points = list(NATIVE_TURN_POINTS)
        points[2] = (*points[2][:2], math.nan)
        self.assertEqual(
            carla_routing.point_to_polyline_distance_3d(CAPTURED_POSITION, points),
            math.inf,
        )

    def test_missing_geometry_has_no_valid_projection(self):
        self.assertEqual(
            carla_routing.point_to_polyline_distance_3d(CAPTURED_POSITION, []), math.inf
        )

    def test_missing_elevation_has_no_valid_projection(self):
        points = [point[:2] for point in NATIVE_TURN_POINTS]
        self.assertEqual(
            carla_routing.point_to_polyline_distance_3d(CAPTURED_POSITION, points),
            math.inf,
        )

    def test_zero_length_connector_uses_its_single_position(self):
        point = NATIVE_TURN_POINTS[0]
        self.assertAlmostEqual(
            carla_routing.point_to_polyline_distance_3d(
                CAPTURED_POSITION, [point, point]
            ),
            math.dist(CAPTURED_POSITION, point),
        )
