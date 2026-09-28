"""Native vehicle velocities must stay in global coordinates at CARLA handoff."""
import math
import unittest

from scenic.core.vectors import Orientation
from scenic.simulators.metsr import cosim_utils as native


class NativeVelocityTests(unittest.TestCase):
    def test_world_velocity_is_independent_of_parent_orientation(self):
        directions = (
            (0, (0.0, 10.0, 0.0)),
            (90, (10.0, 0.0, 0.0)),
            (180, (0.0, -10.0, 0.0)),
            (270, (-10.0, 0.0, 0.0)),
            (45, (math.sqrt(50.0), math.sqrt(50.0), 0.0)),
        )
        for bearing, expected in directions:
            for parent_yaw in (0.0, math.pi / 2, -math.pi / 2, math.pi):
                with self.subTest(bearing=bearing, parent_yaw=parent_yaw):
                    obj = type("NativeObject", (), {})()
                    obj.parentOrientation = Orientation.fromEuler(parent_yaw, 0, 0)
                    state = {
                        "x": 1.0,
                        "y": 2.0,
                        "z": 0.0,
                        "bearing": bearing,
                        "speed": 10.0,
                    }
                    result = native.metsr_properties(state, obj.parentOrientation)
                    velocity = result["velocity"]
                    for actual, target in zip(velocity, expected):
                        self.assertAlmostEqual(actual, target, places=7)
                    # The pose yaw remains parent-relative even though velocity is global.
                    expected_yaw = math.radians(-bearing) - parent_yaw
                    self.assertAlmostEqual(
                        math.sin(result["yaw"]), math.sin(expected_yaw)
                    )
                    self.assertAlmostEqual(
                        math.cos(result["yaw"]), math.cos(expected_yaw)
                    )


if __name__ == "__main__":
    unittest.main()
