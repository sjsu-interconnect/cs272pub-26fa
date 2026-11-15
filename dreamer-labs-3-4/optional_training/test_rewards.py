"""Regression checks for the student-editable reward weights."""

import unittest
from unittest.mock import patch

from embodied.envs.highway import HighwayEnv
from optional_training.rewards import settings, training_score, verify_environment


class RewardWeightsTest(unittest.TestCase):
    def test_native_a_matches_environment_with_changed_weights(self):
        weights = {"speed_weight": 1.0, "collision_penalty": 10.0}
        env = HighwayEnv("fast-v0", **settings("A", weights))
        try:
            env.step({"reset": True, "action": 1})
            raw = verify_environment(env, "A", weights)
            for crashed in (False, True):
                raw.vehicle.speed = 25.0
                raw.vehicle.heading = 0.0
                raw.vehicle.crashed = crashed
                self.assertAlmostEqual(training_score(raw, 1, "A"), raw._reward(1), places=6)
        finally:
            env.close()

    def test_c_adapter_uses_modified_weights(self):
        weights = {"speed_weight": 2.0, "collision_penalty": 10.0}
        env = HighwayEnv("fast-v0", **settings("C", weights))
        try:
            first = env.step({"reset": True, "action": 1})
            raw = verify_environment(env, "C", weights)
            for crashed in (False, True):
                raw.vehicle.speed = 25.0
                raw.vehicle.heading = 0.0
                raw.vehicle.crashed = crashed
                env._done = False
                with patch.object(env._env, "step", return_value=(
                        first["obs"], 999.0, crashed, False, {})):
                    result = env.step({"reset": False, "action": 1})
                self.assertAlmostEqual(float(result["reward"]), 1.0 - 10.0 * crashed)
                self.assertAlmostEqual(float(result["reward"]), training_score(raw, 1, "C"))
        finally:
            env.close()

    def test_invalid_weights_are_rejected(self):
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                settings("C", {"speed_weight": value, "collision_penalty": 10.0})


if __name__ == "__main__":
    unittest.main()
