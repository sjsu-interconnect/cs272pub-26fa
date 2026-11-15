"""The two existing reward designs, not new environment or reward algorithms."""

A_CONFIG = {
    "high_speed_reward": 0.4,
    "collision_reward": -1.0,
    "right_lane_reward": 0.1,
    "lane_change_reward": 0.0,
    "normalize_reward": True,
    "reward_speed_range": [20, 30],
}
C_CONFIG = {
    "high_speed_reward": 1.0,
    "collision_reward": -40.0,
    "right_lane_reward": 0.0,
    "lane_change_reward": 0.0,
    "normalize_reward": False,
    "reward_speed_range": [20, 30],
}
ACTION_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]


def settings(style, weights=None):
    import copy
    import math

    if style not in ("A", "C"):
        raise ValueError("Reward style must be A or C")
    config = copy.deepcopy(A_CONFIG if style == "A" else C_CONFIG)
    if weights is not None:
        if set(weights) != {"speed_weight", "collision_penalty"}:
            raise ValueError("Expected speed_weight and collision_penalty")
        for value in weights.values():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Reward weights must be finite nonnegative numbers")
        config["high_speed_reward"] = float(weights["speed_weight"])
        config["collision_reward"] = -float(weights["collision_penalty"])
        if style == "A" and weights["speed_weight"] + weights["collision_penalty"] + 0.1 <= 0:
            raise ValueError("A's normalization range must be positive")
    return {
        "config": config,
        "reward_mode": "native" if style == "A" else "speed_collision",
    }


def scores(raw, action):
    """Cross-score live components/state; never infer these from old replay."""
    import numpy as np

    components = {key: float(value) for key, value in raw._rewards(action).items()}
    forward = raw.vehicle.speed * np.cos(raw.vehicle.heading)
    return {
        "A": (1 - components["collision_reward"] +
              0.1 * components["right_lane_reward"] +
              0.4 * components["high_speed_reward"]) /
             1.5 * components["on_road_reward"],
        "C": float(np.clip((forward - 20) / 10, 0, 1) -
                   40 * float(raw.vehicle.crashed)),
    }


def training_score(raw, action, style):
    if style == "A":
        components = raw._rewards(action)
        speed = raw.config["high_speed_reward"]
        penalty = -raw.config["collision_reward"]
        return float((penalty * (1 - components["collision_reward"]) +
                      0.1 * components["right_lane_reward"] +
                      speed * components["high_speed_reward"]) /
                     (penalty + 0.1 + speed) * components["on_road_reward"])
    import numpy as np

    forward = raw.vehicle.speed * np.cos(raw.vehicle.heading)
    return float(raw.config["high_speed_reward"] * np.clip((forward - 20) / 10, 0, 1)
                 + raw.config["collision_reward"] * float(raw.vehicle.crashed))


def verify_environment(adapter, style, weights=None):
    raw = adapter._env.unwrapped
    expected = settings(style, weights)
    if adapter._reward_mode != expected["reward_mode"]:
        raise ValueError("Unexpected reward adapter mode")
    for key, value in expected["config"].items():
        actual = raw.config[key]
        if key == "reward_speed_range":
            actual = list(actual)
        if actual != value:
            raise ValueError(f"Unexpected active reward setting {key}: {actual}")
    for key, value in {
        "offroad_terminal": False, "duration": 30,
        "policy_frequency": 1, "simulation_frequency": 5,
    }.items():
        if raw.config[key] != value:
            raise ValueError(f"Unexpected environment setting {key}")
    if raw.spec.id != "highway-fast-v0":
        raise ValueError("Expected highway-fast-v0")
    return raw
