"""Read-only CPU/reward preflight: no trainer, policy episode, or saved checkpoint."""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import site
import sys
import tempfile
import time
from unittest.mock import patch

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def digest(tree):
    import jax
    import numpy as np

    result = hashlib.sha256()
    for key, value in sorted(tree.items()):
        array = np.asarray(jax.device_get(value))
        result.update(key.encode())
        result.update(str((array.shape, array.dtype)).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def environment_checks():
    import numpy as np
    from embodied.envs.highway import HighwayEnv

    rows = []
    spaces = None
    for coefficient in (0.4, 0.8):
        env = HighwayEnv("fast-v0", config={"high_speed_reward": coefficient})
        try:
            observation = env.step({"reset": True, "action": 1})
            assert observation["obs"].shape == (5, 5)
            assert observation["obs"].dtype == np.float32
            assert observation["is_first"] and observation["reward"] == 0
            raw = env._env.unwrapped
            expected = {
                "high_speed_reward": coefficient, "collision_reward": -1,
                "right_lane_reward": 0.1, "lane_change_reward": 0,
                "normalize_reward": True, "offroad_terminal": False,
                "duration": 30, "simulation_frequency": 5, "policy_frequency": 1,
            }
            assert all(raw.config[key] == value for key, value in expected.items())
            assert list(raw.config["reward_speed_range"]) == [20, 30]
            assert list(raw.action_type.target_speeds) == [20, 25, 30]
            denominator = 1.1 + coefficient
            checks = []
            # Ephemeral reward fixtures only; no simulation transition or policy.
            raw.vehicle.heading = 0.0
            for speed in (20, 25, 30):
                for crashed in (False, True):
                    raw.vehicle.speed = speed
                    raw.vehicle.crashed = crashed
                    components = raw._rewards(1)
                    expected_reward = (
                        (1 - float(crashed) +
                         0.1 * components["right_lane_reward"] +
                         coefficient * np.clip((speed - 20) / 10, 0, 1)) /
                        denominator * components["on_road_reward"])
                    actual = float(raw._reward(1))
                    np.testing.assert_allclose(actual, expected_reward, atol=1e-12)
                    checks.append({
                        "forward_speed": speed, "collision": crashed,
                        "reward": actual,
                    })
            rows.append({
                "environment": raw.spec.id,
                "high_speed_reward": coefficient,
                "normalization_denominator": denominator,
                "numeric_checks": checks,
                "reset_only": True, "simulation_steps": 0,
            })
            spaces = env.obs_space, {
                key: value for key, value in env.act_space.items() if key != "reset"}
        finally:
            env.close()
    return rows, spaces


def c_reward_checks():
    import numpy as np
    from embodied.envs.highway import HighwayEnv
    from optional_training.rewards import settings, verify_environment

    env = HighwayEnv("fast-v0", **settings("C"))
    checks = []
    try:
        reset = env.step({"reset": True, "action": 1})
        raw = verify_environment(env, "C")
        for on_road in (True, False):
            if not on_road:
                raw.vehicle.position[1] = 1000
            assert bool(raw.vehicle.on_road) == on_road
            for speed, heading, base in (
                    (20, 0, 0), (25, 0, 0.5), (30, 0, 1),
                    (50, np.pi / 3, 0.5)):
                for crashed in (False, True):
                    raw.vehicle.speed = speed
                    raw.vehicle.heading = heading
                    raw.vehicle.crashed = crashed
                    env._done = False
                    with patch.object(env._env, "step", return_value=(
                            reset["obs"], 999.0, crashed, False, {})):
                        result = env.step({"reset": False, "action": 1})
                    expected = base - 40 * crashed
                    np.testing.assert_allclose(result["reward"], expected, atol=2e-6)
                    assert bool(result["is_last"]) == crashed
                    assert bool(result["is_terminal"]) == crashed
                    checks.append({
                        "speed": speed, "heading": heading,
                        "on_road": on_road, "collision": crashed,
                        "reward": float(result["reward"]),
                    })
        env._done = False
        with patch.object(env._env, "step", return_value=(
                reset["obs"], 999.0, False, True, {})):
            result = env.step({"reset": False, "action": 1})
        assert result["is_last"] and not result["is_terminal"]
        return {
            "reward_mode": "speed_collision", "numeric_checks": checks,
            "truncation_is_not_terminal": True, "simulation_steps": 0,
            "fixtures": "Real adapter; mocked physics return, not policy episodes",
        }
    finally:
        env.close()


def model_checks(config, spaces):
    import embodied.jax.nets as nn
    from embodied.jax import internal, transform
    from dreamerv3.agent import Agent
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    import ninjax as nj
    import numpy as np

    obs_space, act_space = spaces
    model = object.__new__(Agent)
    Agent.__init__(model, obs_space, act_space, config)
    observation = {
        "obs": jnp.zeros((1, 5, 5), jnp.float32),
        "is_first": jnp.ones((1,), bool),
    }

    def forward(obs):
        enc, dyn, dec, previous = model.init_policy(1)
        enc, _, tokens = model.enc(
            enc, obs, obs["is_first"], training=False, single=True)
        dyn, _, features = model.dyn.observe(
            dyn, tokens, previous, obs["is_first"], training=False, single=True)
        _, (prior, _) = model.dyn.imagine(
            dyn, {"action": jnp.ones((1,), jnp.int32)}, 1,
            training=False, single=True)
        _, _, decoded = model.dec(
            dec, features, obs["is_first"], training=False, single=True)
        tensor = model.feat2tensor(features)
        actor = model.pol(tensor, 1)["action"]
        value = model.val(tensor, 1).pred()
        slow_value = model.slowval(tensor, 1).pred()
        model.retnorm.stats()
        model.valnorm.stats()
        model.advnorm.stats()
        decoded = decoded["obs"].pred()
        if model.dec.symlog:
            decoded = nn.symexp(decoded)
        return {
            "posterior_deter": features["deter"],
            "prior_deter": prior["deter"],
            "decoded_observation": decoded,
            "action_probs": jnp.stack([
                actor.prob(jnp.full((1,), index, jnp.int32))
                for index in range(5)], -1),
            "reward": model.rew(tensor, 1).pred(),
            "continuation": model.con(tensor, 1).prob(1),
            "value": value, "slow_value": slow_value,
        }

    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("d",))
    mirrored = NamedSharding(mesh, PartitionSpec())
    obs_sharding = jax.tree.map(lambda _: mirrored, observation)
    seed = jax.random.PRNGKey(0)
    started = time.perf_counter()
    with mesh:
        params, param_sharding = transform.init(
            forward, mesh, ({}, mirrored, obs_sharding),
            dummy_inputs=({}, seed, observation))
        fingerprint = digest(params)
        pure = nj.pure(forward)

        def readonly(params, obs, seed):
            return pure(params, obs, seed=seed, create=False, modify=False)

        apply = transform.apply(
            readonly, mesh, (param_sharding, mirrored, obs_sharding),
            (mirrored,), single_output=True)
        output = apply(params, seed, observation)
        repeated = apply(params, seed, observation)
        jax.block_until_ready(output)
        for key in output:
            assert np.isfinite(np.asarray(output[key], dtype=np.float32)).all()
            np.testing.assert_array_equal(output[key], repeated[key])
        assert output["posterior_deter"].shape == (1, 512)
        assert output["prior_deter"].shape == (1, 512)
        assert output["decoded_observation"].shape == (1, 5, 5)
        np.testing.assert_allclose(output["action_probs"].sum(-1), 1, atol=1e-6)
        assert np.all((output["continuation"] >= 0) & (output["continuation"] <= 1))
        gather, shard = internal.ckpt_fn(params)
        assert digest(shard(gather(params))) == fingerprint
        assert digest(params) == fingerprint
    return {
        "config": "defaults + highway + size1m; one CPU device",
        "parameter_entries": len(params),
        "shapes": {key: list(value.shape) for key, value in output.items()},
        "compute_dtype": str(nn.COMPUTE_DTYPE),
        "elapsed_seconds": time.perf_counter() - started,
        "repeat_identical": True, "readonly_parameter_digest": fingerprint,
        "transform_init_apply": True, "in_memory_checkpoint_sharding_roundtrip": True,
        "optimizer_initialized_or_called": False,
        "full_training_wrapper_initialized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="Optional new JSON file; existing files are not overwritten")
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 11), "Use CPython 3.11"
    assert sys.flags.isolated and site.ENABLE_USER_SITE is False, "Run python -I"
    prefix = Path(sys.prefix).resolve()
    assert prefix != Path(sys.base_prefix).resolve(), "Use a separate venv"
    assert prefix != (REPO / ".venv").resolve(), "Do not use the historical .venv"
    assert not any((REPO / ".venv").resolve() in Path(p).resolve().parents
                   for p in sys.path if p)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    with tempfile.TemporaryDirectory(prefix="dreamer-cpu-smoke-") as directory:
        os.environ["MPLCONFIGDIR"] = str(Path(directory) / "matplotlib")
        import elements
        import embodied
        import jax
        import numpy as np
        import scope
        from packaging.requirements import Requirement
        from dreamerv3.main import make_logger
        from ruamel.yaml import YAML

        versions = {}
        for line in (Path(__file__).parent / "requirements-cpu.lock").read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            req = Requirement(line)
            if req.marker and not req.marker.evaluate():
                continue
            dist = metadata.distribution(req.name)
            assert req.specifier.contains(dist.version), (req.name, dist.version)
            assert Path(dist.locate_file("")).resolve().is_relative_to(prefix)
            versions[req.name] = dist.version
        installed = {dist.metadata["Name"].lower().replace("_", "-")
                     for dist in metadata.distributions()}
        prohibited_packages = {
            name for name in installed
            if name.startswith(("nvidia-", "jax-cuda", "ale-py", "autorom",
                                "minerl", "procgen", "dm-control", "crafter"))}
        assert not prohibited_packages, prohibited_packages
        assert all(device.platform == "cpu" for device in jax.devices())
        configs = YAML(typ="safe").load((REPO / "dreamerv3/configs.yaml").read_text())
        config = elements.Config(configs["defaults"])
        for name in ("highway", "size1m"):
            config = config.update(configs[name])
        config = config.update({
            "jax.platform": "cpu", "jax.mock_devices": 0,
            "jax.policy_devices": [0], "jax.train_devices": [0],
            "logdir": str(Path(directory) / "logger"),
        })
        model_config = elements.Config(
            **config.agent, **{key: config[key] for key in (
                "logdir", "seed", "jax", "batch_size", "batch_length",
                "replay_context", "report_length", "replica", "replicas")})
        forbidden = [
            "dreamerv3.agent.Agent.train", "dreamerv3.agent.Agent.loss",
            "embodied.jax.Optimizer.__call__", "ninjax.grad",
            "jax.grad", "jax.value_and_grad",
            "embodied.jax.utils.Normalize.update",
            "embodied.jax.utils.SlowModel.update",
            "embodied.run.train", "embodied.run.train_eval", "embodied.run.eval_only",
        ]
        with ExitStack() as stack:
            mocks = [stack.enter_context(patch(
                name, side_effect=AssertionError(f"Forbidden call: {name}")))
                for name in forbidden]
            environments, spaces = environment_checks()
            c_reward = c_reward_checks()
            model = model_checks(model_config, spaces)
            logger = make_logger(config)
            logger.scalar("smoke/finite", np.float32(1))
            logger.write()
            logger.close()
            assert (Path(config.logdir) / "metrics.jsonl").is_file()
            assert (Path(config.logdir) / "scope").is_dir()
            assert all(not mock.called for mock in mocks)
        report = {
            "passed": True, "python": platform.python_version(),
            "platform": platform.platform(), "prefix": str(prefix),
            "base_prefix": sys.base_prefix, "isolated_python": True,
            "third_party_packages_from_fresh_venv": True,
            "devices": [str(device) for device in jax.devices()],
            "versions": versions, "environments": environments, "model": model,
            "c_reward": c_reward,
            "jsonl_scope_logger": True,
            "forbidden_calls": {name: 0 for name in forbidden},
            "checkpoint_loaded": False, "training_or_evaluation_experiment": False,
        }
        if args.output:
            with args.output.open("x") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
