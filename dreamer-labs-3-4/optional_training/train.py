"""Fresh, isolated A-style or exact C-style CPU training. No resume mode."""

import argparse
from functools import partial
import math
import pickle
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optional_training.common import (
    RUN_SCHEMA, claim_output, configure_process, provenance, sha256,
    step_budget, seed_value, training_config, write_json,
)
from optional_training.rewards import settings, training_score, verify_environment
from optional_training.seeding import PROTOCOL, replay_seed, seed_training_environment


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--reward", choices=("A", "C"), required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--steps", type=step_budget, default=24996)
    result.add_argument("--seed", type=seed_value, default=0,
                        help="Teaching base seed for model, policy, replay and per-worker episode resets")
    result.add_argument("--speed-weight", type=float, default=None)
    result.add_argument("--collision-penalty", type=float, default=None)
    return result


class CheckedEnvironment:
    def __init__(self, env, style, weights, seed, index, journal):
        import embodied

        self.env = env
        self.style = style
        adapter = env
        while isinstance(adapter, embodied.wrappers.Wrapper):
            adapter = adapter.env
        self.raw = verify_environment(adapter, style, weights)
        seed_training_environment(adapter, seed, index, journal)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def step(self, action):
        observation = self.env.step(action)
        if not observation["is_first"]:
            expected = training_score(self.raw, int(action["action"]), self.style)
            if not math.isclose(float(observation["reward"]), expected, rel_tol=1e-6, abs_tol=4e-6):
                raise RuntimeError("Training reward does not match the declared A/C equation")
        return observation

    def close(self):
        self.env.close()


def make_checked_env(config, style, weights, index):
    from dreamerv3.main import make_env

    return CheckedEnvironment(
        make_env(config, index), style, weights, config.seed, index,
        Path(config.logdir) / "training-seeds" / f"env-{index}.jsonl")


def make_seeded_replay(config, folder):
    import embodied
    from dreamerv3.main import make_replay

    if config.replay.fracs.uniform != 1.0:
        raise ValueError("Teaching seed protocol requires the existing uniform replay")
    replay = make_replay(config, folder)
    replay.sampler = embodied.replay.selectors.Uniform(replay_seed(config.seed))
    return replay


def run(args):
    values = (args.speed_weight, args.collision_penalty)
    if any(value is not None for value in values) and not all(value is not None for value in values):
        raise ValueError("Specify both reward weights")
    weights = None if values[0] is None else dict(zip(
        ("speed_weight", "collision_penalty"), values))
    settings(args.reward, weights)
    output = claim_output(args.output)
    configure_process(output)
    started = time.monotonic()
    try:
        import elements
        import embodied
        import portal
        from dreamerv3.main import make_agent, make_logger, make_stream
        from optional_training.smoke import environment_checks, c_reward_checks

        config = training_config(output, args.reward, args.steps, args.seed, weights)
        config.save(str(output / "config.yaml"))
        a_b, _ = environment_checks()
        c = c_reward_checks()
        write_json(output / "reward-preflight.json", {"A_B": a_b, "C": c})
        run_id = str(uuid.uuid4())
        write_json(output / "run.json", {
            "schema": RUN_SCHEMA, "run_id": run_id, "reward_style": args.reward,
            "reward_weights": weights,
            "historical_experiment": False, "fresh_initialization": True,
            "fresh_replay": True, "requested_steps": args.steps,
            "model_seed": args.seed, "training_reset_seed": args.seed,
            "seed_protocol": PROTOCOL, "replay_seed": replay_seed(args.seed),
            "training_seed_semantics": (
                "Model/policy base seed; domain-derived uniform replay seed; "
                "reset seed=SeedSequence([base,0x48575931,worker,episode]).uint32"),
            "training_seed_journals": "training-seeds/env-{worker}.jsonl",
            "config_sha256": sha256(output / "config.yaml"),
            "reward_checked_each_training_transition": True, "provenance": provenance(),
        })

        def init():
            elements.timer.global_timer.enabled = config.logger.timer

        portal.setup(
            errfile=False, clientkw=dict(logging_color="cyan"),
            serverkw=dict(logging_color="cyan"), initfns=[init], ipv6=config.ipv6)
        runner_args = elements.Config(
            **config.run, **{key: config[key] for key in (
                "replica", "replicas", "logdir", "batch_size", "batch_length",
                "report_length", "consec_train", "consec_report", "replay_context")})
        agents, drivers = [], []

        def create_agent():
            agent = make_agent(config)
            agents.append(agent)
            return agent

        original_driver = embodied.Driver

        class ClosingDriver(original_driver):
            def __init__(self, *values, **kwargs):
                super().__init__(*values, **kwargs)
                drivers.append(self)

        try:
            with patch.object(embodied, "Driver", ClosingDriver):
                embodied.run.train(
                    create_agent, partial(make_seeded_replay, config, "replay"),
                    partial(make_checked_env, config, args.reward, weights),
                    partial(make_stream, config), partial(make_logger, config),
                    runner_args)
        finally:
            for driver in drivers:
                driver.close()
        final = output / "ckpt" / (output / "ckpt/latest").read_text().strip()
        with (final / "step.pkl").open("rb") as stream:
            actual_steps = int(pickle.load(stream))
        if actual_steps != args.steps:
            raise RuntimeError(f"Expected {args.steps} steps, found {actual_steps}")
        if not (final / "done").is_file():
            raise RuntimeError("Final checkpoint was not completed")
        updates = int(agents[0].n_updates)
        write_json(output / "completed.json", {
            "run_id": run_id, "actual_steps": actual_steps,
            "optimizer_updates": updates, "checkpoint": str(final.relative_to(output)),
            "checkpoint_sha256": sha256(final / "agent.pkl"),
            "elapsed_seconds": time.monotonic() - started,
            "scope": "student_training" if args.steps == 24996 else "short_smoke_or_custom_budget",
        })
        print(f"STUDENT_{args.reward}_COMPLETE steps={actual_steps} updates={updates}: {output}")
    except BaseException:
        write_json(output / "failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    run(parser().parse_args())
