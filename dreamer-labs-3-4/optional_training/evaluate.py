"""Matched stochastic evaluation of a completed NEW student A/C checkpoint."""

import argparse
import csv
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optional_training.common import (
    EVAL_SCHEMA, SEEDS, claim_output, configure_process, load_student_run,
    provenance, seed_value, sha256, write_json,
)
from optional_training.rewards import ACTION_NAMES, scores, settings, training_score, verify_environment

METRICS = (
    "mean_speed_mps", "displacement_m", "collision", "duration_s",
    "lane_changes", "episode_steps", "own_reward_return", "return_A", "return_C",
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--seeds", type=seed_value, nargs="+", default=list(SEEDS),
                        help="Omit for all 20 seeds; a subset is for smoke checks only")
    result.add_argument("--videos", action="store_true",
                        help="Save labeled GIF/MP4 decision-frame videos for seeds 1000-1002")
    return result


def save_videos(frames, output):
    import av

    frames[0].save(
        output.with_suffix(".gif"), save_all=True, append_images=frames[1:],
        duration=1000, loop=0)
    with av.open(str(output.with_suffix(".mp4")), "w") as container:
        stream = container.add_stream("libx264", rate=5)
        stream.width, stream.height = frames[0].size
        stream.pix_fmt = "yuv420p"
        for image in frames:
            for _ in range(5):
                for packet in stream.encode(av.VideoFrame.from_image(image)):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def episode(agent, style, weights, run_id, training_steps, seed, output, video):
    import numpy as np
    from embodied.envs.highway import HighwayEnv

    adapter = HighwayEnv("fast-v0", **settings(style, weights))
    frames, transitions, physics = [], [], []
    try:
        raw = verify_environment(adapter, style, weights)
        if video:
            raw.render_mode = "rgb_array"
            raw.config["offscreen_rendering"] = True
        raw_obs, _ = adapter._env.reset(seed=seed)
        adapter._done = False
        initial = raw.vehicle.position.copy()
        initial_speed = float(raw.vehicle.speed)
        initial_lane = int(raw.vehicle.lane_index[2])
        stats = {"speed_integral": 0.0, "time": 0.0, "lane_changes": 0}
        original_step = raw.road.step

        def measure(dt):
            vehicle = raw.vehicle
            previous_speed, previous_lane = vehicle.speed, vehicle.lane_index
            original_step(dt)
            stats["speed_integral"] += (previous_speed + vehicle.speed) * 0.5 * dt
            stats["time"] += dt
            stats["lane_changes"] += int(vehicle.lane_index != previous_lane)
            physics.append({
                "time_s": stats["time"], "speed_mps": float(vehicle.speed),
                "position": vehicle.position.tolist(), "lane": int(vehicle.lane_index[2]),
                "crashed": bool(vehicle.crashed), "on_road": bool(vehicle.on_road),
            })

        raw.road.step = measure

        def capture(step):
            from PIL import Image, ImageDraw

            road = Image.fromarray(adapter._env.render())
            frame = Image.new("RGB", (road.width, road.height + 54), "#102238")
            frame.paste(road, (0, 54))
            draw = ImageDraw.Draw(frame)
            draw.text((8, 4), (
                f"STUDENT {style}-style | run {run_id[:8]} | trained {training_steps} steps"
            ), fill="white")
            draw.text((8, 21), (
                f"seed {seed} | t={step}s | v={raw.vehicle.speed:.1f} m/s | "
                f"crashed={raw.vehicle.crashed}"
            ), fill="white")
            frames.append(frame)

        counter = (2000 + seed) * 1000
        with agent.n_actions.lock:
            agent.n_actions.value = counter
        carry = agent.init_policy(1)
        reward = 0.0
        counts = np.zeros(5, np.int64)
        totals = {"A": 0.0, "C": 0.0}
        own_return = 0.0
        if video:
            capture(0)
        for step in range(1, 32):
            observation = {
                "obs": np.asarray(raw_obs, np.float32)[None],
                "reward": np.asarray([reward], np.float32),
                "is_first": np.asarray([step == 1]),
                "is_last": np.asarray([False]), "is_terminal": np.asarray([False]),
            }
            available = list(map(int, raw.action_type.get_available_actions()))
            carry, action, _ = agent.policy(carry, observation, mode="eval")
            selected = int(action["action"][0])
            counts[selected] += 1
            result = adapter.step({"reset": False, "action": selected})
            raw_obs, reward = result["obs"], float(result["reward"])
            cross = scores(raw, selected)
            if not math.isclose(reward, training_score(raw, selected, style), rel_tol=1e-6, abs_tol=4e-6):
                raise RuntimeError("Evaluation reward does not match the checkpoint's training design")
            if not np.isfinite(raw_obs).all():
                raise RuntimeError("Non-finite evaluation observation")
            own_return += reward
            for key, value in cross.items():
                totals[key] += value
            transitions.append({
                "decision": step, "action": selected, "action_name": ACTION_NAMES[selected],
                "available_actions": available, "selected_available": selected in available,
                "time_s": float(raw.time), "position": raw.vehicle.position.tolist(),
                "speed_mps": float(raw.vehicle.speed), "reward": reward,
                "reward_A": cross["A"], "reward_C": cross["C"],
                "is_last": bool(result["is_last"]),
                "is_terminal": bool(result["is_terminal"]),
                "components": {key: float(value) for key, value in raw._rewards(selected).items()},
            })
            if video:
                capture(step)
            if result["is_last"]:
                break
        else:
            raise RuntimeError("Episode exceeded the unchanged 30-second limit")
        elapsed = stats["time"]
        record = {
            "seed": seed, "reward_style": style, "student_run_id": run_id,
            "training_steps": training_steps, "policy_rng_counter_start": counter,
            "initial_position": initial.tolist(), "initial_speed_mps": initial_speed,
            "initial_lane": initial_lane,
            "mean_speed_mps": stats["speed_integral"] / elapsed,
            "displacement_m": float(raw.vehicle.position[0] - initial[0]),
            "collision": int(raw.vehicle.crashed), "duration_s": elapsed,
            "lane_changes": stats["lane_changes"], "episode_steps": step,
            "own_reward_return": own_return, "return_A": totals["A"], "return_C": totals["C"],
            "action_counts": dict(zip(ACTION_NAMES, map(int, counts))),
            "action_distribution": dict(zip(ACTION_NAMES, map(float, counts / step))),
            "terminated": bool(result["is_terminal"]),
            "truncated": bool(result["is_last"] and not result["is_terminal"]),
            "live_environment_config": raw.config,
        }
        write_json(output / f"seed-{seed}.json", {
            "summary": record, "transitions": transitions, "physics_5hz": physics})
        if video:
            save_videos(frames, output / f"student-{style}-step-{training_steps}-seed-{seed}")
        return record
    finally:
        adapter.close()


def run(args):
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(SEEDS):
        raise ValueError("Choose distinct seeds from 1000-1019")
    # Validate the new destination before touching or unpickling any input.
    from optional_training.common import validate_output
    validate_output(args.output, (args.run_dir,))
    configure_process(Path(args.output).expanduser().resolve())
    run_dir, manifest, completed, config, checkpoint = load_student_run(args.run_dir)
    output = claim_output(args.output, (run_dir,))
    started = time.monotonic()
    try:
        import elements
        import numpy as np
        from dreamerv3.main import make_agent

        before = sha256(checkpoint / "agent.pkl")
        flat = dict(config.flat)
        flat.update({
            "logdir": str(output / "initialization"),
            "jax.platform": "cpu", "jax.mock_devices": 0,
            "jax.policy_devices": [0], "jax.train_devices": [0],
            "jax.precompile": False,
        })
        inference_config = elements.Config(flat)
        inference_config.save(str(output / "inference-config.yaml"))
        write_json(output / "evaluation.json", {
            "schema": EVAL_SCHEMA, "student_run_id": manifest["run_id"],
            "historical_experiment": False, "reward_style": manifest["reward_style"],
            "reward_weights": manifest.get("reward_weights"),
            "training_steps": completed["actual_steps"], "seeds": args.seeds,
            "checkpoint_sha256": before, "model_seed": manifest["model_seed"],
            "training_seed_protocol": manifest.get("seed_protocol", "legacy-unseeded-phase2a"),
            "protocol": "Stochastic policy; env.reset(seed=s); n_actions=(2000+s)*1000 before init_policy",
            "scope": "20_common_seeds" if sorted(args.seeds) == list(SEEDS) else "seed_subset_smoke",
            "videos": args.videos, "provenance": provenance(),
        })
        agent = make_agent(inference_config)
        loader = elements.Checkpoint()
        loader.agent = agent
        loader.load(str(checkpoint), keys=["agent"])
        loaded_updates = int(agent.n_updates)
        if loaded_updates != completed["optimizer_updates"]:
            raise RuntimeError("Loaded checkpoint update counter mismatch")
        records = [
            episode(agent, manifest["reward_style"], manifest.get("reward_weights"), manifest["run_id"],
                    completed["actual_steps"], seed, output,
                    args.videos and seed in (1000, 1001, 1002))
            for seed in args.seeds]
        if int(agent.n_updates) != loaded_updates or sha256(checkpoint / "agent.pkl") != before:
            raise RuntimeError("Evaluation changed training/checkpoint state")
        if sha256(run_dir / "config.yaml") != manifest["config_sha256"]:
            raise RuntimeError("Evaluation changed the source configuration")
        columns = ["seed", "reward_style", "training_steps", *METRICS,
                   *[f"count_{name}" for name in ACTION_NAMES]]
        with (output / "per-seed.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({
                **row, **{f"count_{key}": value for key, value in row["action_counts"].items()}
            } for row in records)
        action_counts = {
            name: sum(row["action_counts"][name] for row in records) for name in ACTION_NAMES}
        total = sum(action_counts.values())
        summary = {
            "schema": EVAL_SCHEMA, "student_run_id": manifest["run_id"],
            "reward_style": manifest["reward_style"], "seeds": args.seeds,
            "scope": "20_common_seeds" if sorted(args.seeds) == list(SEEDS) else "seed_subset_smoke",
            "training_steps": completed["actual_steps"],
            "metrics": {
                name: {
                    "mean": float(np.mean([row[name] for row in records])),
                    "sample_std": float(np.std([row[name] for row in records], ddof=1))
                    if len(records) > 1 else None,
                } for name in METRICS},
            "collision_count": sum(row["collision"] for row in records),
            "episode_count": len(records), "action_counts": action_counts,
            "pooled_action_distribution": {name: count / total for name, count in action_counts.items()},
            "checkpoint_sha256": before, "source_checkpoint_unchanged": True,
            "optimizer_updates_before_after": loaded_updates,
            "elapsed_seconds": time.monotonic() - started,
            "measurement": {
                "speed": "Time-weighted trapezoidal speed at actual 5 Hz physics steps",
                "displacement": "Final minus initial longitudinal x",
                "lane_changes": "Physical lane-index changes, not command counts",
                "returns": "Own training reward plus A/C cross-scores from live components",
                "video": "1 Hz decision frames; MP4 repeats each frame at 5 fps",
            },
        }
        write_json(output / "summary.json", summary)
        print(f"STUDENT_{manifest['reward_style']}_EVALUATED seeds={args.seeds}: {output}")
    except BaseException as error:
        write_json(output / "failed.json", {"error": str(error)})
        raise


if __name__ == "__main__":
    run(parser().parse_args())
