"""Portable configuration, provenance, and new-output-only safety helpers."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform

from optional_training.rewards import settings
from optional_training.seeding import PROTOCOL, replay_seed

REPO = Path(__file__).resolve().parents[1]
RUN_SCHEMA = "dreamerv3-student-ac-run/v1"
EVAL_SCHEMA = "dreamerv3-student-ac-evaluation/v1"
SEEDS = tuple(range(1000, 1020))
HISTORICAL_NAMES = {
    "highway-1m-multicpu", "highway-1m-speed08-seed0", "highway-1m-extreme-seed0",
}


def step_budget(value):
    try:
        value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Steps must be an integer") from error
    if not 12 <= value <= 24996 or value % 12:
        raise argparse.ArgumentTypeError("Steps must be a multiple of 12 in [12,24996]")
    return value


def seed_value(value):
    try:
        value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seed must be an integer") from error
    if not 0 <= value < 2 ** 32:
        raise argparse.ArgumentTypeError("Seed must be in [0,2**32)")
    return value


def validate_output(path, inputs=()):
    requested = Path(path).expanduser()
    if requested.is_symlink() or requested.exists():
        raise ValueError(f"Output must be a NEW directory, including empty directories: {path}")
    requested = Path(os.path.abspath(requested))
    results = REPO / "results"
    if requested.is_relative_to(results):
        for parent in (requested, *requested.parents):
            if parent.is_symlink():
                raise ValueError("Results paths must not contain symlinks")
            if parent == results:
                break
    output = requested.resolve()
    inside_repo = output.is_relative_to(REPO)
    if (inside_repo and (output == results or not output.is_relative_to(results))
            or any(name in HISTORICAL_NAMES for name in output.parts)):
        raise ValueError("Outputs must be outside the source tree and historical experiment trees")
    for source in inputs:
        source = Path(source).expanduser().resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Evaluation outputs must be separate from input runs")
    markers = (
        "config.yaml", "run.json", "agent.pkl", "metrics.jsonl", "scores.jsonl",
        "report.md", "report.html",
    )
    for parent in output.parents:
        if any((parent / marker).exists() for marker in markers):
            raise ValueError(f"Cannot create outputs within an existing run/evidence tree: {parent}")
    return output


def claim_output(path, inputs=()):
    output = validate_output(path, inputs)
    output.mkdir(parents=True, exist_ok=False)
    return output


def configure_process(output):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["MPLCONFIGDIR"] = str(output / "cache" / "matplotlib")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def provenance():
    names = (
        "jax", "jaxlib", "numpy", "ninjax", "elements", "portal", "scope",
        "highway-env", "gymnasium", "optax", "chex",
    )
    sources = (
        "dreamerv3/agent.py", "dreamerv3/rssm.py", "dreamerv3/configs.yaml",
        "embodied/envs/highway.py", "embodied/jax/agent.py",
        "embodied/jax/transform.py", "embodied/jax/internal.py",
        "embodied/run/train.py", "optional_training/common.py",
        "optional_training/rewards.py", "optional_training/train.py",
        "optional_training/evaluate.py", "optional_training/seeding.py",
    )
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in names},
        "source_sha256": {name: sha256(REPO / name) for name in sources},
        "dependency_lock_sha256": sha256(REPO / "optional_training/requirements-cpu.lock"),
        "source_origin": "Bundled verified working-tree snapshot; see PROVENANCE.md",
    }


def training_config(output, style, steps, seed, reward_weights=None):
    import elements
    from ruamel.yaml import YAML

    configs = YAML(typ="safe").load((REPO / "dreamerv3/configs.yaml").read_text())
    config = elements.Config(configs["defaults"])
    for name in ("highway", "size1m", "multicpu"):
        config = config.update(configs[name])
    flat = dict(config.flat)
    reward = settings(style, reward_weights)
    flat.update({
        "logdir": str(output), "seed": seed,
        "jax.platform": "cpu", "jax.precompile": False,
        "run.envs": 4, "run.debug": False, "run.steps": steps,
        "run.from_checkpoint": "", "run.usage.nvsmi": False,
        "env.highway.reward_mode": reward["reward_mode"],
    })
    flat.update({f"env.highway.config.{key}": value
                 for key, value in reward["config"].items()})
    return elements.Config(flat)


def load_student_run(path):
    import elements

    run = Path(path).expanduser().resolve()
    if any(name in HISTORICAL_NAMES for name in run.parts):
        raise ValueError("This entry point accepts only NEW student runs, not historical A/B/C")
    manifest = json.loads((run / "run.json").read_text())
    completed = json.loads((run / "completed.json").read_text())
    if manifest.get("schema") != RUN_SCHEMA or manifest.get("historical_experiment") is not False:
        raise ValueError("Not a student A/C run")
    if manifest.get("seed_protocol") not in (None, PROTOCOL):
        raise ValueError("Unknown teaching seed protocol")
    if manifest.get("seed_protocol") == PROTOCOL and (
            manifest.get("training_reset_seed") != manifest["model_seed"] or
            manifest.get("replay_seed") != replay_seed(manifest["model_seed"])):
        raise ValueError("Inconsistent teaching seed metadata")
    if completed.get("run_id") != manifest["run_id"]:
        raise ValueError("Mismatched completion marker")
    if completed.get("actual_steps") != manifest["requested_steps"]:
        raise ValueError("Completed step budget differs from the requested budget")
    config_path = run / "config.yaml"
    if sha256(config_path) != manifest["config_sha256"]:
        raise ValueError("Effective configuration changed after launch")
    checkpoint = (run / completed["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(run / "ckpt") or not (checkpoint / "done").is_file():
        raise ValueError("Invalid/incomplete final student checkpoint")
    if sha256(checkpoint / "agent.pkl") != completed["checkpoint_sha256"]:
        raise ValueError("Student checkpoint changed after completion")
    config = elements.Config.load(str(config_path))
    expected = training_config(config.logdir, manifest["reward_style"],
                               manifest["requested_steps"], manifest["model_seed"],
                               manifest.get("reward_weights"))
    if dict(config.flat) != dict(expected.flat):
        raise ValueError("Run configuration does not match the recorded A/C protocol")
    return run, manifest, completed, config, checkpoint
