import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inference"))
import lab3_inference as lab

lab._prepare_process()


@pytest.fixture
def student_run(tmp_path):
    from optional_training.common import RUN_SCHEMA, training_config

    run = tmp_path / "run"
    checkpoint = run / "ckpt" / "final"
    checkpoint.mkdir(parents=True)
    weights = {"speed_weight": 1.0, "collision_penalty": 10.0}
    config = training_config("/original/location", "C", 1104, 17, weights)
    config.save(str(run / "config.yaml"))
    (checkpoint / "agent.pkl").write_bytes(b"checkpoint")
    (checkpoint / "done").touch()
    sources = (
        "dreamerv3/agent.py", "dreamerv3/rssm.py", "dreamerv3/configs.yaml",
        "embodied/jax/agent.py", "embodied/jax/transform.py", "embodied/jax/internal.py",
    )
    manifest = {
        "schema": RUN_SCHEMA, "historical_experiment": False, "run_id": "test",
        "reward_style": "C", "reward_weights": weights, "requested_steps": 1104,
        "model_seed": 17, "config_sha256": lab.sha256(run / "config.yaml"),
        "provenance": {"source_sha256": {name: lab.sha256(lab.RUNTIME / name)
                                          for name in sources}},
    }
    completed = {
        "run_id": "test", "actual_steps": 1104, "optimizer_updates": 4,
        "checkpoint": "ckpt/final", "checkpoint_sha256": lab.sha256(checkpoint / "agent.pkl"),
    }
    (run / "run.json").write_text(json.dumps(manifest))
    (run / "completed.json").write_text(json.dumps(completed))
    return run


def test_selects_custom_run_after_move(student_run):
    run = lab._select_run(student_run)
    assert run.checkpoint == student_run / "ckpt" / "final"
    assert run.updates == 4
    assert run.reward_weights == {"speed_weight": 1.0, "collision_penalty": 10.0}


@pytest.mark.parametrize("damage", ["checkpoint", "config", "incomplete", "source", "run_id"])
def test_rejects_invalid_run(student_run, damage):
    if damage == "checkpoint":
        (student_run / "ckpt/final/agent.pkl").write_bytes(b"changed")
    elif damage == "config":
        with (student_run / "config.yaml").open("a") as stream:
            stream.write("\n# changed\n")
    elif damage == "incomplete":
        (student_run / "ckpt/final/done").unlink()
    elif damage == "source":
        path = student_run / "run.json"
        data = json.loads(path.read_text())
        data["provenance"]["source_sha256"]["dreamerv3/agent.py"] = "wrong"
        path.write_text(json.dumps(data))
    else:
        path = student_run / "completed.json"
        data = json.loads(path.read_text())
        data["run_id"] = "wrong"
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        lab._select_run(student_run)


def test_default_keeps_pinned_c25k():
    run = lab._select_run(None)
    assert run.checkpoint == lab.CHECKPOINT
    assert run.checkpoint_hash == lab.CHECKPOINT_SHA256
    assert run.updates == 999
