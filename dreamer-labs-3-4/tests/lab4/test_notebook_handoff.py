import ast
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def notebook():
    return json.loads((ROOT / "lab4_train_dreamerv3.ipynb").read_text())


def source(notebook, cell):
    return "".join(notebook["cells"][cell]["source"])


def configure(monkeypatch, notebook, output, **edits):
    import optional_training.common as common
    import optional_training.notebook_support as support

    monkeypatch.chdir(ROOT)
    calls = []

    def run_process(script, args, run_dir=None):
        calls.append((script, args, run_dir))

    def load_student_run(run):
        args = calls[-1][1]
        completed = {
            "actual_steps": args[args.index("--steps") + 1],
            "optimizer_updates": 4,
        }
        return run.resolve(), {}, completed, None, run / "ckpt"

    monkeypatch.setattr(support, "run_process", run_process)
    monkeypatch.setattr(common, "load_student_run", load_student_run)
    namespace = {"Path": Path, "sys": sys, "ROOT": ROOT}
    exec(compile(source(notebook, 4), "lab4-settings", "exec"), namespace)
    namespace.update(OUTPUT_DIR=str(output), **edits)
    exec(compile(source(notebook, 6), "lab4-train", "exec"), namespace)
    return namespace, calls


def test_one_experiment_settings_cell(notebook):
    tree = ast.parse(source(notebook, 4))
    names = {node.targets[0].id for node in tree.body if isinstance(node, ast.Assign)}
    assert names == {
        "PRESET", "SPEED_WEIGHT", "COLLISION_PENALTY", "TRAINING_STEPS",
        "SEED", "SMOKE_TEST", "EXPERIMENT_NAME", "OUTPUT_DIR",
    }
    assert notebook["cells"][3]["cell_type"] == "markdown"
    assert "Examples — Try These" in source(notebook, 3)
    for cell in notebook["cells"][5:]:
        if cell["cell_type"] != "code":
            continue
        other = ast.parse("".join(cell["source"]))
        assert not any(
            isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            and node.id in names for node in ast.walk(other)
        )


def test_default_settings_select_original_a(monkeypatch, notebook, tmp_path):
    ns, calls = configure(monkeypatch, notebook, tmp_path)
    assert ns["PRESET"] == "A" and ns["SMOKE_TEST"] is True
    assert ns["SPEED_WEIGHT"] is None and ns["COLLISION_PENALTY"] is None
    assert ns["weights"] == {"speed_weight": 0.4, "collision_penalty": 1.0}
    args = calls[0][1]
    assert args[args.index("--reward") + 1] == "A"
    assert args[args.index("--steps") + 1] == 1104
    assert args[args.index("--speed-weight") + 1] == 0.4
    assert args[args.index("--collision-penalty") + 1] == 1.0


@pytest.mark.parametrize(
    "preset,speed,penalty,expected_speed,expected_penalty",
    [
        ("A", None, None, 0.4, 1.0),
        ("C", None, None, 1.0, 40.0),
        ("A", 0.8, None, 0.8, 1.0),
        ("C", None, 10.0, 1.0, 10.0),
        ("A", None, 3.0, 0.4, 3.0),
        ("C", 2.0, None, 2.0, 40.0),
    ],
)
def test_preset_defaults_and_independent_overrides(
        monkeypatch, notebook, tmp_path, capsys, preset, speed, penalty,
        expected_speed, expected_penalty):
    from optional_training.common import training_config
    from optional_training.rewards import settings

    ns, calls = configure(
        monkeypatch, notebook, tmp_path, PRESET=preset,
        SPEED_WEIGHT=speed, COLLISION_PENALTY=penalty,
    )
    args = calls[0][1]
    weights = {"speed_weight": expected_speed, "collision_penalty": expected_penalty}
    assert ns["weights"] == weights
    assert args[args.index("--reward") + 1] == preset
    assert args[args.index("--speed-weight") + 1] == expected_speed
    assert args[args.index("--collision-penalty") + 1] == expected_penalty
    reward = settings(preset, weights)
    config = training_config(ns["run_dir"], preset, 1104, 17, weights)
    assert config.env.highway.reward_mode == reward["reward_mode"]
    assert config.env.highway.config.high_speed_reward == expected_speed
    assert config.env.highway.config.collision_reward == -expected_penalty
    output = capsys.readouterr().out
    assert f"Speed weight: {expected_speed} ({'preset default' if speed is None else 'custom'})" in output
    assert f"Collision penalty: {expected_penalty} ({'preset default' if penalty is None else 'custom'})" in output
    assert "Smoke test: ON | Environment steps: 1,104" in output
    assert ("non_collision" if preset == "A" else "speed_bonus −") in output


def test_switching_preset_recalculates_defaults(monkeypatch, notebook, tmp_path):
    ns, calls = configure(monkeypatch, notebook, tmp_path, PRESET="A")
    ns["PRESET"] = "C"
    exec(compile(source(notebook, 6), "lab4-train-again", "exec"), ns)
    args = calls[-1][1]
    assert args[args.index("--reward") + 1] == "C"
    assert args[args.index("--speed-weight") + 1] == 1.0
    assert args[args.index("--collision-penalty") + 1] == 40.0
    assert ns["SPEED_WEIGHT"] is None and ns["COLLISION_PENALTY"] is None


@pytest.mark.parametrize("steps,smoke,actual", [
    (24996, False, 24996), (1200, False, 1200), (24996, True, 1104),
    (0, True, 1104),
])
def test_budget_and_reward_reach_one_trainer(monkeypatch, notebook, tmp_path, steps, smoke, actual):
    ns, calls = configure(
        monkeypatch, notebook, tmp_path / "my results",
        EXPERIMENT_NAME="traffic-test", TRAINING_STEPS=steps, SMOKE_TEST=smoke,
        PRESET="A", SPEED_WEIGHT=0.8, COLLISION_PENALTY=3.0, SEED=29,
    )
    assert len(calls) == 1
    script, args, run_dir = calls[0]
    assert script.name == "train.py" and run_dir == ns["run_dir"]
    assert args[args.index("--steps") + 1] == actual
    assert args[args.index("--seed") + 1] == 29
    assert args[args.index("--reward") + 1] == "A"
    assert args[args.index("--speed-weight") + 1] == 0.8
    assert args[args.index("--collision-penalty") + 1] == 3.0
    assert ns["run_id"].startswith("traffic-test-")
    assert ns["eval_dir"] != ns["run_dir"]


@pytest.mark.parametrize("steps", [0, 12, 1103, 1105, 25008, 1.5, True, "1200"])
def test_invalid_training_budget_rejected(monkeypatch, notebook, tmp_path, steps):
    with pytest.raises(ValueError, match="TRAINING_STEPS"):
        configure(monkeypatch, notebook, tmp_path, TRAINING_STEPS=steps, SMOKE_TEST=False)


def test_unique_names_and_invalid_experiment_name(monkeypatch, notebook, tmp_path):
    first, _ = configure(monkeypatch, notebook, tmp_path)
    second, _ = configure(monkeypatch, notebook, tmp_path)
    assert first["run_dir"] != second["run_dir"]
    with pytest.raises(ValueError, match="EXPERIMENT_NAME"):
        configure(monkeypatch, notebook, tmp_path, EXPERIMENT_NAME="../bad")


def test_smoke_suppresses_curves(notebook, capsys):
    namespace = {"completed": {"actual_steps": 1104, "optimizer_updates": 4},
                 "SMOKE_TEST": True}
    exec(compile(source(notebook, 8), "lab4-curves", "exec"), namespace)
    assert "curves are suppressed" in capsys.readouterr().out


def test_sparse_learning_logs_are_points(notebook, capsys, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = {
        "completed": {"actual_steps": 24996, "optimizer_updates": 999},
        "SMOKE_TEST": False,
        "run_dir": Path("/unused"),
        "read_metrics": lambda _: [
            {"step": 1000, "train/loss/obs": 3.0},
            {"step": 2000, "train/loss/obs": 2.0},
        ],
    }
    exec(compile(source(notebook, 8), "lab4-curves", "exec"), namespace)
    assert not namespace["axes"][0].lines
    assert len(namespace["axes"][0].collections) == 1
    assert "fewer than five logged points" in capsys.readouterr().out
    plt.close(namespace["fig"])


def test_generated_line_quotes_actual_run(notebook, tmp_path, capsys):
    path = tmp_path / 'student "quoted" run'
    namespace = {
        "run_dir": path,
        "load_student_run": lambda run: (run.resolve(), None, {}, None, None),
        "json": json, "Path": Path, "ROOT": ROOT, "SMOKE_TEST": False,
        "summary": {
            "episode_count": 3, "collision_count": 2,
            "metrics": {"own_reward_return": {"mean": 2.5},
                        "mean_speed_mps": {"mean": 21.0},
                        "displacement_m": {"mean": 120.0}},
        },
    }
    namespace["load_student_run"] = lambda run: (
        run.resolve(), None, {"actual_steps": 1104, "optimizer_updates": 4}, None, None)
    exec(compile(source(notebook, 12), "lab4-handoff", "exec"), namespace)
    lines = capsys.readouterr().out.splitlines()
    assignment = next(line for line in lines if line.startswith("CHECKPOINT = "))
    assert ast.literal_eval(assignment.split(" = ", 1)[1]) == os.fspath(path.resolve())
    assert any("collisions: 2/3" in line for line in lines)
    assert any("Mean episode return (this reward): 2.50" in line for line in lines)
    assert any("2 ended in collisions" in line for line in lines)


def test_summary_uses_home_relative_path_and_warns_on_smoke(notebook, capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    path = tmp_path / "lab4-results" / "driver-abcd1234"
    namespace = {
        "run_dir": path, "Path": Path, "ROOT": ROOT, "json": json, "SMOKE_TEST": True,
        "load_student_run": lambda run: (
            run, None, {"actual_steps": 1104, "optimizer_updates": 4}, None, None),
        "summary": {
            "episode_count": 3, "collision_count": 3,
            "metrics": {"own_reward_return": {"mean": -5.0},
                        "mean_speed_mps": {"mean": 25.0},
                        "displacement_m": {"mean": 200.0}},
        },
    }
    exec(compile(source(notebook, 12), "lab4-handoff", "exec"), namespace)
    output = capsys.readouterr().out
    assert 'CHECKPOINT = "~/lab4-results/driver-abcd1234"' in output
    assert "does not establish" in output and "comparison between reward presets" in output


def test_summary_uses_project_relative_results_path(notebook, capsys):
    path = ROOT / "results" / "driver-abcd1234"
    namespace = {
        "run_dir": path, "Path": Path, "ROOT": ROOT, "json": json, "SMOKE_TEST": True,
        "load_student_run": lambda run: (
            run, None, {"actual_steps": 1104, "optimizer_updates": 4}, None, None),
        "summary": {
            "episode_count": 3, "collision_count": 3,
            "metrics": {"own_reward_return": {"mean": -5.0},
                        "mean_speed_mps": {"mean": 25.0},
                        "displacement_m": {"mean": 200.0}},
        },
    }
    exec(compile(source(notebook, 12), "lab4-handoff", "exec"), namespace)
    assert 'CHECKPOINT = "results/driver-abcd1234"' in capsys.readouterr().out
