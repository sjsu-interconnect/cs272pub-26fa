"""Compare two completed student evaluations with matching protocols."""
import argparse
import json
from pathlib import Path


def load(path, style):
    root = Path(path)
    summary = json.loads((root / "summary.json").read_text())
    evaluation = json.loads((root / "evaluation.json").read_text())
    if summary["reward_style"] != style or evaluation["reward_style"] != style:
        raise ValueError(f"{root} is not a {style}-style evaluation")
    if summary["student_run_id"] != evaluation["student_run_id"]:
        raise ValueError(f"{root} has mismatched run identifiers")
    if summary["seeds"] != evaluation["seeds"]:
        raise ValueError(f"{root} has mismatched evaluation seeds")
    return summary, evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="A evaluation directory")
    parser.add_argument("--c", required=True, help="C evaluation directory")
    args = parser.parse_args()
    a, ae = load(args.a, "A")
    c, ce = load(args.c, "C")
    if a["seeds"] != c["seeds"] or a["training_steps"] != c["training_steps"]:
        raise ValueError("A and C must use matching evaluation seeds and training steps")
    if ae["model_seed"] != ce["model_seed"]:
        raise ValueError("A and C must use the same training seed for this comparison")
    if a["training_steps"] != ae["training_steps"] or c["training_steps"] != ce["training_steps"]:
        raise ValueError("Training budget differs within an evaluation")
    print(f"Your runs: {a['training_steps']} steps, seed {ae['model_seed']}; {len(a['seeds'])} evaluation episodes per model")
    print(f"{'Metric':<30} {'A':>15} {'C':>15}")
    for label, key in (
        ("Mean speed (m/s)", "mean_speed_mps"),
        ("Mean displacement (m)", "displacement_m"),
        ("Mean duration (s)", "duration_s"),
        ("Mean lane changes", "lane_changes"),
    ):
        print(f"{label:<30} {a['metrics'][key]['mean']:>15.2f} {c['metrics'][key]['mean']:>15.2f}")
    print(f"{'Collisions / episodes':<30} {str(a['collision_count']) + '/' + str(a['episode_count']):>15} {str(c['collision_count']) + '/' + str(c['episode_count']):>15}")
    print("Different reward components and one training seed limit causal conclusions; faster is not necessarily safer.")


if __name__ == "__main__":
    main()
