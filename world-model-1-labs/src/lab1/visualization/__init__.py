"""Visualizations for the lab.

Primary teaching figures: :func:`plot_imagination_grid` and
:func:`plot_error_vs_horizon`. The remaining figures are diagnostics or
optional-extension material.
"""

from lab1.visualization.animations import (
    ACTION_NAMES,
    animate_episode,
    animate_policy_run,
)
from lab1.visualization.learning import (
    plot_experience_to_world_model,
    plot_latent_before_after,
    plot_learning_diagram,
    plot_learning_to_imagination,
    plot_reward_before_after,
    plot_saved_transition,
    prediction_metrics,
)
from lab1.visualization.plots import (
    one_step_latent_error,
    plot_candidate_plans,
    plot_cumulative_reward_prediction,
    plot_error_vs_disagreement,
    plot_error_vs_horizon,
    plot_imagination_budget,
    plot_imagination_grid,
    plot_lambda_sweep,
    plot_latent_pca,
    plot_losses,
    plot_overoptimism_vs_budget,
    plot_planner_comparison,
    plot_real_vs_imagined,
    plot_reconstruction,
    plot_return_vs_budget,
    plot_reward_overoptimism,
    plot_reward_prediction,
    plot_training_budget_comparison,
)
from lab1.visualization.story import (
    animate_mpc_run,
    mpc_frame_records,
    planner_probe_table,
    plot_candidate_decision,
    plot_horizon_error,
    plot_lab_synthesis,
    plot_latent_error_paths,
    plot_latent_rollout,
    plot_mpc_cycle,
    plot_predicted_vs_actual,
    plot_replanning_storyboard,
    plot_ten_seed_evidence,
    plot_visual_legend,
    plot_world_model_architecture,
)

__all__ = [
    "animate_mpc_run",
    "mpc_frame_records",
    "plot_horizon_error",
    "plot_lab_synthesis",
    "plot_mpc_cycle",
    "plot_replanning_storyboard",
    "plot_ten_seed_evidence",
    "plot_visual_legend",
    "plot_world_model_architecture",
    "plot_latent_error_paths",
    "plot_predicted_vs_actual",
    "planner_probe_table",
    # Lab 1: how the world model learns
    "plot_experience_to_world_model",
    "plot_learning_to_imagination",
    "plot_saved_transition",
    "plot_learning_diagram",
    "prediction_metrics",
    "plot_latent_before_after",
    "plot_reward_before_after",
    # inline animations
    "animate_episode",
    "animate_policy_run",
    "ACTION_NAMES",
    # primary teaching figures
    "plot_imagination_grid",
    "plot_latent_rollout",
    "plot_error_vs_horizon",
    "one_step_latent_error",
    # diagnostics
    "plot_losses",
    "plot_training_budget_comparison",
    "plot_reconstruction",
    "plot_latent_pca",
    "plot_real_vs_imagined",
    # optional extension
    "plot_reward_prediction",
    "plot_cumulative_reward_prediction",
    "plot_candidate_decision",
    "plot_candidate_plans",
    "plot_planner_comparison",
    "plot_imagination_budget",
    # Lab 2: robust planning
    "plot_return_vs_budget",
    "plot_reward_overoptimism",
    "plot_overoptimism_vs_budget",
    "plot_error_vs_disagreement",
    "plot_lambda_sweep",
]
