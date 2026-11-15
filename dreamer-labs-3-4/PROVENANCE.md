# Provenance and limited distribution

This package is a local teaching candidate, not an approved public release.
Its `dreamerv3/`, `embodied/` and `optional_training/` code was copied from
the locally corrected working tree of Dreamerv3_RL_project, including the
verified JAX compatibility changes. The upstream source license is included
as `LICENSE` (copyright 2024 Danijar Hafner). Verify rights and course
authorization for all local modifications, teaching materials and reference
experiment data before distribution; no additional permission is assumed.
The source working tree was based on commit
`1bf6d51aca5b0fc9ef3a7017838d5b330ac13c77` with additional local corrections
and untracked teaching code; that commit alone is not the complete source.

The distribution copy of `optional_training/common.py` differs from the
verified workflow: source file hashes are still recorded, but Git commands
and Git metadata were replaced with a source-origin label because an extracted
ZIP has no repository. `compare.py` is a new presentation-only utility; it
does not train or evaluate. The notebook and `notebook_support.py` call the
existing trainer and evaluator in isolated processes. `embodied/run/train.py`
flushes the last logged metric interval on completion so a short smoke run
exposes its actual optimizer signals; this does not add optimizer updates.
For student reward experiments, the distribution's existing reward adapter,
configuration and per-transition checks now accept explicit speed and collision
weights; omitted weights retain the original A/C behavior. Evaluation reads
the recorded weights and also reports canonical A/C cross-scores. No model
architecture or training algorithm changes were made.

`reference/example_summary.json` contains selected aggregate metrics from
the prior A/C student workflow (seed 17; 24,996 runner steps and 999 updates
each; 20 evaluation episodes). These are example results, not a safety claim
or a single-coefficient ablation. No previous checkpoint, replay or video is
included; aggregate values were copied from the original A/C evaluation
summaries without recomputing or rounding. The original CPU requirements lock
pins the tested environment but does not contain hashes; dependencies are
downloaded separately and carry their own licenses. Other platforms are not
validated.

The combined Labs 3–4 distribution shares the runtime at the project root.
It retains Lab 4's final logging and customized-reward evaluation and Lab 3's
relocated-run configuration validation. Its `uv.lock` records resolved
distribution hashes. `optional_training/common.py` permits new outputs under
the project's `results/`, subject to path and overwrite checks. See
`inference/PROVENANCE.md` for the supplied C25K checkpoint and inference code.
