# Build the two labs from source

The lightweight Source Package contains code, one editable configuration,
locked dependencies, and the current Lab 1 and Lab 2 notebooks. It intentionally
excludes generated data and trained checkpoints.

The required path is:

```text
configs/lab.yaml
        ↓
generate datasets/highway.npz
        ↓
Lab 1 trains and exports a world model
        ↓
Lab 2 uses that model for imagination and MPC
```

## 1. Install the locked environment

Extract the whole ZIP and open a terminal in its top-level folder, where
`pyproject.toml`, `uv.lock`, `src/`, `notebooks/`, and `configs/` are visible.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed,
then run:

```bash
uv sync --locked --python 3.13
```

JupyterLab and ipykernel are project dependencies. Do not install or launch a
different global Jupyter environment.

For a tiny installation and pipeline check, run:

```bash
uv run python scripts/try_from_source.py
```

This creates `outputs/source_check/`, collects 128 transitions, trains one
epoch, and runs short planner and random-policy episodes. It checks the pipeline
only and does not create the dataset used by the notebooks.

## 2. Generate the notebook dataset

The default uses **20,000 samples** (saved transitions). If you want a different
number, edit `data.num_transitions` in `configs/lab.yaml`. Then run one command
from the extracted top-level folder:

```bash
uv run python scripts/collect.py --config configs/lab.yaml
```

The default configuration creates:

```text
datasets/highway.npz    20,000 saved transitions used by both labs
```

If a matching dataset already exists, the command reports that it is ready and
you can continue to Lab 1. It never silently overwrites an existing file. The
relevant setting is:

```yaml
data:
  num_transitions: 20000
```

Both labs automatically use the resulting `datasets/highway.npz`.

To preserve that dataset and later collect 30,000 new transitions, set
`num_transitions: 30000` and `dataset_path: datasets/highway_30000.npz`. To
replace it instead, move or delete the old dataset before collecting again.

## 3. Train and export the Lab 1 model

Launch JupyterLab through the locked project environment:

```bash
uv run python scripts/start_jupyter.py
```

Open `notebooks/1_learn_world_model.ipynb`. The visible setting starts as:

```python
EPOCHS = 20
```

Any positive integer is accepted. Select **Kernel → Restart Kernel and Run All
Cells**. The notebook trains from random initialization, keeps the
best-validation checkpoint, and exports a model-only handoff under
`outputs/my_build/`. The final cell prints the exact path.

The handoff contains model weights and compatibility metadata, not optimizer
state.

## 4. Use the Lab 1 model in Lab 2

Open `notebooks/2_imagine_and_plan.ipynb` in the same JupyterLab session and
select **Kernel → Restart Kernel and Run All Cells**.

Lab 2 finds the most recently exported Lab 1 model automatically. It uses that
model for one-step prediction, recursive imagination, candidate scoring,
closed-loop MPC, replanning, animation, and fixed-seed evaluation. Both labs
read `datasets/highway.npz` and `configs/lab.yaml`.

## 5. What the package includes

Included:

- current Lab 1 and Lab 2 notebooks;
- every helper imported by those notebooks;
- `src/lab1` and the collection, training, and planning scripts;
- one editable configuration, `configs/lab.yaml`;
- Python 3.13-compatible locked dependencies;
- this guide and the pre-lab primer.

Excluded:

- generated `.npz` datasets;
- `.pt` checkpoints and optimizer state;
- cached environments and notebook autosaves;
- Colab notebooks and the Colab data bundle;
- generated experiment artifacts and large media.

Saved grayscale observations are flipped vertically only when displayed.
Stored arrays, model inputs, losses, predictions, and planner computations are
unchanged.

## Troubleshooting

- **`uv` not found:** install uv, reopen the terminal, and return to the
  extracted top-level folder.
- **`No module named lab1`:** run `uv sync` and use
  `uv run python scripts/start_jupyter.py` from the folder containing
  `pyproject.toml`. The launcher gives the locked project kernel priority over
  unrelated kernels already installed on the computer.
- **Missing dataset:** run the single `configs/lab.yaml` collection command.
- **Missing Lab 1 model:** complete Lab 1 through its final export cell, then
  rerun Lab 2.
- **Black observations:** do not set `SDL_VIDEODRIVER=dummy`. On headless Linux,
  use a real virtual framebuffer such as `xvfb-run -a`.

Read [PRELAB.md](PRELAB.md), then run Lab 1 followed by Lab 2.
