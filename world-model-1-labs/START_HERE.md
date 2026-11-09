# Start here: build the two labs from source

This lightweight package contains the current source and the two canonical
notebooks. It deliberately contains **no generated datasets and no trained
checkpoints**.

The local source workflow is:

```text
Generate saved HighwayEnv experience
        ↓
Lab 1: train and export a world model
        ↓
Lab 2: use that model for imagination and MPC
```

Lab 1 and Lab 2 are both required. The package is an optional local alternative
to the public Colab notebooks linked from the lecture PDF.

## 1. Prepare the environment

Extract the entire ZIP. Open a terminal in the extracted folder that contains
`pyproject.toml`, `uv.lock`, `src/`, `notebooks/`, and `configs/`.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed,
then run from that folder:

```bash
uv sync --locked --python 3.13
```

No separate global Python or JupyterLab installation is required.

## 2. Generate the saved experience

The ZIP excludes generated data. The default uses **20,000 samples** (saved
transitions). If you want a different number, edit `data.num_transitions` in
`configs/lab.yaml`. Then run:

```bash
uv run python scripts/collect.py --config configs/lab.yaml
```

This creates `datasets/highway.npz`, which both labs use. Existing files are
never overwritten silently. If a matching 20,000-transition dataset already
exists, the command reports that it is ready; continue to Lab 1.

See [FROM_SOURCE](docs/FROM_SOURCE.md) for expected files, alternative model
paths, and a tiny end-to-end installation check.

## 3. Run Lab 1 and train your model

Launch JupyterLab:

```bash
uv run python scripts/start_jupyter.py
```

Open `notebooks/1_learn_world_model.ipynb`. The visible `EPOCHS = 20` setting is
editable; any positive integer is accepted. Select **Kernel → Restart Kernel
and Run All Cells**.

Lab 1 trains from random initialization and exports a compatible model-only
checkpoint under `outputs/my_build/`. Keep the exact path printed by its final
cell.

## 4. Run Lab 2 with your Lab 1 model

Open `notebooks/2_imagine_and_plan.ipynb` in the same JupyterLab session, then
select **Kernel → Restart Kernel and Run All Cells**. Local Jupyter finds
the most recently exported Lab 1 checkpoint automatically. Lab 2 uses that one
model throughout prediction, imagination, MPC, replanning, animation, and the
fixed-seed evaluation.

## 5. Change the experiment

- More or fewer samples: edit `data.num_transitions` in `configs/lab.yaml`.
- More or fewer training epochs: edit `EPOCHS = 20` in Lab 1.

For example, to keep the 20,000-sample dataset and create a separate
30,000-sample dataset, edit:

```yaml
data:
  num_transitions: 30000
  dataset_path: datasets/highway_30000.npz
```

Both labs automatically read the new path from the same configuration. To
replace the old dataset instead, move or delete `datasets/highway.npz` before
running the collection command again.

## Troubleshooting

- **`uv` not found:** install uv, close the terminal, reopen it, and return to
  the extracted package folder.
- **`No module named lab1`:** launch Jupyter only with
  `uv run python scripts/start_jupyter.py` from the folder containing
  `pyproject.toml`. The launcher makes this package's locked Python kernel take
  priority over old Jupyter kernels already installed on the computer.
- **Missing dataset:** run the single `configs/lab.yaml` collection command.
- **Missing personal checkpoint:** finish Lab 1 through its final export cell,
  then rerun Lab 2.
- **Blank rendering:** do not set `SDL_VIDEODRIVER=dummy`; on headless Linux,
  use a real virtual framebuffer such as `xvfb-run -a`.

Read the [pre-lab primer](docs/PRELAB.md) for notation and the Lab 1 to Lab 2
model handoff.
