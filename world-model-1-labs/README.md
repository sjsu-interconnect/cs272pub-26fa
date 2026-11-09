# World Models Lecture 1: local source package

Begin with [START_HERE.md](START_HERE.md).

This lightweight package contains the current canonical Lab 1 and Lab 2
notebooks, `src/lab1`, locked Python dependencies, configurations, and the
scripts needed to generate data and train models. It contains no generated
datasets or trained checkpoints.

Required learning sequence:

1. generate one saved HighwayEnv dataset with `configs/lab.yaml`;
2. train and export a world model in `notebooks/1_learn_world_model.ipynb`;
3. run `notebooks/2_imagine_and_plan.ipynb`, which uses that exported model for
   imagination and MPC.

Dataset size and path are editable in the single configuration. The ZIP does
not contain generated data or a pretrained model.

Run these commands from the extracted top-level folder, in this order:

```bash
uv sync --locked --python 3.13
uv run python scripts/collect.py --config configs/lab.yaml
uv run python scripts/start_jupyter.py
```

The second command creates `datasets/highway.npz` with 20,000 samples. To use a
different number, edit `data.num_transitions` in `configs/lab.yaml` before
running that command. If the matching dataset already exists, the command
reports that it is ready and you can continue to Lab 1, followed by Lab 2.

More detail is in [START_HERE.md](START_HERE.md). The notebooks retain only
their tagged static teaching figures; other saved code outputs are cleared.
