# Getting Started — Dreamer Labs 3 and 4

Lab 3 lets you watch a trained Actor drive. Lab 4 trains one policy that you
can then watch in Lab 3.

**Supported:** macOS on Apple silicon, Python 3.11.12 and CPU. Intel Macs
are not supported by the pinned JAX version; Linux and Windows are unverified.
Installation requires internet access.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Extract `dreamer-labs-3-4.zip` and open a terminal in the extracted
   `dreamer-labs-3-4` folder.
3. Run:

   ```bash
   uv sync --locked
   uv run --locked jupyter lab
   ```

4. Open `lab3_dreamerv3_decisions.ipynb`, then
   `lab4_train_dreamerv3.ipynb`. If prompted, select **Python 3 (ipykernel)**
   for each notebook. Run each notebook with **Kernel → Restart Kernel and
   Run All Cells**.

Lab 4 saves each run under this folder's `results/`. Copy the `CHECKPOINT`
assignment printed at the end of Lab 4 into the settings cell in Lab 3, then
restart and run Lab 3 again. Only load trusted checkpoints: pickle files can
execute code. The Lab 4 smoke test checks the pipeline, not policy quality.

If a notebook reports a wrong kernel, close JupyterLab and relaunch it using
the second command above from this folder, then select **Python 3 (ipykernel)**.
If it reports another runtime is already loaded, restart that notebook's
kernel before running it. Each notebook normally has its own kernel session.
