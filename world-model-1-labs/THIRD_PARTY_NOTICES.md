# Third-party notices

Lab 1 code is distributed under the MIT license in `LICENSE`.
Dependencies are installed separately using `uv.lock`; their licenses and
copyright notices remain authoritative in the installed distributions.

The lab uses PyTorch, HighwayEnv, Gymnasium, pygame-ce, NumPy, Matplotlib,
scikit-learn, TensorBoard and Jupyter, among other locked dependencies.
The ZIPs do not redistribute those dependency installations.

Datasets and checkpoints in the Core package are course-generated artifacts
from the configured HighwayEnv simulation, not third-party pretrained models.
`BUNDLE_INFO.json` records checkpoint provenance and unchanged model tensors;
`BUNDLE_MANIFEST.json` records packaged file sizes and hashes.
