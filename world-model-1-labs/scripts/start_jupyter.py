"""Launch JupyterLab with this project's locked Python kernel first."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    project_kernels = Path(sys.prefix) / "share" / "jupyter"
    existing = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = os.pathsep.join(
        part for part in (str(project_kernels), existing) if part
    )
    os.chdir(project)
    os.execv(
        sys.executable,
        [sys.executable, "-m", "jupyterlab", "--ServerApp.root_dir=.", *sys.argv[1:]],
    )


if __name__ == "__main__":
    main()
