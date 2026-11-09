"""Small helpers that make the teaching notebooks fail clearly.

The experiment code stays in the regular :mod:`lab1` package.  These helpers
only handle notebook ergonomics: checking prerequisites before a long cell is
started and explaining how to create anything that is missing.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

BUILD_OR_CORE = (
    "To create them, follow docs/FROM_SOURCE.md (Source package).\n"
    "Or use the recommended Core Student Package, which already includes "
    "these pretrained artifacts."
)


def require_artifacts(
    artifacts: Mapping[str | Path, str],
    *,
    heading: str = "This notebook is missing required artifacts.",
    remedy: str | None = None,
) -> None:
    """Raise one actionable error listing every missing file and its remedy.

    Parameters
    ----------
    artifacts:
        Mapping from a required path to a short description of that file.
    heading:
        First line of the error message.
    remedy:
        Optional closing lines explaining how to obtain the missing files.
    """
    missing = [
        (Path(path), hint)
        for path, hint in artifacts.items()
        if not Path(path).exists()
    ]
    if not missing:
        return

    lines = [heading]
    for path, hint in missing:
        lines.append(f"- {path}: {hint}")
    if remedy:
        lines.append(remedy)
    raise FileNotFoundError("\n".join(lines))
