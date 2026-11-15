import json
import sys
import types
from pathlib import Path

import pytest


def test_lab3_and_lab4_use_same_kernel_and_managed_guard():
    root = Path(__file__).resolve().parents[2]
    for name, cell in (("lab3_dreamerv3_decisions.ipynb", 4),
                       ("lab4_train_dreamerv3.ipynb", 2)):
        nb = json.loads((root / name).read_text())
        assert nb["metadata"]["kernelspec"]["name"] == "python3"
        source = "".join(nb["cells"][cell]["source"])
        assert "Path(sys.prefix).resolve().parent" in source
        assert "another runtime" in source
        assert "Wrong kernel" in source


@pytest.mark.parametrize(
    "notebook,cell,conflict",
    [("lab3_dreamerv3_decisions.ipynb", 4, "optional_training"),
     ("lab4_train_dreamerv3.ipynb", 2, "dreamerv3")],
)
def test_guard_rejects_cached_other_runtime(monkeypatch, notebook, cell, conflict):
    root = Path(__file__).resolve().parents[2]
    source = "".join(json.loads((root / notebook).read_text())["cells"][cell]["source"])
    guard = source[:source.index("sys.path.insert(0, str(runtime))")]
    monkeypatch.setattr(sys, "prefix", str(root / ".venv"))
    monkeypatch.setenv("VIRTUAL_ENV", str(root / ".venv"))
    module = types.ModuleType(conflict)
    other = root / "inference" / "runtime"
    module.__file__ = str(other / conflict / "__init__.py")
    monkeypatch.setitem(sys.modules, conflict, module)
    with pytest.raises(RuntimeError, match="another runtime"):
        exec(compile(guard, notebook, "exec"), {})


def test_guard_rejects_old_kernel(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    source = "".join(json.loads((root / "lab3_dreamerv3_decisions.ipynb").read_text())["cells"][4]["source"])
    guard = source[:source.index("runtime =")]
    monkeypatch.setattr(sys, "prefix", str(root / ".venv"))
    monkeypatch.setenv("VIRTUAL_ENV", str(root / "old-environment"))
    with pytest.raises(RuntimeError, match="Wrong kernel"):
        exec(compile(guard, "lab3-guard", "exec"), {})


@pytest.mark.parametrize("name", ["lab3_inference", "live_view"])
def test_guard_rejects_cached_other_inference_adapter(monkeypatch, name):
    root = Path(__file__).resolve().parents[2]
    source = "".join(json.loads((root / "lab3_dreamerv3_decisions.ipynb").read_text())["cells"][4]["source"])
    guard = source[:source.index("sys.path.insert(0, str(runtime))")]
    monkeypatch.setattr(sys, "prefix", str(root / ".venv"))
    monkeypatch.setenv("VIRTUAL_ENV", str(root / ".venv"))
    module = types.ModuleType(name)
    module.__file__ = str(root.parent / "old-package" / "inference" / f"{name}.py")
    monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(RuntimeError, match="another runtime"):
        exec(compile(guard, "lab3-guard", "exec"), {})
