import ast
import io
import runpy
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_compatibility_requirements_include_shared_lock():
    requirements = ROOT / "inference" / "requirements.txt"
    includes = [
        (requirements.parent / line[3:]).resolve()
        for line in requirements.read_text().splitlines() if line.startswith("-r ")
    ]
    assert includes == [ROOT / "optional_training" / "requirements-cpu.lock"]
    assert all(path.is_file() for path in includes)


def test_distributed_training_local_imports_after_extraction(tmp_path):
    included = runpy.run_path(str(ROOT / "build_distribution.py"))["included"]
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        for path in ROOT.rglob("*"):
            if included(path):
                archive.write(path, path.relative_to(ROOT))
    extracted = tmp_path / "extracted distribution"
    with zipfile.ZipFile(content) as archive:
        members = set(archive.namelist())
        tree = ast.parse(archive.read("optional_training/train.py"))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module)
        local = sorted(name for name in modules
                       if (ROOT / name.split(".")[0]).is_dir())
        for name in local:
            path = name.replace(".", "/")
            assert path + ".py" in members or path + "/__init__.py" in members, name
        archive.extractall(extracted)
    script = """
import importlib
from pathlib import Path
import sys
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
for name in sys.argv[2:]:
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(root), name
from optional_training.smoke import environment_checks, c_reward_checks
assert callable(environment_checks) and callable(c_reward_checks)
"""
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(extracted), *local],
        check=True, cwd=tmp_path,
    )
