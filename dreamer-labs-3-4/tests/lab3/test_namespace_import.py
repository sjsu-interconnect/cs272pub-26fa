import os
import subprocess
import sys
from pathlib import Path


def test_shared_runtime_imports_regular_package_without_namespace_merging():
    root = Path(__file__).resolve().parents[2]
    script = """
import sys
from pathlib import Path
root = Path(sys.argv[1])
runtime = root
sys.path.insert(0, str(root / 'inference' / 'runtime'))
sys.path.insert(0, str(root))
import optional_training
import optional_training.common
import optional_training.rewards
assert list(map(Path, optional_training.__path__)) == [runtime / 'optional_training']
for module in (optional_training, optional_training.common, optional_training.rewards):
    assert Path(module.__file__).resolve().is_relative_to(runtime / 'optional_training')
"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run([sys.executable, "-c", script, str(root)], check=True, env=env)
