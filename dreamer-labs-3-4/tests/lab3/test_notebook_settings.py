import ast
import json
from pathlib import Path


def test_student_settings_and_live_drive_connection():
    path = Path(__file__).resolve().parents[2] / "lab3_dreamerv3_decisions.ipynb"
    notebook = json.loads(path.read_text())
    cells = notebook["cells"]
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert notebook["metadata"]["kernelspec"]["display_name"] == "Python 3 (ipykernel)"
    assert "SETTINGS — EDIT HERE" in "".join(cells[1]["source"])
    assert cells[1]["cell_type"] == "markdown"
    assert "Examples — Try These" in "".join(cells[1]["source"])
    settings = ast.parse("".join(cells[2]["source"]))
    assert len(settings.body) == 2
    assignments = {node.targets[0].id: ast.literal_eval(node.value)
                   for node in settings.body if isinstance(node, ast.Assign)}
    assert assignments == {"CHECKPOINT": None, "SEED": 1000}
    for cell in cells[3:]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        assert not any(
            isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            and node.id in assignments for node in ast.walk(tree)
        )
    setup = ast.parse("".join(cells[4]["source"]))
    call = next(node for node in ast.walk(setup)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "LiveDrive")
    keywords = {key.arg: key.value for key in call.keywords}
    assert isinstance(keywords["seed"], ast.Name) and keywords["seed"].id == "SEED"
    assert isinstance(keywords["checkpoint"], ast.Name) and keywords["checkpoint"].id == "run"
