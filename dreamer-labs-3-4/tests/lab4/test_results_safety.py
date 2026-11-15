from pathlib import Path

import pytest

from optional_training import common


@pytest.fixture
def project(monkeypatch, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(common, "REPO", root)
    return root


def test_results_created_exclusively(project):
    output = project / "results" / "driver-12345678"
    assert common.claim_output(output) == output
    with pytest.raises(ValueError, match="NEW"):
        common.claim_output(output)


def test_source_and_results_root_protected(project):
    for path in (project, project / "source" / "run", project / "results"):
        with pytest.raises(ValueError):
            common.validate_output(path)
    assert not (project / "results").exists()


def test_symlinked_results_rejected(project, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "results").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        common.claim_output(project / "results" / "run")
    assert not (outside / "run").exists()


def test_symlink_in_results_rejected(project, tmp_path):
    results = project / "results"
    results.mkdir()
    (results / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        common.validate_output(results / "link" / "run")


def test_traversal_and_evidence_paths_rejected(project):
    with pytest.raises(ValueError):
        common.validate_output(project / "results" / ".." / "source" / "run")
    run = project / "results" / "existing"
    run.mkdir(parents=True)
    (run / "run.json").write_text("{}")
    with pytest.raises(ValueError, match="existing run"):
        common.validate_output(run / "nested")
    with pytest.raises(ValueError, match="separate"):
        common.validate_output(project / "results" / "eval", (project / "results",))


def test_external_custom_output_still_supported(project, tmp_path):
    output = tmp_path / "external" / "run"
    assert common.claim_output(output) == Path(output)
