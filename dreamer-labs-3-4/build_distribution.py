"""Build a local teaching candidate; redistribution requires separate approval."""

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "PACKAGING_MANIFEST.json"
ARCHIVE = ROOT.parent / f"{ROOT.name}.zip"
EXCLUDED = {".venv", "results", "__pycache__", ".pytest_cache",
            ".ruff_cache", ".ipynb_checkpoints"}


def included(path):
    return (path.is_file() and path != MANIFEST
            and not any(part in EXCLUDED for part in path.relative_to(ROOT).parts)
            and not path.name.endswith((".pyc", ".pyo"))
            and path.name != ".DS_Store")


def main():
    paths = sorted(path for path in ROOT.rglob("*") if included(path))
    files = {}
    for path in paths:
        content = path.read_bytes()
        files[path.relative_to(ROOT).as_posix()] = {
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
        }
    MANIFEST.write_text(json.dumps({
        "schema": "dreamer-labs-3-4/v1",
        "note": "The manifest lists all other package files, not itself.",
        "files": files,
    }, indent=2) + "\n")
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted([*paths, MANIFEST]):
            archive.write(path, path.relative_to(ROOT.parent))
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC check failed")
        expected = {f"{ROOT.name}/{path.relative_to(ROOT).as_posix()}"
                    for path in [*paths, MANIFEST]}
        if set(archive.namelist()) != expected:
            raise RuntimeError("Archive file list mismatch")
        for name, info in files.items():
            content = archive.read(f"{ROOT.name}/{name}")
            if len(content) != info["bytes"] or hashlib.sha256(content).hexdigest() != info["sha256"]:
                raise RuntimeError(f"Archive hash mismatch: {name}")
    print(f"{ARCHIVE}: {len(files)} files, {ARCHIVE.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
