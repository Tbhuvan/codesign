import json
from pathlib import Path

import pytest

from codesign.dataset import load_samples


def test_load_shipped_samples():
    samples = load_samples("data/cwe_samples")
    assert len(samples) >= 10
    cwes = {s.cwe for s in samples}
    assert "CWE-78" in cwes
    assert "CWE-89" in cwes


def test_unique_ids():
    samples = load_samples("data/cwe_samples")
    ids = [s.id for s in samples]
    assert len(ids) == len(set(ids))


def test_includes_negative_control():
    samples = load_samples("data/cwe_samples")
    safe = [s for s in samples if not s.vulnerable]
    assert safe


def test_missing_root(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_samples(tmp_path / "nope")


def test_orphan_file_raises(tmp_path: Path):
    (tmp_path / "orphan.py").write_text("x = 1\n")
    with pytest.raises(RuntimeError):
        load_samples(tmp_path)


def test_invalid_metadata_skipped(tmp_path: Path):
    (tmp_path / "good.py").write_text("x = 1\n")
    (tmp_path / "good.json").write_text(json.dumps({
        "id": "good", "cwe": "CWE-78", "vulnerable": True, "description": "ok"
    }))
    (tmp_path / "bad.py").write_text("y = 2\n")
    (tmp_path / "bad.json").write_text(json.dumps({"id": "bad"}))
    samples = load_samples(tmp_path)
    assert {s.id for s in samples} == {"good"}
