"""Loader for the CWE sample corpus under data/cwe_samples/."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sample:
    id: str
    cwe: str
    vulnerable: bool
    description: str
    tags: list[str]
    code: str
    path: Path


def _read(code_path: Path) -> Sample:
    meta_path = code_path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"missing metadata for {code_path}: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    required = {"id", "cwe", "vulnerable", "description"}
    missing = required - set(meta)
    if missing:
        raise ValueError(f"{meta_path}: missing fields {missing}")
    return Sample(
        id=str(meta["id"]),
        cwe=str(meta["cwe"]),
        vulnerable=bool(meta["vulnerable"]),
        description=str(meta["description"]),
        tags=list(meta.get("tags", [])),
        code=code_path.read_text(encoding="utf-8"),
        path=code_path,
    )


def load_samples(root: str | Path = "data/cwe_samples") -> list[Sample]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    out: list[Sample] = []
    for p in sorted(root.glob("*.py")):
        try:
            out.append(_read(p))
        except (FileNotFoundError, ValueError) as e:
            log.warning("skipping %s: %s", p, e)
    if not out:
        raise RuntimeError(f"no valid samples under {root}")
    return out
