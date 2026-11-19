"""Append-only storage: a response cache and per-cell results files.

Both are JSONL so that a Colab disconnect loses at most the line being written, and both
work on a mounted Google Drive (no SQLite locking issues).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .openrouter import Completion


def _needs_newline(path: Path) -> bool:
    """True if a crash left a partial last line (no trailing newline)."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) != b"\n"


def _append(path: Path, rec: dict) -> None:
    prefix = "\n" if _needs_newline(path) else ""   # never glue a new row onto a half-written one
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(prefix)
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read(path: Path):
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue   # half-written last line after a crash


def cache_key(model_id: str, provider: str | None, messages: list[dict], params: dict) -> str:
    blob = json.dumps({"model": model_id, "provider": provider, "messages": messages, "params": params},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """(model, provider, exact messages, params) -> Completion. A rerun never pays twice."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem = {r["key"]: r["completion"] for r in _read(self.path)}

    def __len__(self):
        return len(self._mem)

    def get(self, key: str) -> Completion | None:
        d = self._mem.get(key)
        if d is None:
            return None
        c = Completion(**d)
        c.cached = True
        return c

    def put(self, key: str, c: Completion) -> None:
        if not c.text.strip():
            return   # never cache empty output; it may be transient
        d = c.to_dict()
        d["cached"] = False
        self._mem[key] = d
        _append(self.path, {"key": key, "completion": d})


class ResultStore:
    """results/<dataset>/<model_key>__<method>.jsonl, one row per judged (item, order)."""

    def __init__(self, root, dataset: str, model_key: str, method: str):
        self.path = Path(root) / dataset / f"{model_key}__{method}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done = {r["row_id"] for r in _read(self.path)}

    @property
    def done(self) -> set[str]:
        return set(self._done)

    def write(self, row: dict) -> None:
        if row["row_id"] in self._done:
            return
        _append(self.path, row)
        self._done.add(row["row_id"])

    def rows(self) -> list[dict]:
        return list(_read(self.path))
