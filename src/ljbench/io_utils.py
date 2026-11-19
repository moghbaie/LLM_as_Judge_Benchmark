"""JSONL IO and hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .schema import COLUMNS


def _clean(v):
    if v is None or (isinstance(v, float) and v != v):  # None / NaN -> null
        return None
    return v


def write_jsonl(df: pd.DataFrame, path) -> None:
    """One JSON object per line, sorted keys, LF line endings -> byte-identical across OSes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in df[COLUMNS].to_dict(orient="records"):
            f.write(json.dumps({k: _clean(v) for k, v in rec.items()}, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows, columns=COLUMNS)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
