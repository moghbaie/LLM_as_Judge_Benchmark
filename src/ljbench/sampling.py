"""Length filtering and seeded, stratified, label-balanced sampling."""
from __future__ import annotations

import math

import pandas as pd

from . import config

TEXT_COLS = ["context", "question", "answer", "answer_b"]


def estimate_tokens(text, chars_per_token: float = config.CHARS_PER_TOKEN) -> int:
    if not isinstance(text, str) or not text:
        return 0
    return math.ceil(len(text) / chars_per_token)


def input_tokens(df: pd.DataFrame) -> pd.Series:
    """Estimated tokens of everything the judge must read (excluding the prompt template)."""
    return sum(df[c].map(estimate_tokens) for c in TEXT_COLS)


def filter_by_length(df: pd.DataFrame, max_tokens: int = config.MAX_INPUT_TOKENS):
    """Return (kept, dropped)."""
    n = input_tokens(df)
    return df[n <= max_tokens].reset_index(drop=True), df[n > max_tokens].reset_index(drop=True)


def stratified_balanced_sample(df: pd.DataFrame, n: int, strata_col: str | None,
                               label_col: str = "human_label", seed: int = config.SEED) -> pd.DataFrame:
    """Draw n rows split equally across every (stratum x label) cell.

    Deterministic: input row order does not matter (rows are sorted by id first).
    Raises if n does not divide evenly or a cell lacks enough rows.
    """
    df = df.sort_values("id").reset_index(drop=True)
    keys = [label_col] if strata_col is None else [strata_col, label_col]
    groups = {k: g for k, g in df.groupby(keys, sort=True)}
    if n % len(groups):
        raise ValueError(f"n={n} is not divisible by {len(groups)} cells {list(groups)}")
    per_cell = n // len(groups)
    parts = []
    for key, g in groups.items():
        if len(g) < per_cell:
            raise ValueError(f"cell {key} has {len(g)} rows, need {per_cell}")
        parts.append(g.sample(n=per_cell, random_state=seed))
    # final seeded shuffle so any prefix of the file is still roughly balanced
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def split_off(df: pd.DataFrame, n: int, seed: int = config.SEED):
    """Seeded holdout: return (held_out[n], rest)."""
    df = df.sort_values("id").reset_index(drop=True)
    held = df.sample(n=n, random_state=seed)
    rest = df.drop(held.index)
    return held.reset_index(drop=True), rest.reset_index(drop=True)
