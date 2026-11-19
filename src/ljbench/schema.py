"""Unified item schema shared by every dataset, prompt and runner."""
from __future__ import annotations

import pandas as pd

TASK_FAITH = "faithfulness"
TASK_PAIR = "pairwise"

LABELS = {
    TASK_FAITH: ("SUPPORTED", "UNSUPPORTED"),
    TASK_PAIR: ("A", "B"),
}

COLUMNS = ["id", "task", "source", "subset", "context", "question",
           "answer", "answer_b", "human_label", "meta"]


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError listing every schema problem (empty list = valid)."""
    problems: list[str] = []
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    extra = [c for c in df.columns if c not in COLUMNS]
    if extra:
        problems.append(f"unexpected columns: {extra}")
    if not df["id"].is_unique:
        problems.append(f"duplicate ids: {df['id'][df['id'].duplicated()].head().tolist()}")
    bad_task = set(df["task"]) - set(LABELS)
    if bad_task:
        problems.append(f"unknown task values: {bad_task}")
    for task, labels in LABELS.items():
        sub = df[df["task"] == task]
        bad = set(sub["human_label"]) - set(labels)
        if bad:
            problems.append(f"{task}: invalid labels {bad}")
        if task == TASK_PAIR:
            if sub["answer_b"].isna().any() or (sub["answer_b"].astype(str).str.strip() == "").any():
                problems.append("pairwise items need a non-empty answer_b")
        else:
            if sub["answer_b"].notna().any():
                problems.append("faithfulness items must have answer_b = None")
            if (sub["context"].astype(str).str.strip() == "").any():
                problems.append("faithfulness items need a non-empty context")
    if (df["answer"].astype(str).str.strip() == "").any():
        problems.append("empty answer")
    if not df["meta"].map(lambda m: isinstance(m, dict)).all():
        problems.append("meta must be a dict")
    if problems:
        raise ValueError("; ".join(problems))
