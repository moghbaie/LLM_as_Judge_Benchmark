"""Raw dataset -> unified schema.

`*_to_items` functions are pure (DataFrame in, DataFrame out) so they can be unit
tested offline. `fetch_*` functions do the network download at a pinned revision.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config
from .schema import COLUMNS, TASK_FAITH, TASK_PAIR

# --------------------------------------------------------------------------- RAGTruth
RAGTRUTH_KEEP_QUALITY = {"good"}  # drops incorrect_refusal / truncated


def _spans(s) -> list:
    return json.loads(s) if isinstance(s, str) and s.strip() else []


def ragtruth_to_items(raw: pd.DataFrame, split: str) -> pd.DataFrame:
    """wandb/RAGTruth-processed rows -> unified items.

    Label: UNSUPPORTED if human annotators marked at least one hallucinated span.
    """
    df = raw[raw["quality"].isin(RAGTRUTH_KEEP_QUALITY)].reset_index(drop=True)

    def meta(r) -> dict:
        sp = _spans(r["hallucination_labels"])
        return {
            "split": split,
            "generator": r["model"],
            "quality": r["quality"],
            "n_spans": len(sp),
            "span_types": sorted({x.get("label_type", "") for x in sp}),
        }

    out = pd.DataFrame({
        "id": "ragtruth-" + df["id"].astype(str),
        "task": TASK_FAITH,
        "source": "ragtruth",
        "subset": df["task_type"].astype(str),
        "context": df["context"].astype(str),
        "question": df["query"].astype(str),
        "answer": df["output"].astype(str),
        "answer_b": None,
        "human_label": df["hallucination_labels"].map(
            lambda s: "UNSUPPORTED" if _spans(s) else "SUPPORTED"),
        "meta": [meta(r) for _, r in df.iterrows()],
    })
    return out[COLUMNS].reset_index(drop=True)


def fetch_ragtruth(split: str) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset(config.RAGTRUTH_REPO, split=split, revision=config.RAGTRUTH_REVISION)
    return ds.to_pandas()


# --------------------------------------------------------------------------- FaithBench
# Official FaithBench binarization (scripts/binarize.py, "worst" aggregation):
# Benign hallucinations count as faithful; Questionable is ambiguous -> excluded.
FAITHBENCH_LABEL_MAP = {
    "Consistent": "SUPPORTED",
    "Benign": "SUPPORTED",
    "Unwanted": "UNSUPPORTED",
    "Questionable": None,  # dropped
}


def faithbench_to_items(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.reset_index(drop=True).copy()
    unknown = set(df["worst-label"]) - set(FAITHBENCH_LABEL_MAP)
    if unknown:
        raise ValueError(f"unknown FaithBench labels: {unknown}")
    df["row"] = df.index  # row number in the pinned CSV = stable id
    df["label"] = df["worst-label"].map(FAITHBENCH_LABEL_MAP)
    df = df[df["label"].notna()].reset_index(drop=True)
    out = pd.DataFrame({
        "id": "faithbench-" + df["row"].astype(str).str.zfill(4),
        "task": TASK_FAITH,
        "source": "faithbench",
        "subset": df["LLM"].astype(str),
        "context": df["source"].astype(str).str.strip(),
        "question": "Summarize the following passage.",
        "answer": df["summary"].astype(str).str.strip(),
        "answer_b": None,
        "human_label": df["label"],
        "meta": [{"worst_label": r["worst-label"], "best_label": r["best-label"], "generator": r["LLM"]}
                 for _, r in df.iterrows()],
    })
    return out[COLUMNS].reset_index(drop=True)


def fetch_faithbench() -> pd.DataFrame:
    return pd.read_csv(config.FAITHBENCH_URL)


# --------------------------------------------------------------------------- JudgeBench
JUDGEBENCH_LABEL_MAP = {"A>B": "A", "B>A": "B"}


def judgebench_category(source: str) -> str:
    """Collapse the 14 'mmlu-pro-<subject>' sources into 'mmlu-pro'; keep livebench/livecodebench."""
    return "mmlu-pro" if source.startswith("mmlu-pro") else source


def judgebench_to_items(raw: pd.DataFrame) -> pd.DataFrame:
    unknown = set(raw["label"]) - set(JUDGEBENCH_LABEL_MAP)
    if unknown:
        raise ValueError(f"unknown JudgeBench labels: {unknown}")
    df = raw.reset_index(drop=True)

    def meta(r) -> dict:
        oid = r["original_id"]
        return {
            "fine_source": r["source"],
            "response_model": r["response_model"],
            "split": r["split"] if "split" in r else "",
            "original_id": None if pd.isna(oid) else str(oid),
        }

    out = pd.DataFrame({
        "id": "judgebench-" + df["pair_id"].astype(str),
        "task": TASK_PAIR,
        "source": "judgebench",
        "subset": df["source"].astype(str).map(judgebench_category),
        "context": "",
        "question": df["question"].astype(str),
        "answer": df["response_A"].astype(str),
        "answer_b": df["response_B"].astype(str),
        "human_label": df["label"].map(JUDGEBENCH_LABEL_MAP),
        "meta": [meta(r) for _, r in df.iterrows()],
    })
    return out[COLUMNS].reset_index(drop=True)


def fetch_judgebench() -> pd.DataFrame:
    from datasets import load_dataset
    parts = []
    for split in config.JUDGEBENCH_SPLITS:
        d = load_dataset(config.JUDGEBENCH_REPO, split=split, revision=config.JUDGEBENCH_REVISION).to_pandas()
        d["split"] = split
        parts.append(d)
    return pd.concat(parts, ignore_index=True)
