"""Step 1 tests.

Unit tests (offline, tiny fixtures) check the logic.
Integration tests (marked `frozen`) check the real files in data/ and are skipped if absent.
Run:  pytest -q tests/test_step1_data.py
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from ljbench import config
from ljbench.io_utils import read_jsonl, sha256, write_jsonl
from ljbench.loaders import faithbench_to_items, judgebench_to_items, ragtruth_to_items
from ljbench.sampling import estimate_tokens, filter_by_length, split_off, stratified_balanced_sample
from ljbench.schema import COLUMNS, validate

DATA = Path(__file__).resolve().parents[1] / "data"


# ============================================================================ fixtures
@pytest.fixture
def raw_ragtruth():
    span = json.dumps([{"start": 0, "end": 5, "text": "Paris", "label_type": "Evident Conflict"}])
    return pd.DataFrame({
        "id": ["1", "2", "3", "4"],
        "query": ["q1", "q2", "q3", "q4"],
        "context": ["ctx1", "ctx2", "ctx3", "ctx4"],
        "output": ["a1", "a2", "a3", "a4"],
        "task_type": ["QA", "QA", "Summary", "Data2txt"],
        "quality": ["good", "good", "good", "incorrect_refusal"],
        "model": ["m"] * 4,
        "hallucination_labels": ["[]", span, "[]", "[]"],
    })


@pytest.fixture
def raw_faithbench():
    return pd.DataFrame({
        "source": ["s0", "s1", "s2", "s3"],
        "summary": [" y0", "y1", "y2", "y3"],
        "LLM": ["L"] * 4,
        "worst-label": ["Consistent", "Benign", "Unwanted", "Questionable"],
        "best-label": ["Consistent", "Benign", "Benign", "Questionable"],
    })


@pytest.fixture
def raw_judgebench():
    return pd.DataFrame({
        "pair_id": ["p1", "p2"], "original_id": [1.0, float("nan")],
        "source": ["mmlu-pro-law", "livebench-math"], "question": ["Q1", "Q2"],
        "response_model": ["gpt-4o", "gpt-4o"], "response_A": ["ra1", "ra2"],
        "response_B": ["rb1", "rb2"], "label": ["A>B", "B>A"], "split": ["gpt", "gpt"],
    })


def make_items(n_per_cell: int, subsets=("QA", "Summary"), labels=("SUPPORTED", "UNSUPPORTED")):
    rows = []
    for s in subsets:
        for lab in labels:
            for i in range(n_per_cell):
                rows.append({"id": f"{s}-{lab}-{i:03d}", "task": "faithfulness", "source": "x", "subset": s,
                             "context": "c", "question": "q", "answer": "a", "answer_b": None,
                             "human_label": lab, "meta": {}})
    return pd.DataFrame(rows, columns=COLUMNS)


# ============================================================================ loaders
def test_ragtruth_labels_and_quality_filter(raw_ragtruth):
    out = ragtruth_to_items(raw_ragtruth, "test")
    validate(out)
    assert list(out["id"]) == ["ragtruth-1", "ragtruth-2", "ragtruth-3"]   # refusal row dropped
    assert list(out["human_label"]) == ["SUPPORTED", "UNSUPPORTED", "SUPPORTED"]
    assert out.loc[1, "meta"]["span_types"] == ["Evident Conflict"]
    assert out["answer_b"].isna().all()


def test_faithbench_worst_label_mapping(raw_faithbench):
    out = faithbench_to_items(raw_faithbench)
    validate(out)
    # Consistent & Benign -> SUPPORTED, Unwanted -> UNSUPPORTED, Questionable dropped
    assert dict(zip(out["id"], out["human_label"])) == {
        "faithbench-0000": "SUPPORTED", "faithbench-0001": "SUPPORTED", "faithbench-0002": "UNSUPPORTED"}
    assert out.loc[0, "answer"] == "y0"   # whitespace stripped


def test_faithbench_unknown_label_raises(raw_faithbench):
    raw_faithbench.loc[0, "worst-label"] = "Weird"
    with pytest.raises(ValueError):
        faithbench_to_items(raw_faithbench)


def test_judgebench_mapping(raw_judgebench):
    out = judgebench_to_items(raw_judgebench)
    validate(out)
    assert list(out["human_label"]) == ["A", "B"]
    assert list(out["subset"]) == ["mmlu-pro", "livebench-math"]
    assert out.loc[0, "answer"] == "ra1" and out.loc[0, "answer_b"] == "rb1"
    assert out.loc[1, "meta"]["original_id"] is None


def test_validate_catches_problems():
    df = make_items(2)
    df.loc[0, "human_label"] = "MAYBE"
    df.loc[1, "id"] = df.loc[2, "id"]
    with pytest.raises(ValueError, match="duplicate ids.*invalid labels|invalid labels|duplicate ids"):
        validate(df)


# ============================================================================ sampling
def test_estimate_tokens():
    assert estimate_tokens(None) == 0 and estimate_tokens("") == 0
    assert estimate_tokens("x" * 35) == 10 and estimate_tokens("x" * 36) == 11


def test_filter_by_length():
    df = make_items(2)
    df.loc[0, "context"] = "x" * int(config.MAX_INPUT_TOKENS * config.CHARS_PER_TOKEN + 100)
    kept, dropped = filter_by_length(df)
    assert len(dropped) == 1 and len(kept) == len(df) - 1


def test_stratified_sample_exact_cells():
    s = stratified_balanced_sample(make_items(50), 40, "subset")
    assert (pd.crosstab(s["subset"], s["human_label"]) == 10).all().all()
    assert s["id"].is_unique


def test_sampling_is_deterministic_and_order_invariant():
    df = make_items(50)
    a = stratified_balanced_sample(df, 40, "subset", seed=1)
    b = stratified_balanced_sample(df.sample(frac=1, random_state=99), 40, "subset", seed=1)
    c = stratified_balanced_sample(df, 40, "subset", seed=2)
    assert list(a["id"]) == list(b["id"])          # input order doesn't matter
    assert set(a["id"]) != set(c["id"])             # seed does matter


def test_sampling_raises_when_cell_too_small_or_indivisible():
    with pytest.raises(ValueError, match="need"):
        stratified_balanced_sample(make_items(5), 40, "subset")
    with pytest.raises(ValueError, match="divisible"):
        stratified_balanced_sample(make_items(50), 42, "subset")


def test_split_off_disjoint_and_complete():
    df = make_items(10)
    held, rest = split_off(df, 7)
    assert len(held) == 7 and len(rest) == len(df) - 7
    assert not set(held["id"]) & set(rest["id"])


def test_jsonl_roundtrip(tmp_path):
    df = make_items(2)
    df.at[0, "meta"] = {"k": [1, 2], "u": "é"}
    p = tmp_path / "x.jsonl"
    write_jsonl(df, p)
    back = read_jsonl(p)
    assert back["id"].tolist() == df["id"].tolist()
    assert back.loc[0, "meta"] == {"k": [1, 2], "u": "é"}
    assert back["answer_b"].isna().all()
    write_jsonl(back, tmp_path / "y.jsonl")
    assert sha256(p) == sha256(tmp_path / "y.jsonl")   # byte-stable


# ============================================================================ frozen files
frozen = pytest.mark.skipif(not (DATA / "manifest.json").exists(), reason="run build_data first")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frozen_files(manifest):
    return {name: read_jsonl(DATA / name) for name in manifest["files"]}


@frozen
def test_hashes_match_manifest(manifest):
    for name, info in manifest["files"].items():
        assert sha256(DATA / name) == info["sha256"], f"{name} changed after freezing!"


@frozen
def test_frozen_schema_and_sizes(frozen_files):
    expected = {"ragtruth_test_600.jsonl": 600, "faithbench_250.jsonl": 250,
                "ragtruth_pilot_50.jsonl": 50, "ragtruth_fewshot_pool.jsonl": 30,
                "judgebench_dev_20.jsonl": 20}
    for name, n in expected.items():
        assert len(frozen_files[name]) == n, name
    for df in frozen_files.values():
        validate(df)


@frozen
def test_frozen_balance(frozen_files):
    rt = frozen_files["ragtruth_test_600.jsonl"]
    assert (pd.crosstab(rt["subset"], rt["human_label"]) == 100).all().all()
    fb = frozen_files["faithbench_250.jsonl"]
    assert fb["human_label"].value_counts().to_dict() == {"SUPPORTED": 125, "UNSUPPORTED": 125}


@frozen
def test_no_leakage_between_splits(frozen_files):
    test = frozen_files["ragtruth_test_600.jsonl"]
    for name in ["ragtruth_pilot_50.jsonl", "ragtruth_fewshot_pool.jsonl"]:
        tr = frozen_files[name]
        assert not set(tr["id"]) & set(test["id"]), name
        assert not set(tr["context"]) & set(test["context"]), f"{name} shares a source document with test"
        assert all(m["split"] == "train" for m in tr["meta"])
    assert not set(frozen_files["ragtruth_pilot_50.jsonl"]["id"]) & set(frozen_files["ragtruth_fewshot_pool.jsonl"]["id"])
    dev = frozen_files["judgebench_dev_20.jsonl"]
    ev = next(v for k, v in frozen_files.items() if k.startswith("judgebench_eval"))
    assert not set(dev["id"]) & set(ev["id"])
    assert len(dev) + len(ev) <= 620


@frozen
def test_all_within_length_budget(frozen_files):
    from ljbench.sampling import input_tokens
    for name, df in frozen_files.items():
        assert input_tokens(df).max() <= config.MAX_INPUT_TOKENS, name
