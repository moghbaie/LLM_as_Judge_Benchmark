"""Step 2 tests: prompt rendering and output parsing (offline)."""
import re
from pathlib import Path

import pytest

from ljbench import prompts
from ljbench.io_utils import read_jsonl
from ljbench.parsing import (normalize_confidence, normalize_label, parse_claims, parse_judgement,
                             parse_verification)

DATA = Path(__file__).resolve().parents[1] / "data"

FAITH = {"id": "t1", "task": "faithfulness", "context": "Price is $5 {not json}.", "question": "Q?",
         "answer": "It costs $5.", "answer_b": None, "human_label": "SUPPORTED"}
PAIR = {"id": "p1", "task": "pairwise", "context": "", "question": "2+2?", "answer": "4",
        "answer_b": "5", "human_label": "A"}
ANCHORS = [{"context": f"c{i}", "question": "q", "answer": "a", "label": "UNSUPPORTED",
            "reasoning": "r", "split": "train"} for i in range(3)]


# ============================================================================ rendering
def _user(msgs):
    assert [m["role"] for m in msgs] == ["system", "user"]
    return msgs[1]["content"]


def test_m1_contains_item_text_verbatim():
    u = _user(prompts.faith_m1(FAITH))
    assert "Price is $5 {not json}." in u and "It costs $5." in u and "Q?" in u


@pytest.mark.parametrize("builder", [
    lambda: prompts.faith_m1(FAITH),
    lambda: prompts.faith_m2_extract(FAITH),
    lambda: prompts.faith_m2_verify(FAITH, ["claim one", "claim two"]),
    lambda: prompts.faith_m3(FAITH, ANCHORS),
    lambda: prompts.pairwise_m1(PAIR),
])
def test_no_unfilled_placeholders(builder):
    u = _user(builder())
    # the only '$' allowed is the one that came from the item text ("$5")
    leftovers = [m for m in re.findall(r"\$[A-Za-z_]\w*", u)]
    assert leftovers == [], leftovers


def test_missing_value_raises():
    with pytest.raises(KeyError):
        prompts.render("faith_m1", context="c", question="q")   # no answer


def test_m2_verify_numbers_claims():
    u = _user(prompts.faith_m2_verify(FAITH, ["A is B", "C is D"]))
    assert "1. A is B\n2. C is D" in u


def test_m3_guards():
    with pytest.raises(ValueError, match="exactly 3"):
        prompts.faith_m3(FAITH, ANCHORS[:2])
    with pytest.raises(ValueError, match="train"):
        prompts.faith_m3(FAITH, ANCHORS[:2] + [{**ANCHORS[2], "split": "test"}])


def test_swap_pair_flips_answers_and_gold():
    s = prompts.swap_pair(PAIR)
    assert (s["answer"], s["answer_b"], s["human_label"]) == ("5", "4", "B")
    assert prompts.swap_pair(s) == PAIR                       # swapping twice is identity
    assert PAIR["answer"] == "4"                              # original untouched
    assert "<response_a>\n5\n</response_a>" in _user(prompts.pairwise_m1(s))


def test_same_item_same_hash_and_version_hashes():
    assert prompts.messages_hash(prompts.faith_m1(FAITH)) == prompts.messages_hash(prompts.faith_m1(dict(FAITH)))
    assert prompts.messages_hash(prompts.faith_m1(FAITH)) != prompts.messages_hash(prompts.pairwise_m1(PAIR))
    h = prompts.template_hashes()
    assert {"system", "faith_m1", "faith_m2_extract", "faith_m2_verify", "faith_m3", "pairwise_m1"} <= set(h)


@pytest.mark.skipif(not (DATA / "manifest.json").exists(), reason="frozen data missing")
def test_renders_every_frozen_item():
    for name, builder in [("ragtruth_test_600.jsonl", prompts.faith_m1), ("faithbench_250.jsonl", prompts.faith_m1),
                          ("judgebench_eval_600.jsonl", prompts.pairwise_m1)]:
        for item in read_jsonl(DATA / name).to_dict(orient="records"):
            u = _user(builder(item))
            assert item["answer"] in u


# ============================================================================ parsing: judgements
F, P = "faithfulness", "pairwise"
CASES = [
    # (raw output, task, expected label, expected confidence, expected mode)
    ('{"reasoning": "ok", "label": "SUPPORTED", "confidence": 90}', F, "SUPPORTED", 90, "strict"),
    ('{"reasoning":"x","label":"unsupported","confidence":"75"}', F, "UNSUPPORTED", 75, "strict"),
    ('```json\n{"reasoning": "x", "label": "UNSUPPORTED", "confidence": 60}\n```', F, "UNSUPPORTED", 60, "fenced"),
    ('Sure! Here is my verdict:\n{"reasoning": "x", "label": "SUPPORTED", "confidence": 80}\nHope it helps.', F, "SUPPORTED", 80, "extracted"),
    ('<think>Claim 2 says {"label": "SUPPORTED"} hmm, no.</think>{"reasoning": "x", "label": "UNSUPPORTED", "confidence": 70}', F, "UNSUPPORTED", 70, "strict"),
    ('{"reasoning": "x", "label": "SUPPORTED", "confidence": 90,}', F, "SUPPORTED", 90, "repaired"),
    ("{'reasoning': 'x', 'label': 'SUPPORTED', 'confidence': 55}", F, "SUPPORTED", 55, "repaired"),
    ('{"reasoning": "x", "label": "NOT SUPPORTED", "confidence": 65}', F, "UNSUPPORTED", 65, "strict"),
    ('{"reasoning": "x", "label": "SUPPORTED", "confidence": 0.85}', F, "SUPPORTED", 85, "strict"),
    ('{"reasoning": "x", "label": "SUPPORTED", "confidence": "90%"}', F, "SUPPORTED", 90, "strict"),
    ('{"reasoning": "x", "label": "SUPPORTED", "confidence": 140}', F, "SUPPORTED", 100, "strict"),
    ('{"reasoning": "x", "label": "SUPPORTED"}', F, "SUPPORTED", None, "strict"),
    ('{"reasoning": "unterminated, "label": "UNSUPPORTED", "confidence": 80', F, "UNSUPPORTED", 80, "regex"),
    ('{"reasoning": "x", "label": "B", "confidence": 70}', P, "B", 70, "strict"),
    ('{"reasoning": "x", "label": "Response A", "confidence": 70}', P, "A", 70, "strict"),
    ('{"reasoning": "x", "label": "a", "confidence": 51}', P, "A", 51, "strict"),
    ('{"step": 1}\n{"reasoning": "final", "label": "B", "confidence": 99}', P, "B", 99, "extracted"),
    # failures -> label None
    ('', F, None, None, "failed"),
    (None, F, None, None, "failed"),
    ('I think the response is supported.', F, None, None, "failed"),
    ('{"reasoning": "x", "label": "MAYBE", "confidence": 50}', F, None, None, "failed"),
    ('{"reasoning": "x", "label": "SUPPORTED", "confidence": 50}', P, None, None, "failed"),  # wrong task
    ('<think>I need to check the claims... the price is', F, None, None, "failed"),          # cut off while thinking
]


@pytest.mark.parametrize("raw,task,label,conf,mode", CASES)
def test_parse_judgement(raw, task, label, conf, mode):
    p = parse_judgement(raw, task)
    assert p.label == label
    assert p.confidence == (None if conf is None else pytest.approx(conf))
    assert p.mode == mode
    assert p.ok == (label is not None)


def test_normalizers():
    assert normalize_label(" **Supported** ", F) == "SUPPORTED"
    assert normalize_label("C", P) is None
    assert normalize_confidence(True) is None and normalize_confidence("high") is None
    assert normalize_confidence(1) == 1 and normalize_confidence(0) == 0     # integers are already 0-100


# ============================================================================ parsing: M2
def test_parse_claims():
    assert parse_claims('{"claims": ["A is B", " ", "C"]}') == (["A is B", "C"], "strict")
    assert parse_claims('{"claims": []}') == ([], "strict")
    assert parse_claims("no json") == (None, "failed")


def test_parse_verification_any_false_is_unsupported():
    p = parse_verification('{"verdicts": [{"claim": 1, "supported": true}, {"claim": 2, "supported": false}], "confidence": 80}', 2)
    assert (p.label, p.confidence, p.mode) == ("UNSUPPORTED", 80, "strict")
    assert p.data["unsupported_claims"] == [2]


def test_parse_verification_all_true_and_string_bools():
    p = parse_verification('{"verdicts": [{"claim": 1, "supported": "true"}, {"claim": 2, "supported": true}], "confidence": 90}', 2)
    assert p.label == "SUPPORTED"


def test_parse_verification_missing_verdict_is_conservative():
    p = parse_verification('{"verdicts": [{"claim": 1, "supported": true}], "confidence": 90}', 2)
    assert (p.label, p.mode, p.data["missing"]) == ("UNSUPPORTED", "partial", [2])


def test_parse_verification_failure():
    assert parse_verification("nope", 2).label is None
