"""Prompt rendering. Templates live in prompts/<version>/*.txt and are never tuned per model.

Templates use `$name` placeholders (string.Template). Values are inserted verbatim, so
`$` or `{}` inside dataset text cannot break rendering.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from string import Template

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
PROMPT_VERSION = "v0"   # v0 = pilot drafts; frozen as v1 after the Step 6 pilot

METHODS = {
    "faithfulness": ("m1", "m2", "m3"),
    "pairwise": ("m1",),
}


@lru_cache(maxsize=None)
def load_template(name: str, version: str = PROMPT_VERSION) -> str:
    return (PROMPT_DIR / version / f"{name}.txt").read_text(encoding="utf-8").strip()


def render(name: str, version: str = PROMPT_VERSION, **values) -> str:
    """Substitute every placeholder; raises KeyError if one is missing."""
    return Template(load_template(name, version)).substitute(**{k: "" if v is None else str(v) for k, v in values.items()})


def _messages(user: str, version: str) -> list[dict]:
    return [{"role": "system", "content": load_template("system", version)},
            {"role": "user", "content": user}]


# ------------------------------------------------------------------------ Task A: faithfulness
def faith_m1(item: dict, version: str = PROMPT_VERSION) -> list[dict]:
    return _messages(render("faith_m1", version, context=item["context"], question=item["question"],
                            answer=item["answer"]), version)


def faith_m2_extract(item: dict, version: str = PROMPT_VERSION) -> list[dict]:
    return _messages(render("faith_m2_extract", version, answer=item["answer"]), version)


def format_claims(claims: list[str]) -> str:
    return "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))


def faith_m2_verify(item: dict, claims: list[str], version: str = PROMPT_VERSION) -> list[dict]:
    return _messages(render("faith_m2_verify", version, context=item["context"],
                            claims=format_claims(claims)), version)


def format_examples(anchors: list[dict]) -> str:
    """Anchors: dicts with context, question, answer, label, reasoning (hand-picked from TRAIN only)."""
    blocks = []
    for i, a in enumerate(anchors, 1):
        verdict = json.dumps({"reasoning": a["reasoning"], "label": a["label"]}, ensure_ascii=False)
        blocks.append(f"### Example {i}\n<source>\n{a['context']}\n</source>\n"
                      f"<task_given_to_the_ai>\n{a['question']}\n</task_given_to_the_ai>\n"
                      f"<response>\n{a['answer']}\n</response>\nVerdict: {verdict}")
    return "\n\n".join(blocks)


def load_anchors(version: str = PROMPT_VERSION) -> list[dict]:
    p = PROMPT_DIR / version / "fewshot_anchors.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: M3 anchors are hand-picked from data/ragtruth_fewshot_pool.jsonl in Step 6")
    return json.loads(p.read_text(encoding="utf-8"))


def faith_m3(item: dict, anchors: list[dict], version: str = PROMPT_VERSION) -> list[dict]:
    if len(anchors) != 3:
        raise ValueError(f"M3 needs exactly 3 anchors, got {len(anchors)}")
    if any(a.get("split") == "test" for a in anchors):
        raise ValueError("M3 anchors must come from the train split")
    return _messages(render("faith_m3", version, examples=format_examples(anchors), context=item["context"],
                            question=item["question"], answer=item["answer"]), version)


# ------------------------------------------------------------------------ Task B: pairwise
FLIP = {"A": "B", "B": "A"}


def swap_pair(item: dict) -> dict:
    """Same pair with the responses in the other order; the gold label flips accordingly."""
    out = dict(item)
    out["answer"], out["answer_b"] = item["answer_b"], item["answer"]
    out["human_label"] = FLIP[item["human_label"]]
    return out


def pairwise_m1(item: dict, version: str = PROMPT_VERSION) -> list[dict]:
    return _messages(render("pairwise_m1", version, question=item["question"], answer=item["answer"],
                            answer_b=item["answer_b"]), version)


# ------------------------------------------------------------------------ utils
def messages_hash(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def template_hashes(version: str = PROMPT_VERSION) -> dict[str, str]:
    """Hash of every template file: recorded in results so a run is tied to exact prompt text."""
    return {p.stem: hashlib.sha256(p.read_bytes()).hexdigest()[:12]
            for p in sorted((PROMPT_DIR / version).glob("*.txt"))}
