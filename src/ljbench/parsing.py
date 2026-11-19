"""Turn raw judge output into (label, confidence).

Rules (identical for every model):
- Thinking blocks (<think>...</think>) are removed first.
- Parse modes, from cleanest to most lenient: strict > fenced > extracted > repaired > regex.
  The mode is stored with each result, so lenient parsing can be reported separately.
- If no valid label can be found, `label=None`. Metrics count that as a wrong answer, and the
  parse-failure rate is reported separately.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

VALID = {
    "faithfulness": ("SUPPORTED", "UNSUPPORTED"),
    "pairwise": ("A", "B"),
}
SYNONYMS = {
    "NOT SUPPORTED": "UNSUPPORTED", "NOT_SUPPORTED": "UNSUPPORTED", "NOTSUPPORTED": "UNSUPPORTED",
    "RESPONSE A": "A", "RESPONSE_A": "A", "RESPONSE B": "B", "RESPONSE_B": "B",
}


@dataclass
class Parsed:
    label: str | None = None
    confidence: float | None = None
    reasoning: str | None = None
    mode: str = "failed"          # strict | fenced | extracted | repaired | regex | failed
    error: str | None = None
    data: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.label is not None


# ------------------------------------------------------------------------ text -> dict
_THINK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)
_THINK_OPEN = re.compile(r"<(think|thinking|reasoning)>.*", re.S | re.I)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def strip_thinking(text: str) -> str:
    """Remove closed thinking blocks. For an unclosed one (output cut off, or tag never closed),
    keep the text before it if that holds JSON, otherwise keep the text after the opening tag."""
    text = _THINK.sub("", text)
    m = _THINK_OPEN.search(text)
    if m:
        before = text[: m.start()]
        text = before if "{" in before else text[m.start() + len(m.group(1)) + 2:]
    return text


def _candidates(text: str):
    """Every JSON object that json can decode, starting at each '{'."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
            if isinstance(obj, dict):
                yield obj
        except json.JSONDecodeError:
            continue


def _repair(s: str):
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)                       # trailing commas
    for attempt in (s, re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", s)))):
        try:
            obj = json.loads(attempt)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(attempt)                   # single quotes, Python literals
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_json(text: str | None, want_key: str | None = None) -> tuple[dict | None, str]:
    """Return (dict, mode). Prefers objects that contain `want_key`."""
    if not text or not text.strip():
        return None, "failed"
    t = strip_thinking(text).strip()

    def good(o):
        return isinstance(o, dict) and (want_key is None or want_key in o)

    try:
        obj = json.loads(t)
        if good(obj):
            return obj, "strict"
    except json.JSONDecodeError:
        pass
    for block in _FENCE.findall(t):
        try:
            obj = json.loads(block.strip())
            if good(obj):
                return obj, "fenced"
        except json.JSONDecodeError:
            pass
    found = [o for o in _candidates(t) if good(o)]
    if found:
        return found[-1], "extracted"          # last one = the final answer
    if "{" in t and "}" in t:
        obj = _repair(t[t.index("{"): t.rindex("}") + 1])
        if good(obj):
            return obj, "repaired"
    return None, "failed"


# ------------------------------------------------------------------------ field normalisation
def normalize_label(value, task: str) -> str | None:
    if value is None:
        return None
    s = re.sub(r"[\s\"'`.*:]+$", "", re.sub(r"^[\s\"'`*:]+", "", str(value))).upper()
    s = re.sub(r"\s+", " ", s)
    s = SYNONYMS.get(s, s)
    return s if s in VALID[task] else None


def normalize_confidence(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if not m:
            return None
        pct = "%" in value
        value = float(m.group())
        if pct:
            return max(0.0, min(100.0, value))
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    if 0 < v < 1:          # a probability like 0.85 -> 85
        v *= 100
    return max(0.0, min(100.0, v))


# ------------------------------------------------------------------------ public parsers
_LABEL_RE = {
    "faithfulness": re.compile(r"[\"']?label[\"']?\s*[:=]\s*[\"']?(UNSUPPORTED|SUPPORTED|NOT[ _]SUPPORTED)\b", re.I),
    "pairwise": re.compile(r"[\"']?label[\"']?\s*[:=]\s*[\"']?(?:RESPONSE[ _])?([AB])\b", re.I),
}
_CONF_RE = re.compile(r"[\"']?confidence[\"']?\s*[:=]\s*[\"']?(\d+(?:\.\d+)?)", re.I)


def parse_judgement(text: str | None, task: str) -> Parsed:
    obj, mode = extract_json(text, want_key="label")
    if obj is not None:
        label = normalize_label(obj.get("label"), task)
        if label is None:
            return Parsed(mode="failed", error=f"invalid label {obj.get('label')!r}", data=obj)
        r = obj.get("reasoning")
        return Parsed(label, normalize_confidence(obj.get("confidence")), r if isinstance(r, str) else None, mode, None, obj)
    t = strip_thinking(text or "")
    m = _LABEL_RE[task].search(t)
    if m:
        c = _CONF_RE.search(t)
        return Parsed(normalize_label(m.group(1), task), normalize_confidence(c.group(1)) if c else None,
                      None, "regex", None, {})
    return Parsed(mode="failed", error="no JSON object with a label found")


def parse_claims(text: str | None) -> tuple[list[str] | None, str]:
    """M2 step 1. Returns (claims, mode); claims=None means the output could not be parsed."""
    obj, mode = extract_json(text, want_key="claims")
    if obj is None or not isinstance(obj.get("claims"), list):
        return None, "failed"
    claims = [str(c).strip() for c in obj["claims"] if str(c).strip()]
    return claims, mode


def parse_verification(text: str | None, n_claims: int) -> Parsed:
    """M2 step 2. Label is computed by us: any unsupported claim -> UNSUPPORTED.

    Missing verdicts count as unsupported (conservative) and set mode 'partial'.
    """
    obj, mode = extract_json(text, want_key="verdicts")
    if obj is None or not isinstance(obj.get("verdicts"), list):
        return Parsed(mode="failed", error="no verdicts list found")
    supported: dict[int, bool] = {}
    for i, v in enumerate(obj["verdicts"], 1):
        if not isinstance(v, dict):
            continue
        idx = v.get("claim", i)
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = i
        s = v.get("supported")
        if isinstance(s, str):
            s = s.strip().lower() in ("true", "yes", "supported")
        if isinstance(s, bool):
            supported[idx] = s
    missing = [i for i in range(1, n_claims + 1) if i not in supported]
    unsupported = [i for i in range(1, n_claims + 1) if supported.get(i) is False]
    label = "UNSUPPORTED" if (unsupported or missing) else "SUPPORTED"
    data = {"verdicts": obj["verdicts"], "unsupported_claims": unsupported, "missing": missing}
    return Parsed(label, normalize_confidence(obj.get("confidence")), None,
                  "partial" if missing else mode, None, data)
