"""Step 4 tests: OpenRouter client, cache, results store and run loop. Fully offline:
HTTP is replaced by httpx.MockTransport, and the run loop uses a fake client."""
import asyncio
import json

import httpx
import pytest

from ljbench import models
from ljbench.run_api import score_rows
from ljbench.runners.judge import Judge, RunConfig, run
from ljbench.runners.openrouter import (CallFailed, Completion, FatalAPIError, OpenRouterClient, build_body,
                                        check_pin)
from ljbench.runners.store import ResponseCache, ResultStore

SPEC = models.get("glm-5.3")
MSGS = [{"role": "user", "content": "hi"}]


def ok_body(content='{"reasoning":"r","label":"A","confidence":80}', **kw):
    return {"id": "x", "model": SPEC.model_id, "provider": "Z.AI",
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0002,
                      "completion_tokens_details": {"reasoning_tokens": 5}}, **kw}


def client_with(responses):
    """Mock transport that replays `responses` (status, json[, headers]) and records requests."""
    seen = []
    it = iter(responses)

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content) if request.content else None)
        status, body, *hdr = next(it)
        return httpx.Response(status, json=body, headers=hdr[0] if hdr else None)

    c = OpenRouterClient(api_key="sk-test-not-real", transport=httpx.MockTransport(handler),
                         max_retries=3, base_delay=0, max_delay=0)
    return c, seen


def complete(c, **kw):
    async def go():
        try:
            return await c.complete(SPEC.model_id, MSGS, provider=SPEC.provider, **kw)
        finally:
            await c.aclose()
    return asyncio.run(go())


# ============================================================================ client
def test_request_body_pins_provider_and_is_deterministic():
    b = build_body("m", MSGS, provider="z-ai/fp8", temperature=0.0, max_tokens=100, json_mode=True,
                   reasoning=None, seed=None)
    assert b["provider"] == {"only": ["z-ai/fp8"], "allow_fallbacks": False, "require_parameters": True}
    assert b["temperature"] == 0.0 and b["response_format"] == {"type": "json_object"}
    assert b["usage"] == {"include": True} and "reasoning" not in b


def test_parses_usage_cost_and_provider():
    c, seen = client_with([(200, ok_body())])
    r = complete(c)
    assert (r.text, r.tokens_in, r.tokens_out, r.reasoning_tokens) == ('{"reasoning":"r","label":"A","confidence":80}', 100, 20, 5)
    assert r.cost_usd == pytest.approx(0.0002) and r.provider == "Z.AI" and r.attempts == 1
    assert seen[0]["provider"]["only"] == [SPEC.provider]


def test_retries_rate_limit_then_succeeds():
    c, seen = client_with([(429, {"error": "slow down"}, {"retry-after": "0"}), (503, {}), (200, ok_body())])
    assert complete(c).attempts == 3 and len(seen) == 3


def test_retries_provider_error_in_200_body():
    c, seen = client_with([(200, {"error": {"message": "upstream"}}), (200, ok_body())])
    assert complete(c).attempts == 2


def test_gives_up_after_max_retries():
    c, seen = client_with([(500, {})] * 4)
    with pytest.raises(CallFailed, match="gave up after 4"):
        complete(c)
    assert len(seen) == 4


@pytest.mark.parametrize("status", [401, 402, 403])
def test_fatal_errors_stop_immediately(status):
    c, seen = client_with([(status, {"error": "no"})])
    with pytest.raises(FatalAPIError):
        complete(c)
    assert len(seen) == 1


def test_bad_request_not_retried():
    c, seen = client_with([(400, {"error": "bad param"})])
    with pytest.raises(CallFailed, match="not retried"):
        complete(c)
    assert len(seen) == 1


def test_empty_output_retried_twice_then_accepted():
    c, seen = client_with([(200, ok_body(content=""))] * 3)
    r = complete(c)
    assert r.text == "" and len(seen) == 3


def test_check_pin():
    ep = {"data": {"endpoints": [{"tag": SPEC.provider, "quantization": "fp8",
                                  "pricing": {"prompt": str(SPEC.price_in / 1e6), "completion": str(SPEC.price_out / 1e6)},
                                  "supported_parameters": ["response_format", "temperature"]}]}}
    t = httpx.MockTransport(lambda r: httpx.Response(200, json=ep))
    assert check_pin(SPEC, transport=t)["ok"]
    ep["data"]["endpoints"][0]["pricing"]["completion"] = "0.00001"
    assert "price changed" in check_pin(SPEC, transport=t)["problem"]
    ep["data"]["endpoints"][0]["tag"] = "other/fp4"
    assert "not listed" in check_pin(SPEC, transport=t)["problem"]


# ============================================================================ fake client for the run loop
class FakeClient:
    """Answers correctly for faithfulness (from a hidden map) and always says 'A' for pairs
    (a maximally position-biased judge). Can be told to fail on certain items."""

    def __init__(self, fail_on=(), fatal_after=None):
        self.calls = 0
        self.fail_on = set(fail_on)
        self.fatal_after = fatal_after

    async def complete(self, model_id, messages, **params):
        self.calls += 1
        if self.fatal_after is not None and self.calls > self.fatal_after:
            raise FatalAPIError("HTTP 402: out of credits")
        u = messages[-1]["content"]
        if any(tag in u for tag in self.fail_on):
            raise CallFailed("simulated outage")
        if "Break the response below" in u:              # M2 extract
            text = ('{"claims": []}' if "NOCLAIMS" in u else
                    '{"claims": ["c1", "HALLU c2"]}' if "HALLU" in u else '{"claims": ["c1", "c2"]}')
        elif "Check each numbered claim" in u:           # M2 verify
            text = '{"verdicts": [{"claim": 1, "supported": true}, {"claim": 2, "supported": %s}], "confidence": 70}' % (
                "false" if "HALLU" in u else "true")
        elif "<response_a>" in u:
            text = '{"reasoning": "first", "label": "A", "confidence": 90}'
        else:
            text = '{"reasoning": "r", "label": "%s", "confidence": 80}' % ("UNSUPPORTED" if "HALLU" in u else "SUPPORTED")
        return Completion(text=text, tokens_in=1000, tokens_out=50, provider="Fake")


def faith_items(n=6):
    return [{"id": f"f{i}", "task": "faithfulness", "source": "s", "subset": "QA", "context": "ctx",
             "question": "q", "answer": ("HALLU " if i % 2 else "") + f"answer {i}", "answer_b": None,
             "human_label": "UNSUPPORTED" if i % 2 else "SUPPORTED", "meta": {}} for i in range(n)]


def pair_items(n=4):
    return [{"id": f"p{i}", "task": "pairwise", "source": "jb", "subset": "math", "context": "",
             "question": f"q{i}", "answer": "x", "answer_b": "y", "human_label": "A" if i % 2 == 0 else "B",
             "meta": {}} for i in range(n)]


def go(judge, items, store, method="m1", **kw):
    return asyncio.run(run(judge, items, "ds", method, store, log=lambda *_: None, **kw))


def test_run_writes_rows_with_cost_and_metadata(tmp_path):
    judge = Judge(FakeClient(), SPEC, ResponseCache(tmp_path / "cache.jsonl"))
    store = ResultStore(tmp_path, "ds", SPEC.key, "m1")
    s = go(judge, faith_items(), store)
    assert s["written"] == 6 and s["complete"] and s["parse_failures"] == 0
    rows = store.rows()
    assert all(r["correct"] for r in rows)
    r = rows[0]
    assert r["provider_requested"] == SPEC.provider and r["provider_used"] == "Fake"
    assert r["cost_usd"] == pytest.approx(SPEC.cost_usd(1000, 50))   # filled from registry price
    assert r["prompt_version"] == "v0" and r["params"]["temperature"] == 0.0
    assert "sk-" not in store.path.read_text(encoding="utf-8")         # key never written


def test_cache_prevents_paying_twice(tmp_path):
    items = faith_items()
    f1 = FakeClient()
    go(Judge(f1, SPEC, ResponseCache(tmp_path / "c.jsonl")), items, ResultStore(tmp_path / "a", "ds", SPEC.key, "m1"))
    f2 = FakeClient()   # fresh process, same cache file, new results dir
    j2 = Judge(f2, SPEC, ResponseCache(tmp_path / "c.jsonl"))
    s = go(j2, items, ResultStore(tmp_path / "b", "ds", SPEC.key, "m1"))
    assert f1.calls == 6 and f2.calls == 0 and s["cache_hits"] == 6 and s["new_spend_usd"] == 0


def test_resume_after_crash_only_runs_missing(tmp_path):
    items = faith_items()
    store = ResultStore(tmp_path, "ds", SPEC.key, "m1")
    s1 = go(Judge(FakeClient(fail_on={"answer 2", "answer 5"}), SPEC), items, store)
    assert s1["written"] == 4 and s1["call_failures"] == 2 and not s1["complete"]
    with open(store.path, "a", encoding="utf-8") as f:
        f.write('{"row_id": "half-written')          # simulate a crash mid-line
    f2 = FakeClient()
    store2 = ResultStore(tmp_path, "ds", SPEC.key, "m1")   # reopen from disk
    s2 = go(Judge(f2, SPEC), items, store2)
    assert f2.calls == 2 and s2["written"] == 2 and s2["complete"]
    assert sorted(r["row_id"] for r in store2.rows()) == [f"f{i}" for i in range(6)]


def test_pairwise_runs_both_orders_and_flips_gold(tmp_path):
    store = ResultStore(tmp_path, "ds", SPEC.key, "m1")
    s = go(Judge(FakeClient(), SPEC), pair_items(), store)
    assert s["written"] == 8
    rows = {r["row_id"]: r for r in store.rows()}
    assert rows["p0"]["gold"] == "A" and rows["p0::swap"]["gold"] == "B"
    sc = score_rows(store.rows())
    # an always-"A" judge: 50% single-order accuracy but 0 position consistency
    assert sc["acc_single_order"] == pytest.approx(0.5)
    assert sc["position_consistency"] == 0 and sc["acc_both_orders"] == 0


def test_m2_two_stage(tmp_path):
    items = faith_items(4)
    items[0]["answer"] = "NOCLAIMS"
    store = ResultStore(tmp_path, "ds", SPEC.key, "m2")
    fake = FakeClient()
    go(Judge(fake, SPEC), items, store, method="m2")
    rows = {r["row_id"]: r for r in store.rows()}
    assert rows["f0"]["parse_mode"] == "no_claims" and len(rows["f0"]["calls"]) == 1
    assert rows["f1"]["label"] == "UNSUPPORTED" and rows["f1"]["parsed_data"]["unsupported_claims"] == [2]
    assert rows["f2"]["label"] == "SUPPORTED" and len(rows["f2"]["calls"]) == 2
    assert fake.calls == 1 + 2 + 2 + 2


def test_budget_stops_new_spending(tmp_path):
    cost_per_call = SPEC.cost_usd(1000, 50)
    cfg = RunConfig(budget_usd=cost_per_call * 2.5, concurrency=1)
    store = ResultStore(tmp_path, "ds", SPEC.key, "m1")
    s = go(Judge(FakeClient(), SPEC, cfg=cfg), faith_items(), store)
    assert s["written"] == 3 and "budget" in s["stopped"] and not s["complete"]


def test_fatal_error_stops_run(tmp_path):
    store = ResultStore(tmp_path, "ds", SPEC.key, "m1")
    s = go(Judge(FakeClient(fatal_after=2), SPEC, cfg=RunConfig(concurrency=1)), faith_items(), store)
    assert s["written"] == 2 and s["stopped"].startswith("fatal")


def test_unknown_method_rejected(tmp_path):
    with pytest.raises(ValueError):
        asyncio.run(Judge(FakeClient(), SPEC).judge(pair_items()[0], "m2"))


# ============================================================================ roster-specific behaviour
class RecordingClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.params = []

    async def complete(self, model_id, messages, **params):
        self.params.append(params)
        return await super().complete(model_id, messages, **params)


def test_no_temperature_sent_to_endpoints_that_reject_it(tmp_path):
    spec = models.get("claude-haiku-5.5")
    assert not spec.temperature_supported
    rc = RecordingClient()
    store = ResultStore(tmp_path, "ds", spec.key, "m1")
    go(Judge(rc, spec), faith_items(2), store)
    assert all(p["temperature"] is None for p in rc.params)
    assert store.rows()[0]["params"]["temperature"] is None          # recorded as actually sent
    b = build_body("m", MSGS, provider="anthropic", temperature=None, max_tokens=10, json_mode=True,
                   reasoning=None, seed=None)
    assert "temperature" not in b


def test_temperature_zero_sent_where_supported(tmp_path):
    rc = RecordingClient()
    go(Judge(rc, SPEC), faith_items(2), ResultStore(tmp_path, "ds", SPEC.key, "m1"))
    assert all(p["temperature"] == 0.0 for p in rc.params)


def test_specialists_refuse_pairwise(tmp_path):
    spec = models.get("hhem-2.1-open")
    with pytest.raises(ValueError, match="cannot judge"):
        go(Judge(FakeClient(), spec), pair_items(), ResultStore(tmp_path, "ds", spec.key, "m1"))


def test_check_pin_without_temperature():
    spec = models.get("claude-opus-5.5")
    ep = {"data": {"endpoints": [{"tag": "anthropic", "quantization": None,
                                  "pricing": {"prompt": "0.000004", "completion": "0.00002"},
                                  "supported_parameters": ["response_format", "reasoning"]}]}}
    t = httpx.MockTransport(lambda r: httpx.Response(200, json=ep))
    assert check_pin(spec, transport=t)["ok"]
