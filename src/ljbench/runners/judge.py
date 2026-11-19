"""Run one judge model over a dataset with one method. Backend-agnostic: any client with an
async `complete(model_id, messages, **params) -> Completion` works (OpenRouter, a mock, ...).

Usage in Colab (top-level await works in notebooks):
    async with OpenRouterClient() as client:
        judge = Judge(client, models.get("glm-5.3"), ResponseCache("results/cache/glm-5.3.jsonl"), RunConfig())
        summary = await run(judge, items, "judgebench_dev_20", "m1", ResultStore("results", ...))
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from .. import prompts
from ..parsing import Parsed, parse_claims, parse_judgement, parse_verification
from .openrouter import CallFailed, Completion, FatalAPIError
from .store import ResponseCache, ResultStore, cache_key


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class RunConfig:
    temperature: float = 0.0
    max_tokens: int = 4096           # includes reasoning tokens on most providers
    json_mode: bool = True
    reasoning: dict | None = None    # passed through to OpenRouter, e.g. {"enabled": False}
    seed: int | None = None
    concurrency: int = 8
    budget_usd: float | None = None  # hard stop for NEW spending in this run
    prompt_version: str = prompts.PROMPT_VERSION

    def params(self) -> dict:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens, "json_mode": self.json_mode,
                "reasoning": self.reasoning, "seed": self.seed}


class Judge:
    def __init__(self, client, spec, cache: ResponseCache | None = None, cfg: RunConfig | None = None):
        self.client, self.spec, self.cache, self.cfg = client, spec, cache, cfg or RunConfig()
        self.spent_usd = 0.0
        self.n_calls = 0
        self.n_cached = 0
        self._anchors = None

    def params(self) -> dict:
        """Parameters actually sent: no temperature for endpoints that reject it."""
        p = self.cfg.params()
        if not self.spec.temperature_supported:
            p["temperature"] = None
        return p

    async def call(self, messages: list[dict]) -> Completion:
        params = self.params()
        key = cache_key(self.spec.model_id, self.spec.provider, messages, params)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            self.n_cached += 1
            return hit
        if self.cfg.budget_usd is not None and self.spent_usd >= self.cfg.budget_usd:
            raise BudgetExceeded(f"budget ${self.cfg.budget_usd:.2f} reached")
        c = await self.client.complete(self.spec.model_id, messages, provider=self.spec.provider, **params)
        if c.cost_usd is None and self.spec.price_in is not None:
            c.cost_usd = self.spec.cost_usd(c.tokens_in, c.tokens_out)
        self.spent_usd += c.cost_usd or 0.0
        self.n_calls += 1
        if self.cache is not None:
            self.cache.put(key, c)
        return c

    # ------------------------------------------------------------------ one item
    async def judge(self, item: dict, method: str, order: str = "orig") -> dict:
        task = item["task"]
        if task not in self.spec.tasks:
            raise ValueError(f"{self.spec.key} does not do task {task}")
        if method not in prompts.METHODS[task]:
            raise ValueError(f"method {method} not defined for task {task}")
        it = prompts.swap_pair(item) if order == "swap" else item
        v = self.cfg.prompt_version
        calls: list[Completion] = []

        if task == "pairwise":
            c = await self.call(prompts.pairwise_m1(it, v)); calls.append(c)
            parsed = parse_judgement(c.text, task)
        elif method == "m1":
            c = await self.call(prompts.faith_m1(it, v)); calls.append(c)
            parsed = parse_judgement(c.text, task)
        elif method == "m3":
            if self._anchors is None:
                self._anchors = prompts.load_anchors(v)
            c = await self.call(prompts.faith_m3(it, self._anchors, v)); calls.append(c)
            parsed = parse_judgement(c.text, task)
        else:  # m2: extract claims, then verify them
            c1 = await self.call(prompts.faith_m2_extract(it, v)); calls.append(c1)
            claims, mode = parse_claims(c1.text)
            if claims is None:
                parsed = Parsed(mode="failed", error="claim extraction unparseable")
            elif not claims:
                parsed = Parsed(label="SUPPORTED", confidence=None, mode="no_claims")
            else:
                c2 = await self.call(prompts.faith_m2_verify(it, claims, v)); calls.append(c2)
                parsed = parse_verification(c2.text, len(claims))
                parsed.data["claims"] = claims

        return {
            "row_id": item["id"] if order == "orig" else f"{item['id']}::swap",
            "item_id": item["id"],
            "order": order,
            "task": task,
            "source": item["source"],
            "subset": item["subset"],
            "model_key": self.spec.key,
            "model_id": self.spec.model_id,
            "provider_requested": self.spec.provider,
            "provider_used": calls[-1].provider,
            "method": method,
            "prompt_version": v,
            "params": self.params(),
            "gold": it["human_label"],
            "label": parsed.label,
            "correct": parsed.label == it["human_label"],
            "confidence": parsed.confidence,
            "parse_mode": parsed.mode,
            "parse_error": parsed.error,
            "reasoning": parsed.reasoning,
            "parsed_data": parsed.data,
            "raw_outputs": [c.text for c in calls],
            "calls": [{k: v for k, v in c.to_dict().items() if k != "text"} for c in calls],
            "tokens_in": sum(c.tokens_in for c in calls),
            "tokens_out": sum(c.tokens_out for c in calls),
            "reasoning_tokens": sum(c.reasoning_tokens for c in calls),
            "cost_usd": round(sum(c.cost_usd or 0 for c in calls), 8),
            "latency_s": round(sum(c.latency_s for c in calls), 3),
            "all_cached": all(c.cached for c in calls),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


# ---------------------------------------------------------------------- one dataset
async def run(judge: Judge, items: list[dict], dataset: str, method: str, store: ResultStore, *,
              limit: int | None = None, progress_every: int = 25, log=print) -> dict:
    """Judge every (item, order) not already in `store`. Pairwise items run in both orders."""
    bad = {it["task"] for it in items} - set(judge.spec.tasks)
    if bad:
        raise ValueError(f"{judge.spec.key} cannot judge task(s) {bad}")
    units = [(it, o) for it in items[:limit] for o in (("orig", "swap") if it["task"] == "pairwise" else ("orig",))]
    done = store.done
    todo = [(it, o) for it, o in units if (it["id"] if o == "orig" else f"{it['id']}::swap") not in done]
    log(f"[{judge.spec.key} | {method} | {dataset}] {len(units)} units, {len(units) - len(todo)} already done, "
        f"{len(todo)} to run (concurrency {judge.cfg.concurrency})")

    sem = asyncio.Semaphore(judge.cfg.concurrency)
    stop = asyncio.Event()
    stats = {"written": 0, "failed": 0, "parse_fail": 0, "errors": [], "stopped": None}
    t0 = time.perf_counter()

    async def one(it, order):
        if stop.is_set():
            return
        async with sem:
            if stop.is_set():
                return
            try:
                row = await judge.judge(it, method, order)
            except BudgetExceeded as e:
                stats["stopped"] = str(e); stop.set(); return
            except FatalAPIError as e:
                stats["stopped"] = f"fatal: {e}"; stop.set(); return
            except CallFailed as e:
                stats["failed"] += 1
                stats["errors"].append({"item_id": it["id"], "order": order, "error": str(e)[:300]})
                return
        store.write(row)
        stats["written"] += 1
        stats["parse_fail"] += row["label"] is None
        n = stats["written"]
        if progress_every and n % progress_every == 0:
            log(f"  {n}/{len(todo)} done | new spend ${judge.spent_usd:.4f} | failed {stats['failed']} | "
                f"parse failures {stats['parse_fail']}")

    await asyncio.gather(*(one(it, o) for it, o in todo))
    elapsed = time.perf_counter() - t0
    summary = {
        "model": judge.spec.key, "method": method, "dataset": dataset,
        "units_total": len(units), "written": stats["written"], "call_failures": stats["failed"],
        "parse_failures": stats["parse_fail"], "api_calls": judge.n_calls, "cache_hits": judge.n_cached,
        "new_spend_usd": round(judge.spent_usd, 4), "elapsed_s": round(elapsed, 1),
        "units_per_min": round(60 * stats["written"] / elapsed, 1) if elapsed > 0 else None,
        "stopped": stats["stopped"], "errors": stats["errors"][:10],
        "complete": len(store.done & {u[0]["id"] if u[1] == "orig" else f"{u[0]['id']}::swap" for u in units}) == len(units),
    }
    log(f"[{judge.spec.key} | {method} | {dataset}] wrote {summary['written']}, failures {summary['call_failures']}, "
        f"parse failures {summary['parse_failures']}, new spend ${summary['new_spend_usd']}, "
        f"complete={summary['complete']}" + (f", STOPPED: {summary['stopped']}" if summary["stopped"] else ""))
    return summary

