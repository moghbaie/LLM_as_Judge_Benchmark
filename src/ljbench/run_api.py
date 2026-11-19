"""CLI for API runs and scoring.

  python -m ljbench.run_api check                                   # pins live? prices unchanged? key ok?
  python -m ljbench.run_api run --model glm-5.3 --dataset judgebench_dev_20 --method m1 --budget 1
  python -m ljbench.run_api score --dataset judgebench_dev_20

Results go to results/<dataset>/<model>__<method>.jsonl; the response cache to results/cache/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from . import metrics, models
from .io_utils import read_jsonl
from .runners.judge import Judge, RunConfig, run
from .runners.openrouter import OpenRouterClient, check_pin
from .runners.store import ResponseCache, ResultStore

ROOT = Path(__file__).resolve().parents[2]
DATA, RESULTS = ROOT / "data", ROOT / "results"


def load_items(dataset: str) -> list[dict]:
    return read_jsonl(DATA / f"{dataset}.jsonl").to_dict(orient="records")


async def run_one(model_key: str, dataset: str, method: str, cfg: RunConfig, limit=None, results_dir=RESULTS) -> dict:
    spec = models.get(model_key)
    items = load_items(dataset)
    async with OpenRouterClient() as client:
        judge = Judge(client, spec, ResponseCache(Path(results_dir) / "cache" / f"{spec.key}.jsonl"), cfg)
        return await run(judge, items, dataset, method, ResultStore(results_dir, dataset, spec.key, method), limit=limit)


def score_rows(rows: list[dict]) -> dict:
    """Headline metrics for one results file."""
    task = rows[0]["task"]
    out = {"rows": len(rows), "cost_usd": sum(r["cost_usd"] for r in rows),
           "tokens_out_mean": sum(r["tokens_out"] for r in rows) / len(rows)}
    if task == "pairwise":
        by = {(r["item_id"], r["order"]): r for r in rows}
        ids = sorted({r["item_id"] for r in rows if (r["item_id"], "swap") in by and (r["item_id"], "orig") in by})
        out.update(metrics.pairwise_outcomes([by[(i, "orig")]["gold"] for i in ids],
                                             [by[(i, "orig")]["label"] for i in ids],
                                             [by[(i, "swap")]["label"] for i in ids]))
        out["parse_fail"] = metrics.parse_failure_rate([r["label"] for r in rows])
    else:
        out.update(metrics.summary([r["gold"] for r in rows], [r["label"] for r in rows], task))
    out["cost_per_1000"] = metrics.cost_per_1000(out["cost_usd"], len(rows))
    return out


def score_dataset(dataset: str, results_dir=RESULTS) -> pd.DataFrame:
    recs = []
    for f in sorted((Path(results_dir) / dataset).glob("*__*.jsonl")):
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if rows:
            model_key, method = f.stem.split("__")
            recs.append({"model": model_key, "method": method, **score_rows(rows)})
    return pd.DataFrame(recs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=sorted(models.REGISTRY))
    r.add_argument("--dataset", required=True)
    r.add_argument("--method", default="m1")
    r.add_argument("--limit", type=int)
    r.add_argument("--budget", type=float, required=True, help="hard USD cap for new spending")
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--max-tokens", type=int, default=4096)
    s = sub.add_parser("score")
    s.add_argument("--dataset", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "check":
        for spec in models.MODELS:   # core and optional
            if spec.backend == "openrouter":
                r = check_pin(spec)
                print(f"{'core' if spec.enabled else 'opt ':4s} {spec.key:22s} {spec.provider:18s} "
                      f"ok={r['ok']}  {r.get('problem') or ''}")
        async def key():
            async with OpenRouterClient() as c:
                return await c.key_info()
        info = asyncio.run(key())
        print({k: info.get(k) for k in ("label", "limit", "limit_remaining", "usage")})
    elif a.cmd == "run":
        cfg = RunConfig(budget_usd=a.budget, concurrency=a.concurrency, max_tokens=a.max_tokens)
        print(json.dumps(asyncio.run(run_one(a.model, a.dataset, a.method, cfg, a.limit)), indent=2))
    else:
        pd.set_option("display.width", 250)
        print(score_dataset(a.dataset).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
