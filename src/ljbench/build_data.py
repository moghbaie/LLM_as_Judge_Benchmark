"""Step 1 pipeline: fetch -> convert -> profile -> filter -> sample -> freeze.

Run as a script:   python -m ljbench.build_data --out data
Or call the functions one by one from notebooks/01_data.ipynb.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from . import config, loaders
from .io_utils import sha256, write_jsonl
from .sampling import filter_by_length, input_tokens, nested_subsample, split_off, stratified_balanced_sample
from .schema import validate


# ----------------------------------------------------------------------------- display helpers
def preview(df: pd.DataFrame, n: int = 3, width: int = 120, cols=None) -> pd.DataFrame:
    """First n rows with long strings truncated, for printing samples."""
    d = df.head(n)[cols] if cols else df.head(n)

    def cut(v):
        s = v if isinstance(v, str) else json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)
        s = s.replace("\n", " ")
        return s if len(s) <= width else s[: width - 3] + "..."
    return d.apply(lambda col: col.map(cut))


def label_table(df: pd.DataFrame, by: str = "subset") -> pd.DataFrame:
    t = pd.crosstab(df[by], df["human_label"], margins=True, margins_name="TOTAL")
    return t


# ----------------------------------------------------------------------------- pipeline
def fetch_all_raw() -> dict[str, pd.DataFrame]:
    return {
        "ragtruth_test": loaders.fetch_ragtruth("test"),
        "ragtruth_train": loaders.fetch_ragtruth("train"),
        "faithbench": loaders.fetch_faithbench(),
        "judgebench": loaders.fetch_judgebench(),
    }


def convert_all(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    items = {
        "ragtruth_test": loaders.ragtruth_to_items(raw["ragtruth_test"], "test"),
        "ragtruth_train": loaders.ragtruth_to_items(raw["ragtruth_train"], "train"),
        "faithbench": loaders.faithbench_to_items(raw["faithbench"]),
        "judgebench": loaders.judgebench_to_items(raw["judgebench"]),
    }
    for df in items.values():
        validate(df)
    return items


def length_profile(items: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for name, df in items.items():
        t = input_tokens(df)
        rows[name] = {"n": len(df), "p50": int(t.median()), "p90": int(t.quantile(0.9)),
                      "p99": int(t.quantile(0.99)), "max": int(t.max()),
                      f"> {config.MAX_INPUT_TOKENS}": int((t > config.MAX_INPUT_TOKENS).sum())}
    return pd.DataFrame(rows).T


def build_samples(items: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    """Apply length filter, then draw every frozen sample. Returns (samples, drop_report)."""
    kept, dropped = {}, {}
    for name, df in items.items():
        kept[name], d = filter_by_length(df)
        dropped[name] = len(d)

    # Train items must not share a source document with the test sample (RAGTruth reuses
    # one context for 6 generator models); remove any train item whose context is in test.
    test_contexts = set(kept["ragtruth_test"]["context"])
    train = kept["ragtruth_train"]
    leak = train["context"].isin(test_contexts)
    dropped["ragtruth_train_context_overlap"] = int(leak.sum())
    train = train[~leak]

    s = {}
    s["ragtruth_test_600"] = stratified_balanced_sample(kept["ragtruth_test"], config.RAGTRUTH_TEST_N, "subset")
    s["faithbench_250"] = stratified_balanced_sample(kept["faithbench"], config.FAITHBENCH_N, None)
    s["ragtruth_pilot_50"] = stratified_balanced_sample(train, config.RAGTRUTH_PILOT_N, None)
    rest = train[~train["id"].isin(s["ragtruth_pilot_50"]["id"])]
    s["ragtruth_fewshot_pool"] = stratified_balanced_sample(rest, config.RAGTRUTH_FEWSHOT_POOL_N, "subset")
    dev, ev = split_off(kept["judgebench"], config.JUDGEBENCH_DEV_N)
    s["judgebench_dev_20"] = dev
    # shuffle with seed so any prefix of the file is a representative subset, like the others
    s[f"judgebench_eval_{len(ev)}"] = ev.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    # nested API subsets of the frozen samples (see config.API_SUBSETS)
    for name, (parent, n, strata) in config.API_SUBSETS.items():
        s[name] = nested_subsample(s[parent], n, strata)
    for df in s.values():
        validate(df)
    return s, dropped


def save(samples: dict[str, pd.DataFrame], out_dir, dropped: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, df in samples.items():
        p = out_dir / f"{name}.jsonl"
        write_jsonl(df, p)
        files[p.name] = {
            "rows": len(df),
            "sha256": sha256(p),
            "labels": df["human_label"].value_counts().sort_index().to_dict(),
            "subsets": df["subset"].value_counts().sort_index().to_dict(),
        }
    manifest = {
        "created": date.today().isoformat(),
        "seed": config.SEED,
        "max_input_tokens": config.MAX_INPUT_TOKENS,
        "chars_per_token": config.CHARS_PER_TOKEN,
        "sources": {
            "ragtruth": {"repo": config.RAGTRUTH_REPO, "revision": config.RAGTRUTH_REVISION, "license": "MIT"},
            "faithbench": {"url": config.FAITHBENCH_URL, "commit": config.FAITHBENCH_COMMIT,
                           "license": "CC BY-NC-SA 4.0"},
            "judgebench": {"repo": config.JUDGEBENCH_REPO, "revision": config.JUDGEBENCH_REVISION, "license": "MIT"},
        },
        "dropped": dropped,
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)

    raw = fetch_all_raw()
    for name, df in raw.items():
        print(f"\n=== RAW {name}: {df.shape}  columns={list(df.columns)}")
    items = convert_all(raw)
    for name, df in items.items():
        print(f"\n=== CONVERTED {name}: {len(df)} rows")
        print(preview(df, 2, 80, ["id", "subset", "question", "answer", "human_label"]).to_string())
        print(label_table(df).to_string())
    print("\n=== LENGTH PROFILE (estimated tokens)\n", length_profile(items).to_string())
    samples, dropped = build_samples(items)
    manifest = save(samples, args.out, dropped)
    print("\n=== DROPPED", dropped)
    for fname, info in manifest["files"].items():
        print(f"{fname:32s} rows={info['rows']:4d} labels={info['labels']}")


if __name__ == "__main__":
    main()
