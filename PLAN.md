# Implementation Plan — LLM-as-Judge Benchmark (Colab edition)

This is the engineering plan that implements `LLM-as-Judge Benchmark Open-Weight vs Frontier — Project Plan.md`.
Every step follows the same loop:

1. **Code**: a small module in `src/ljbench/` plus a Colab notebook in `notebooks/`.
2. **Show the data**: every download or transform prints a sample (raw rows, then converted rows).
3. **Test**: `pytest` unit tests (fast, offline, using tiny fixtures) plus integration checks on the real output files.
4. **Report**: a results cell that explains what the numbers mean and whether the step's gate passed.

A step is done only when its gate passes. All code is written for Google Colab: Drive is mounted for persistence, the GPU is used only from Step 5, and everything before that runs on a CPU runtime.

## What I changed from the original plan (and why)

Checked against the real data on 2026-11-19:

| Finding | Change |
| --- | --- |
| RAGTruth test has 2,700 items; QA has only **160 hallucinated** (Summary 204, Data2txt 579) | 600 = 3 task types × 2 labels × 100 still works, but it uses 62% of the QA hallucinations. |
| 25 RAGTruth items have `quality` = `incorrect_refusal` / `truncated` | Drop them; they aren't ordinary faithfulness cases. |
| FaithBench is **CC BY-NC-SA 4.0** (800 rows, 10 summarizer LLMs) | Fine for a non-commercial blog. Quotes need attribution and share-alike. Labels follow FaithBench's own "worst-label" binarization: Consistent/Benign → SUPPORTED, Unwanted → UNSUPPORTED, Questionable → dropped. |
| FaithBench has only 238 non-hallucinated rows after dropping Questionable | Sample 250 as 125/125. That's feasible but uses ~half the SUPPORTED pool. |
| JudgeBench has **no train split** (620 pairs: gpt 350 + claude 270) | Hold out 20 pairs as a **dev** set for prompt debugging and evaluate on the other 600. Never look at the 600 before prompts are frozen. |
| Pilot must not touch test data | The RAGTruth pilot (50) and the few-shot anchors for M3 (3) come from the **train** split. |
| Metrics were scheduled after the full runs | Write and test the metrics code **before** the pilot, so the pilot is scored with the same code as the final run. |
| Dataset versions drift | Pin HF dataset commits and the FaithBench git commit, and record them in `data/manifest.json` with file SHA-256 hashes. |

## Judge models

The registry lives in `src/ljbench/models.py` and is tested in `tests/test_models.py`. Each entry pins an exact, versioned backend ID (never a `~…-latest` alias) and its open-weights repo. Prices are from OpenRouter, checked 2026-11-19; re-check them on run day.

| Tier | Model | Backend ID | Pinned endpoint (quant.) | Total params | $/1M in / out | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Large open | Kimi K3.1 Pro | `moonshotai/kimi-k3.1-pro` | `moonshotai/mxfp4` (first-party) | 2.78T | 2.00 / 10.00 | core |
| Large open | DeepSeek V4.1 Flash | `deepseek/deepseek-v4.1-flash` | `deepinfra/fp8` (native fp8; no first-party endpoint) | 763B | 0.14 / 0.42 | core |
| Large open | GLM-5.3 | `z-ai/glm-5.3` | `z-ai/fp8` (first-party) | 753B | 1.40 / 4.40 | core |
| Large open | Muse Spark Open | `meta/muse-spark-open` | `together` | 816B (42B active) | 1.25 / 4.25 | core; HF weights gated |
| Closed reference | Qwen3.8 Max | `qwen/qwen3.8-max-0902` | `alibaba` (only provider) | n/a (closed) | 2.00 / 6.00 | core; **never counted as open** |
| Large open | Qwen3.8 2.4T-A95B | `qwen/qwen3.8-2.4t-a95b` | to pin if enabled | 2.4T (95B active) | 2.00 / 6.00 | optional (disabled): open Qwen flagship for a Qwen size curve |
| Small / mid open, frontier, specialists | to be pinned in Step 5 | | | | | |

**Why pin endpoints:** each model is served by 5–50 OpenRouter endpoints at fp4, fp8 or bf16, and the headline price is the cheapest (usually fp4) one. Free routing would mix quantizations within one model's results. Rule: the model maker's own endpoint, else native precision, never the cheapest fp4. `python -m ljbench.run_api check` re-verifies that pins are live and prices unchanged on run day.

Estimated cost of the core run per model at the **pinned** prices (3,750 calls, 3.9M input tokens: 850 Task A items × 3 calls + 600 Task B pairs × 2 orders, real template sizes):

| Model | Short output (300 tok) | Reasoning output (1,500 tok) |
| --- | --- | --- |
| Kimi K3.1 Pro | $19 | $64 |
| DeepSeek V4.1 Flash | $1 | $3 |
| GLM-5.3 | $10 | $30 |
| Muse Spark Open | $10 | $29 |
| Qwen3.8 Max | $15 | $42 |
| **These 5 together** | **≈ $55** | **≈ $168** |

Output length dominates the cost, so the Step 6 pilot measures real output tokens per model and decides whether reasoning or thinking mode stays on. The same mode must be used for every item of a model. Five large judges is more than the original plan's single large-open slot. They're all API calls, so they cost money but no Colab time.

## Repo layout

```
PLAN.md                     this file
requirements.txt
src/ljbench/
  config.py                 seeds, sizes, pinned revisions
  models.py                 judge model registry (IDs, tiers, params, prices)
  schema.py                 unified item schema + validator
  loaders.py                raw dataset -> unified schema (pure functions + fetchers)
  sampling.py               length filter, stratified balanced sampling, holdout
  io_utils.py               JSONL read/write, hashing, manifest
  build_data.py             Step 1 pipeline (CLI + notebook entrypoint)
  prompts.py / parsing.py   Step 2
  metrics.py                Step 3
  runners/                  Step 4: openrouter.py (client), store.py (cache + results), judge.py (run loop)
                            Step 5: vllm + specialist runners
  run_api.py                CLI: check | run | score
  analysis/                 Step 8
prompts/                    versioned prompt templates (frozen at tag v1)
data/                       frozen samples (*.jsonl) + manifest.json
results/                    raw predictions (one JSONL per model x method x dataset)
notebooks/                  01_data.ipynb, 02_prompts.ipynb, ...
tests/                      test_step1_data.py, test_step2_prompts.py, ...
```

Unified item schema (one JSON object per line):
`id, task, source, subset, context, question, answer, answer_b, human_label, meta`

- `task` ∈ {`faithfulness`, `pairwise`}
- `human_label` ∈ {`SUPPORTED`, `UNSUPPORTED`} or {`A`, `B`}
- `answer_b` is only set for pairwise items
- `meta` holds dataset-specific extras (generator model, error types, ...)

## Steps

### Step 1 — Data: loaders, frozen samples (CPU) ✅ implemented
- Fetch RAGTruth (train + test), FaithBench and JudgeBench at pinned revisions. **Show** raw rows.
- Convert each to the unified schema. **Show** converted rows and label/subset tables.
- Profile length. Drop items whose estimated input exceeds `MAX_INPUT_TOKENS` (6,000, so a vLLM `max_model_len` of 8,192 fits prompt + output).
- Freeze the following files:
  - `ragtruth_test_600.jsonl` (3 subsets × 2 labels × 100)
  - `faithbench_250.jsonl` (125/125)
  - `judgebench_eval_600.jsonl` + `judgebench_dev_20.jsonl`
  - `ragtruth_pilot_50.jsonl` + `ragtruth_fewshot_pool.jsonl` (from train)
- **Tests:** label mapping, filtering, exact cell counts, determinism and input-order invariance of sampling, no train/test or dev/eval overlap, hashes match the manifest.
- **Gate:** all tests pass and the files are committed (or saved to Drive).

### Step 2 — Prompts + robust output parsing (CPU) ✅ implemented (`notebooks/02_prompts_parsing.ipynb`, 41 tests)
- Templates: M1 (binary + rationale), M2 (claim extraction → claim verification), M3 (M1 + 3 anchors from train), and the pairwise prompt (with an A/B swap helper).
- One JSON schema: `{"reasoning": str, "label": ..., "confidence": 0-100}`.
- A parser that handles code fences, extra text and trailing commas, plus a strict mode. Failures return `label=None` and are counted as wrong.
- **Show** rendered prompts for one item of each dataset.
- **Tests:** 20+ parsing cases with messy outputs, swap logic (the swapped gold label flips), and template placeholders all filled.

### Step 3 — Metrics (CPU) ✅ implemented (`notebooks/03_metrics.ipynb`, 12 tests)
- Balanced accuracy, 95% bootstrap CI (1,000 resamples), paired bootstrap difference, Cohen's κ, hallucination recall, precision reweighted to a 10%/20% prevalence, parse-failure rate, position consistency, and cost per 1,000 judgements.
- **Tests:** compare against hand-computed values and `sklearn`, and check that the CI covers the true value on synthetic data.

### Step 4 — API runner (CPU, OpenRouter) ✅ implemented (`notebooks/04_api_runner.ipynb`, 19 tests); smoke test waits for the API key
- Async with a concurrency limit, retry with exponential backoff, cache keyed by `sha256(model, provider, prompt, params)`, append-only JSONL results, and resume by ID.
- Each result row stores: id, model, provider, method, raw output, parsed label, confidence, tokens in/out, cost, latency, timestamp and prompt version.
- **Tests:** a mock client (no network) checks retries, the cache hit that prevents a second paid call, and resume after a simulated crash.
- **Show:** 5 real calls on the dev set with a `:free` model.

### Step 5 — Colab GPU runner + specialist baselines (A100)
- vLLM offline `LLM.generate` with guided JSON. Log GPU, quantization, vLLM version and throughput.
- MiniCheck-7B and HHEM-2.1-open wrappers that output the same result schema (score → label at 0.5; the score doubles as confidence).
- **Tests:** a smoke run on 20 pilot items per model; result-file schema checks.

### Step 6 — Pilot and prompt freeze
- 50 RAGTruth-train items + 20 JudgeBench-dev items × 3 models (small open, large open, frontier).
- Read 20 outputs by hand, fix prompts or parsing, then freeze (`git tag v1`).
- **Gate:** frontier parse failures < 5%, plus a cost estimate for the full run (tokens × price table).

### Step 7 — Full runs
- About 6 judges × {M1, M2} × (600 + 250) faithfulness items, plus 600 JudgeBench items × 2 orders. Add MiniCheck and HHEM.
- **Gate:** a completeness check shows every model × method × dataset cell has all IDs.

### Step 8 — Analysis
- Metrics table with CIs, Pareto chart (accuracy vs. log cost), size curve, method effect, cascade threshold sweep and position-bias chart.
- **Tests:** every number in the table is regenerated from `results/` by one script.

### Step 9 — Error gallery + human check
- Sample failures by category. You score 50 random items yourself to estimate label noise.

### Step 10 — Write-up and release
- Blog draft, README, reproduce notebook, raw predictions.

## How to run in Colab

1. Put this folder in Google Drive (e.g. `MyDrive/LLM_as_Judge_Benchmark`), or push it to GitHub and clone it.
2. Open `notebooks/01_data.ipynb` in Colab. The first cell mounts Drive, installs requirements and adds `src/` to the path.
3. Run all cells. Each step ends with a test cell and an explanation cell.
