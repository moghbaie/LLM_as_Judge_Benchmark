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

**21 models: 14 core + 7 optional.** Only one Claude model is in the core run (Opus 5.5; Haiku 5.5 moved to optional on 2026-11-19). All are pinned in `src/ljbench/models.py`; `tests/test_models.py` locks the roster.

- API models are pinned to one OpenRouter endpoint with no fallbacks.
- Colab models are pinned to a Hugging Face commit.
- Prices and endpoints were checked live on 2026-11-19 (`python -m ljbench.run_api check`).

| # | Model | Total size | Active / token | Tier | Runs on | Pin (endpoint or HF commit) | $/1M in / out | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | HHEM-2.1-open | 0.11B | — | Specialist (Task A) | Colab | `de9a5e6` | GPU | core |
| 2 | Qwen3.5-4B | 4.7B | — | Small open | Colab vLLM | `851bf6e` | GPU | core |
| 3 | Bespoke-MiniCheck-7B | 7.7B | — | Specialist (Task A) | Colab | `1ed7786` | GPU | core |
| 4 | Gemma 4 E4B-it | 8.0B | ~4B effective | Small open | Colab vLLM | `4ecc2ba` | GPU | core |
| 5 | gpt-oss-20b | 20.9B | 3.6B | Mid open | Colab vLLM (native MXFP4) | `6cee5e8` | GPU | core |
| 6 | Qwen3.8-27B | 27.8B | — | Mid open | Colab vLLM (official FP8) | `c3a462c` | GPU | core |
| 7 | Muse Glimmer 30B | 29.8B | — | Mid open | OpenRouter | `deepinfra/bf16` | 0.30 / 1.20 | optional |
| 8 | Gemma 4 31B-it | 31.3B | — | Mid open | OpenRouter | `novita/bf16` | 0.14 / 0.40 | optional |
| 9 | GLM-5.3 | 753B | n/p | Large open | OpenRouter | `z-ai/fp8` | 1.40 / 4.40 | core |
| 10 | DeepSeek V4.1 Flash | 763B | n/p | Large open | OpenRouter | `deepinfra/fp8` | 0.14 / 0.42 | core |
| 11 | Muse Spark Open | 816B | 42B | Large open | OpenRouter | `together` | 1.25 / 4.25 | core |
| 12 | Qwen3.8 2.4T-A95B | 2.4T | 95B | Large open | OpenRouter | `alibaba` | 2.00 / 6.00 | optional |
| 13 | Kimi K3.1 Pro | 2.78T | n/p | Large open | OpenRouter | `moonshotai/mxfp4` | 2.00 / 10.00 | core |
| 14 | Qwen3.8 Max | undisclosed | — | Closed reference | OpenRouter | `alibaba` | 2.00 / 6.00 | core |
| 15 | Claude Opus 5.5 | undisclosed | — | Frontier flagship | OpenRouter | `anthropic` | 4.00 / 20.00 | core |
| 16 | Claude Haiku 5.5 | undisclosed | — | Frontier cheap | OpenRouter | `anthropic` | 0.10 / 0.50 | optional |
| 17 | GPT-6 Luna | undisclosed | — | Frontier cheap | OpenRouter | `openai` | 0.10 / 0.50 | core |
| 18 | Gemini 3.8 Flash | undisclosed | — | Frontier cheap | OpenRouter | `google-ai-studio` | 0.75 / 3.75 | core |
| 19 | Claude Fable 5.2 | undisclosed | — | Frontier flagship | OpenRouter | `anthropic` | 8.00 / 40.00 | optional |
| 20 | GPT-6.2 Astra | undisclosed | — | Frontier flagship | OpenRouter | `openai` | 8.00 / 40.00 | optional |
| 21 | Gemini 4 Argon | undisclosed | — | Frontier flagship | OpenRouter | `google-ai-studio` | 2.00 / 10.00 | optional |

n/p = not published. Only open models can go on the size curve; closed models appear on the cost/accuracy chart only.

**Size curves:**
- **Qwen:** 3.5-4B → 3.8-27B → 3.8 2.4T (open, optional) → 3.8 Max (closed). The 4B model is one generation older.
- **Gemma 4:** E4B → 31B (optional). Same generation, so this is the cleanest curve.
- **Meta Muse:** Glimmer 30B (optional) → Spark Open 816B.

**Pinning rules:**
- **Endpoints.** Each API model has 3–50 OpenRouter endpoints at fp4, fp8 or bf16. The headline price is the cheapest (usually fp4). We pin the model maker's own endpoint, else native or full precision, and never fall back to another endpoint.
- **The two optional ~30B models run via the API in bf16.** That's fairer than 4-bit on a 40GB Colab GPU.
- **Temperature.** The Anthropic and OpenAI endpoints reject `temperature`, so those models run at their default sampling. The response cache makes our result files reproducible, and this is listed as a limitation. Every other model runs at temperature 0. For Gemini we pin AI Studio, which accepts temperature, unlike Vertex.
- **Tier names checked against OpenRouter's model descriptions.** GPT-6.2 Astra is OpenAI's flagship, and GPT-6 Luna is its cost-efficient tier. **Gemini 4 Argon is Google's flagship.** Gemini 4 Neon is a mid-priced model.
- **Licenses:** MiniCheck is CC BY-NC 4.0, and the Kimi, GLM, Muse Spark and Qwen 2.4T/Max licenses are "other". Fine for a non-commercial benchmark; check them before redistributing outputs.

**API cost of the full run** (850 Task A items × 3 calls + 600 Task B pairs × 2 orders = 3,750 calls and 3.9M input tokens per model; specialists do Task A only):

| Model | Short output (300 tok/call) | Reasoning output (1,500 tok/call) |
| --- | --- | --- |
| Claude Opus 5.5 | $38 | $128 |
| Kimi K3.1 Pro | $19 | $64 |
| Qwen3.8 Max | $15 | $42 |
| GLM-5.3 | $10 | $30 |
| Muse Spark Open | $10 | $29 |
| Gemini 3.8 Flash | $7 | $24 |
| DeepSeek V4.1 Flash, GPT-6 Luna | $1 each | $3 each |
| **Core API total (8 models)** | **≈ $101** | **≈ $323** |
| Optional API models (7) | ≈ $191 | ≈ $632 |

- The 6 core Colab models cost GPU compute units only.
- Two ways to cut API cost later: OpenRouter `:batch` variants (half price, asynchronous; the runner would need batch support) and OpenAI's `flex` endpoint (half price).
- The Step 6 pilot measures real output length per model, which decides where between these bounds the run lands.

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
- Colab models: Qwen3.5-4B, Gemma 4 E4B, gpt-oss-20b, Qwen3.8-27B (FP8), HHEM-2.1-open and MiniCheck-7B, each loaded at its pinned HF commit.
- vLLM offline `LLM.generate` with guided JSON, using the same prompts, parser and result schema as the API runner. Log GPU, quantization, vLLM version and throughput.
- MiniCheck-7B and HHEM-2.1-open wrappers that output the same result schema (score → label at 0.5; the score doubles as confidence).
- **Tests:** a smoke run on 20 pilot items per model; result-file schema checks.

### Step 6 — Pilot and prompt freeze
- 50 RAGTruth-train items + 20 JudgeBench-dev items × at least 4 models (Qwen3.5-4B, DeepSeek V4.1 Flash, GPT-6 Luna, Claude Opus 5.5). Every core API model also gets the 20-item dev smoke test to measure output length and cost.
- Read 20 outputs by hand, fix prompts or parsing, then freeze (`git tag v1`).
- **Gate:** frontier parse failures < 5%, plus a cost estimate for the full run (tokens × price table).

### Step 7 — Full runs
- 12 core LLM judges × {M1, M2} × (600 + 250) faithfulness items, plus 600 JudgeBench items × 2 orders. MiniCheck and HHEM run Task A only. Optional models follow if time and budget allow.
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
