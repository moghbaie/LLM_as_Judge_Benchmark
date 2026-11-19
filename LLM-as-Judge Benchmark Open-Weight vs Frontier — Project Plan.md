# LLM-as-Judge Benchmark: Open-Weight vs Frontier — Project Plan

Nov 18, 2026 · @Mehrnoosh

## Goal and blog angle

The headline question: **how small and how cheap can an LLM judge be before it stops agreeing with humans?** The project benchmarks open-weight models (run on Google Colab and OpenRouter) against frontier models on the same frozen judge tasks. It ends in a public blog post with code and raw predictions.

The blog's three takeaways should be:

1. An accuracy-vs-cost chart showing which judges sit on the Pareto frontier.
2. Where small judges fail, with real examples.
3. A cascade recipe: a cheap judge first, with uncertain cases escalated to a frontier model, and what that costs.

## Judge tasks

Run two tasks: one core task, plus one contrasting task so the post isn't only about hallucinations. Add the third only if time allows.

| Task | What the judge does | Output | Priority |
| --- | --- | --- | --- |
| A. Faithfulness / hallucination detection | Given context + answer, decide if every claim is supported | SUPPORTED / UNSUPPORTED | Core |
| B. Pairwise preference on hard questions | Given a question + two answers, pick the correct or better one | A / B | Second |
| C. Reference-based correctness | Given question + gold answer + model answer, decide if the answer is correct | CORRECT / INCORRECT | Optional |

Task A is where cheap judges matter most in practice, since RAG monitoring runs at high volume. Task B tests reasoning-heavy judging, where small models are expected to fall off sharply. That contrast makes a good second chart for the blog.

## Datasets

Use RAGTruth as the main faithfulness set and JudgeBench for pairwise preference. Sample each once with a fixed seed and freeze the sample as a file in the repo.

| Dataset | Task | Sample to use | License | Why |
| --- | --- | --- | --- | --- |
| [RAGTruth test split](https://hf.co/datasets/flowaicom/RAGTruth_test) (full set: [wandb/RAGTruth-processed](https://hf.co/datasets/wandb/RAGTruth-processed)) | A | 600, stratified by QA / summarization / data-to-text, 50/50 labels | MIT | Human-labelled RAG outputs; the standard set |
| [RAGTruth++](https://hf.co/datasets/blue-guardrails/ragtruth-plus-plus) | A | All 408 | Check card | Re-annotated subset; use it to check how noisy the original labels are |
| FaithBench (GitHub: vectara/FaithBench) | A, hard mode | 200–300 | Check repo | Subtle hallucinations that strong detectors disagree on |
| [LLM-AggreFact](https://hf.co/datasets/lytang/LLM-AggreFact) | A, generalization | 300 across its sub-datasets | CC BY-ND 4.0, gated | Has a public leaderboard you can place your numbers against |
| [HaluBench](https://hf.co/datasets/PatronusAI/HaluBench) | A, optional | 300 | CC BY-NC 2.0 | Mixes in domain QA (finance, medicine) |
| [JudgeBench](https://hf.co/datasets/ScalerLab/JudgeBench) | B | All (under 1,000 pairs) | MIT | Hard pairwise items with objective correct answers |

Total for the 2-week run: about 850 faithfulness items (RAGTruth 600 + FaithBench 250) + JudgeBench. The other datasets are for a follow-up. Request access to LLM-AggreFact early, since it's gated. Check each license before republishing any examples in the blog: the BY-ND and NC licenses limit what you can redistribute.

## Models

The 2-week run uses about 6 judges, one or two per tier, plus MiniCheck; the rest of the candidates below are for a follow-up. Pin exact model versions on the day you run them, since new releases land monthly.

| Tier | Candidates | Where to run |
| --- | --- | --- |
| Small open (≤ 8B) | Qwen3 4B, a \~8B model (Qwen or Llama), Gemma 4 small variant | Colab GPU, with vLLM |
| Mid open (20–35B) | gpt-oss-20b, a Qwen \~27–35B model, Gemma 4 large variant | Colab A100, 4-bit builds for \~30B (OpenRouter as fallback) |
| Large open | DeepSeek V4.1 Flash, GLM-5.3, Kimi K3.1 | OpenRouter |
| Frontier | One flagship + one cheap tier each from Anthropic, OpenAI and Google | Their APIs, or OpenRouter |
| Specialist baselines | Bespoke-MiniCheck-7B, HHEM-2.1-open | Colab (Task A only) |

The specialist baselines answer the reader's obvious question: do you need a general LLM judge at all? Keep one model family across sizes (e.g. Qwen) to get a clean scaling curve.

## Judge methods

Test two prompting methods on Task A and one on Task B. Every model gets the identical prompt, at temperature 0, returning JSON.

| Method | Tasks | Calls per item | How it works |
| --- | --- | --- | --- |
| M1. Binary + short rationale | A, B | 1 | Judge writes 1–2 sentences of reasoning, then a label |
| M2. Claim decomposition | A | 2 | Extract atomic claims, then verify each against the context; any unsupported claim means UNSUPPORTED |
| M3. Few-shot error anchors | A, optional | 1 | M1 plus 3 labelled tricky examples taken from the train split, never the test sample |

For Task B, also run each pair twice with the answer order swapped. Count a judgement as correct only if both orders agree. Position bias is a known weakness of small judges, and it makes a good blog finding.

Ask every judge for a confidence score (0–100) in its JSON as well. The cascade analysis needs it.

Prompt rules:

- Write prompts once, version them in the repo, and never tune them per model.
- Use one JSON schema everywhere, e.g. `{"reasoning": "...", "label": "SUPPORTED", "confidence": 80}`.
- Use structured or guided decoding where the backend supports it (vLLM does). Count output that still fails to parse as a wrong answer, and report the parse-failure rate separately.

## Metrics and analysis

The primary metric is balanced accuracy with a 95% bootstrap confidence interval (1,000 resamples). Two judges count as "different" only if their intervals don't overlap, or if a paired bootstrap says so.

| Metric | Why it matters |
| --- | --- |
| Balanced accuracy + 95% CI | Main ranking number |
| Cohen's κ vs. human labels | Comparable to published judge papers |
| Recall on hallucinations | How many bad answers the judge catches |
| Precision on hallucinations at a realistic 10–20% hallucination rate | How noisy the judge's alerts are in production (reweight the balanced sample) |
| Parse-failure rate | Small models often break the JSON format |
| Position consistency (Task B) | How often the verdict flips when the answers are swapped |
| Cost per 1,000 judgements | API price, or GPU-hour price for Colab runs |
| Throughput (items per minute) | Report the hardware next to it |

Analyses for the blog:

1. **Pareto chart:** balanced accuracy vs. cost per 1,000 (log scale), one dot per model × method.
2. **Size curve:** accuracy vs. parameter count within one family.
3. **Method effect:** does claim decomposition (M2) help small models more than large ones, or less?
4. **Cascade:** send items below a confidence threshold from the best cheap judge to the best frontier judge. Plot accuracy vs. % escalated and report the cheapest point that matches frontier accuracy.
5. **Error gallery:** 4–6 hand-picked failures, grouped by type (subtle numbers, negation, unsupported inference, flagging a correct answer).

## Compute setup: Colab + APIs

Run small and mid-size open models and the specialist baselines on Colab with vLLM. Send everything bigger through OpenRouter or the vendors' APIs. All of it goes through one Python runner that writes results to Google Drive.

**Colab:**

- Colab Pro+ gives an A100 (40 GB, or 80 GB when available; check with \`nvidia-smi\`). Run models in bf16: 4B and \~8B fit easily, gpt-oss-20b fits in its native MXFP4, and \~27–35B models need a 4-bit AWQ/GPTQ build on 40 GB.
- Use vLLM's offline batch mode (`LLM.generate` over the whole list) instead of one request at a time. It's far faster and doubles as your throughput measurement.
- Sessions disconnect. Mount Drive, write one JSONL line per finished item, and on restart skip IDs already done.
- Record GPU type, model quantization and vLLM version in every results file. Colab speed numbers are only comparable within the same GPU.

**APIs:**

- Use OpenRouter's `:free` variants only to debug prompts. Do final runs on paid endpoints, pinned to a single provider, so quantization doesn't change between runs.
- Use async calls with a concurrency limit and retry with backoff.
- Cache every response by (model, prompt hash), so a rerun never pays twice.

**Call volume per model (core run):** 850 items × 3 calls (M1 + M2) ≈ 2,550, plus JudgeBench × 2 orders, ≈ 1,200 more. That's about 3,750 calls.

**Budget:** estimate it as calls × average tokens × price per token. Open models on OpenRouter should cost a few dollars each. Frontier flagships will be most of the bill. Price a 50-item pilot first, then multiply it out to set the budget.

## Tasks by phase

The plan takes 2 weeks (14 days), in five phases. Each phase ends with a check that gates the next. To fit that, the core run is cut down to about 6 judges (2 small open, 1 mid open, 1 large open, 2 frontier) plus MiniCheck. Datasets are limited to RAGTruth, FaithBench and JudgeBench. M3, HaluBench, LLM-AggreFact and RAGTruth++ move to a follow-up post. Run the Colab models overnight while you work on the API runs.

### Phase 1 — Setup and data (days 1–2)

- [ ] Create a GitHub repo: `data/`, `prompts/`, `runner/`, `results/`, `analysis/`
- [ ] Request access to LLM-AggreFact (gated)
- [ ] Write loaders that turn every dataset into one schema: `id, task, context, question, answer(s), human_label, source`
- [ ] Draw the stratified, seeded samples and save them as frozen JSONL files
- [ ] Check label balance and context-length distribution, and drop items longer than the smallest model's context window

Gate: frozen sample files are committed.

### Phase 2 — Runner and prompts (days 3–4)

- [ ] Write the M1, M2 and M3 prompt templates with the JSON schema
- [ ] Build the API runner: async, retries, response cache, append-only JSONL output to Drive
- [ ] Build the Colab runner: vLLM batch mode, guided JSON, resume from Drive
- [ ] Pilot 50 items on 3 models (one small, one large open, one frontier) using OpenRouter `:free` where possible
- [ ] Read 20 pilot outputs by hand and fix prompt and parsing problems, then freeze prompts (git tag `v1`)

Gate: pilot parse failures under 5% for the frontier model, and a cost estimate for the full run.

### Phase 3 — Full runs (days 5–9)

- [ ] Small and mid open models + MiniCheck + HHEM on Colab A100
- [ ] Large open models on OpenRouter, paid, pinned provider
- [ ] Frontier models via API
- [ ] Task B (JudgeBench) with swapped order for every model
- [ ] Log cost, tokens, wall time and hardware for every run

Gate: every model × method cell is filled, with no missing IDs.

### Phase 4 — Analysis (days 10–12)

- [ ] Metrics table with bootstrap CIs
- [ ] Pareto chart, size curve and method-effect chart
- [ ] Cascade simulation (threshold sweep)
- [ ] Error gallery: hand-pick and categorize failures
- [ ] Sanity check: score 50 random items yourself to see how often you agree with the human labels

Gate: every number in the draft traces back to a results file.

### Phase 5 — Write and publish (days 13–14)

- [ ] Draft the blog (outline below)
- [ ] Clean up the repo: README, how-to-reproduce, raw predictions
- [ ] Ask one person to review the draft for overclaiming
- [ ] Publish the post, plus a short thread with the money chart

## Blog outline and deliverables

Deliverables: the blog post, a public GitHub repo (code, frozen samples, prompts, raw predictions) and one Colab notebook that reproduces a small model's run.

Blog outline:

1. **Hook:** "Your hallucination judge costs more than your RAG app." Lead with the one-line answer and the Pareto chart.
2. **Setup:** tasks, datasets, models and methods, in one table and two paragraphs.
3. **Result 1 — the cost/accuracy frontier:** which judges are worth it.
4. **Result 2 — size curve and method effect:** where small judges break, and whether claim decomposition saves them.
5. **Result 3 — pairwise judging is harder:** JudgeBench results and position bias.
6. **How judges fail:** the error gallery.
7. **Recipe:** the cascade setup with its numbers, plus copy-paste prompts.
8. **Limits:** label noise, English only, prices as of the run date.
9. **Reproduce it:** repo and notebook links.

## Risks and pitfalls

| Risk | Mitigation |
| --- | --- |
| Human labels are noisy, which caps every judge's score | Report RAGTruth++ agreement; do your own 50-item check |
| Prompts tuned on the test set inflate results | Tune only on the 50-item pilot drawn from train; freeze before full runs |
| OpenRouter routes to different providers or quantizations | Pin the provider; record it in results |
| Colab disconnects mid-run | Append-only JSONL on Drive and resume by ID |
| Small models output broken JSON | Guided decoding; count failures as wrong and report the rate |
| Frontier models may have seen RAGTruth or JudgeBench in training | Mention it as a limitation; FaithBench and RAGTruth++ are newer |
| Model versions change after the post goes out | Put the exact model IDs and run dates in the post |
| License limits on quoting examples | Check BY-ND and NC terms before quoting dataset text in the blog |
