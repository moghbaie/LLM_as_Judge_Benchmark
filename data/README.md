# Frozen samples

Built by `python -m ljbench.build_data --out data` (see `manifest.json` for seed, pinned revisions and SHA-256 hashes).
Do not edit these files by hand; `tests/test_step1_data.py` checks the hashes.

`*_api*.jsonl` are balanced subsets judged by the API models (budget), nested inside the full frozen files: a larger subset always contains a smaller one (`sampling.nested_subsample`).

## Sources and licenses

| File(s) | Source | License |
| --- | --- | --- |
| `ragtruth_*.jsonl` | [RAGTruth](https://github.com/ParticleMedia/RAGTruth) via [wandb/RAGTruth-processed](https://hf.co/datasets/wandb/RAGTruth-processed) (Niu et al., 2024) | MIT |
| `faithbench_250.jsonl` | [FaithBench](https://github.com/vectara/FaithBench) (Bao et al., 2024) | CC BY-NC-SA 4.0 — this derived sample is shared under the same license, non-commercial use only |
| `judgebench_*.jsonl` | [JudgeBench](https://hf.co/datasets/ScalerLab/JudgeBench) (Tan et al., 2024) | MIT |

Labels were converted to a unified schema; see `src/ljbench/loaders.py` for the exact mappings.
