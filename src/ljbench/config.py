"""Central configuration: seeds, sample sizes, pinned dataset revisions.

Change nothing here after Step 1 is frozen; the manifest records these values.
"""

SEED = 20261118  # project start date

# Pinned revisions (checked 2026-11-19). Pinning makes the frozen samples reproducible.
RAGTRUTH_REPO = "wandb/RAGTruth-processed"
RAGTRUTH_REVISION = "eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b"

JUDGEBENCH_REPO = "ScalerLab/JudgeBench"
JUDGEBENCH_REVISION = "57dd5e0b9817d07f05ec8f45a91b2ce1e310e308"
JUDGEBENCH_SPLITS = ("gpt", "claude")

FAITHBENCH_COMMIT = "023b72b2996f448eed9d25c882ad130679796ec2"
FAITHBENCH_URL = f"https://raw.githubusercontent.com/vectara/FaithBench/{FAITHBENCH_COMMIT}/FaithBench.csv"

# Sample sizes
RAGTRUTH_TEST_N = 600        # 3 task types x 2 labels x 100
RAGTRUTH_PILOT_N = 50        # from TRAIN split, for prompt debugging
RAGTRUTH_FEWSHOT_POOL_N = 30  # from TRAIN split, M3 anchors are hand-picked from here
FAITHBENCH_N = 250           # 125 / 125
JUDGEBENCH_DEV_N = 20        # held out for prompt debugging; never evaluated

# Nested, balanced subsets judged by the API models (budget: < $40 for the whole API run).
# Colab models still judge the full frozen sets. key: (parent sample, size, stratify-by column)
API_SUBSETS = {
    "ragtruth_test_api240": ("ragtruth_test_600", 240, "subset"),     # 3 task types x 2 labels x 40
    "faithbench_api150": ("faithbench_250", 150, None),              # 75 / 75
    "judgebench_eval_api240": ("judgebench_eval_600", 240, None),    # 120 gold-A / 120 gold-B
}

# Length budget: vLLM max_model_len 8192 = input (<= 6000 est.) + template (~600) + output (<= 1024)
MAX_INPUT_TOKENS = 6000
CHARS_PER_TOKEN = 3.5        # conservative English estimate; replaced by a real tokenizer count in Step 5
