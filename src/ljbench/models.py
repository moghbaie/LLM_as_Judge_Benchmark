"""Judge model registry.

One entry per judge. Rules:
- `model_id` is the exact, versioned backend ID. Never use OpenRouter `~...-latest` aliases:
  they can switch to a newer model mid-run.
- Prices are USD per 1M tokens as listed on OpenRouter on `PRICES_CHECKED`; re-check on run day
  and record the provider actually used in every results row (Step 4).
- `open_weights` is True only when downloadable weights exist (`hf_id` set). Closed models are
  never counted as "open" in the analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

PRICES_CHECKED = "2026-11-19"

TIERS = (
    "small_open",        # <= 8B, Colab
    "mid_open",          # 20-35B, Colab A100 or OpenRouter
    "large_open",        # hundreds of B, OpenRouter
    "closed_reference",  # closed weights, not a frontier-lab flagship (e.g. Qwen Max)
    "frontier",          # Anthropic / OpenAI / Google
    "specialist",        # MiniCheck, HHEM (Task A only)
)
BACKENDS = ("openrouter", "vllm", "hf_classifier", "vendor_api")


@dataclass(frozen=True)
class ModelSpec:
    key: str                      # short name used in results file names
    display: str                  # name used in charts / blog
    tier: str
    backend: str
    model_id: str                 # exact backend ID
    hf_id: str | None = None      # open-weights repo, None if closed
    params_total_b: float | None = None
    params_active_b: float | None = None   # MoE active params, None if unknown/dense
    price_in: float | None = None          # USD per 1M input tokens (API backends)
    price_out: float | None = None         # USD per 1M output tokens
    license: str | None = None
    enabled: bool = True          # False = registered but not in the core run
    notes: str = ""

    @property
    def open_weights(self) -> bool:
        return self.hf_id is not None

    def cost_usd(self, tokens_in: float, tokens_out: float) -> float:
        if self.price_in is None or self.price_out is None:
            raise ValueError(f"{self.key}: no API price (GPU cost is computed from GPU-hours in Step 5)")
        return (tokens_in * self.price_in + tokens_out * self.price_out) / 1e6


MODELS: list[ModelSpec] = [
    # ---------------------------------------------------------------- large open (OpenRouter)
    ModelSpec("kimi-k3.1-pro", "Kimi K3.1 Pro", "large_open", "openrouter",
              "moonshotai/kimi-k3.1-pro", "moonshotai/Kimi-K3.1-Pro",
              params_total_b=2780, price_in=0.25, price_out=8.80, license="other",
              notes="High output price: cost is dominated by reasoning/output length."),
    ModelSpec("deepseek-v4.1-flash", "DeepSeek V4.1 Flash", "large_open", "openrouter",
              "deepseek/deepseek-v4.1-flash", "deepseek-ai/DeepSeek-V4.1-Flash",
              params_total_b=763, price_in=0.30, price_out=1.20, license="MIT"),
    ModelSpec("glm-5.3", "GLM-5.3", "large_open", "openrouter",
              "z-ai/glm-5.3", "zai-org/GLM-5.3",
              params_total_b=753, price_in=0.07, price_out=1.40, license="other"),
    ModelSpec("muse-spark-open", "Muse Spark Open", "large_open", "openrouter",
              "meta/muse-spark-open", "meta-models/Muse-Spark-Open-816B-A42B",
              params_total_b=816, params_active_b=42, price_in=1.25, price_out=4.25, license="other",
              notes="HF weights are gated; few providers serve it, so pin the provider on run day."),
    ModelSpec("qwen3.8-2.4t", "Qwen3.8 2.4T-A95B", "large_open", "openrouter",
              "qwen/qwen3.8-2.4t-a95b", "Qwen/Qwen3.8-2.4T-A95B",
              params_total_b=2400, params_active_b=95, price_in=2.00, price_out=6.00, license="other",
              enabled=False,
              notes="Optional open Qwen flagship: top point of a Qwen size curve; same price as Qwen3.8 Max."),
    # ---------------------------------------------------------------- closed reference
    ModelSpec("qwen3.8-max", "Qwen3.8 Max", "closed_reference", "openrouter",
              "qwen/qwen3.8-max-0902", None,
              price_in=2.00, price_out=6.00, license="proprietary",
              notes="Closed weights (API only). Never counted as open in charts."),
    # Small/mid open, frontier and specialist judges are added in Steps 4-5, when their
    # exact IDs are pinned.
]

REGISTRY: dict[str, ModelSpec] = {m.key: m for m in MODELS}


def get(key: str) -> ModelSpec:
    return REGISTRY[key]


def enabled(tier: str | None = None) -> list[ModelSpec]:
    return [m for m in MODELS if m.enabled and (tier is None or m.tier == tier)]


def estimate_run_cost(spec: ModelSpec, n_calls: int, avg_in: float, avg_out: float) -> float:
    """Rough budget: calls x average tokens x price. Replaced by measured pilot tokens in Step 6."""
    return spec.cost_usd(n_calls * avg_in, n_calls * avg_out)
