"""Judge model registry: every model in the benchmark, pinned.

Rules:
- `model_id` is the exact, versioned backend ID. Never use OpenRouter `~...-latest` aliases.
- API models are pinned to ONE OpenRouter endpoint (`provider`, an endpoint tag) with no fallbacks,
  so quantization cannot change mid-run. Rule: the model maker's own endpoint if it exists,
  otherwise a full/native-precision endpoint, never the cheapest fp4 one.
- Prices are USD per 1M tokens for the PINNED endpoint on `PRICES_CHECKED` (often above the
  headline OpenRouter price). `runners.openrouter.check_pin()` re-checks them on run day.
- Colab models (vllm / hf_classifier / minicheck) are pinned to a Hugging Face commit (`hf_revision`).
- `temperature_supported=False`: the endpoint rejects `temperature` (current Anthropic and OpenAI
  models). We then send no temperature; outputs may vary between calls, which the response cache
  makes reproducible for our results files. This is reported as a limitation.
- `open_weights` is True only when downloadable weights exist (`hf_id` set). Closed models are
  never counted as "open" in the analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

PRICES_CHECKED = "2026-11-19"

TIERS = (
    "small_open",        # <= 8B
    "mid_open",          # 20-35B
    "large_open",        # hundreds of B
    "closed_reference",  # closed weights, not a frontier-lab model (Qwen Max)
    "frontier",          # Anthropic / OpenAI / Google
    "specialist",        # trained hallucination detectors (Task A only)
)
BACKENDS = ("openrouter", "vllm", "hf_classifier", "minicheck")
ALL_TASKS = ("faithfulness", "pairwise")


@dataclass(frozen=True)
class ModelSpec:
    key: str                      # short name used in results file names
    display: str                  # name used in charts / blog
    tier: str
    backend: str
    model_id: str                 # exact backend ID (OpenRouter id, or HF repo actually loaded)
    hf_id: str | None = None      # open-weights repo, None if closed
    hf_revision: str | None = None  # pinned HF commit for Colab-run models
    params_total_b: float | None = None
    params_active_b: float | None = None   # MoE active params per token, None if dense/unknown
    price_in: float | None = None          # USD per 1M input tokens (pinned endpoint)
    price_out: float | None = None         # USD per 1M output tokens
    license: str | None = None
    provider: str | None = None            # pinned OpenRouter endpoint tag
    quantization: str | None = None        # as served (endpoint) or as loaded (Colab)
    temperature_supported: bool = True
    tasks: tuple[str, ...] = ALL_TASKS
    family: str | None = None              # for size curves
    enabled: bool = True                   # False = optional, not in the core run
    notes: str = ""

    @property
    def open_weights(self) -> bool:
        return self.hf_id is not None

    def cost_usd(self, tokens_in: float, tokens_out: float) -> float:
        if self.price_in is None or self.price_out is None:
            raise ValueError(f"{self.key}: no API price (GPU cost is computed from GPU-hours in Step 5)")
        return (tokens_in * self.price_in + tokens_out * self.price_out) / 1e6


FAITH_ONLY = ("faithfulness",)

MODELS: list[ModelSpec] = [
    # ================================================================ specialists (Colab, Task A only)
    ModelSpec("hhem-2.1-open", "HHEM-2.1-open", "specialist", "hf_classifier",
              "vectara/hallucination_evaluation_model", "vectara/hallucination_evaluation_model",
              hf_revision="de9a5e6ba9ba49531a022dba17e35ffeb2a6fad8", params_total_b=0.11,
              license="apache-2.0", quantization="fp32", tasks=FAITH_ONLY,
              notes="flan-t5-base classifier; score >= 0.5 -> SUPPORTED, score doubles as confidence."),
    ModelSpec("minicheck-7b", "Bespoke-MiniCheck-7B", "specialist", "minicheck",
              "bespokelabs/Bespoke-MiniCheck-7B", "bespokelabs/Bespoke-MiniCheck-7B",
              hf_revision="1ed7786bcda3fa1dc35f7c4ed9e3f36b785d33b8", params_total_b=7.7,
              license="CC BY-NC 4.0", quantization="bf16", tasks=FAITH_ONLY,
              notes="Non-commercial license. Sentence-level checker; label = all sentences supported."),
    # ================================================================ small open (Colab, vLLM)
    ModelSpec("qwen3.5-4b", "Qwen3.5-4B", "small_open", "vllm",
              "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B", hf_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
              params_total_b=4.7, license="apache-2.0", quantization="bf16", family="qwen",
              notes="Newest small Qwen (Qwen3.8 has nothing below 27B): one generation older than the rest of the Qwen curve."),
    ModelSpec("gemma-4-e4b", "Gemma 4 E4B-it", "small_open", "vllm",
              "google/gemma-4-E4B-it", "google/gemma-4-E4B-it", hf_revision="4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
              params_total_b=8.0, params_active_b=4.0, license="apache-2.0", quantization="bf16", family="gemma4",
              notes="8.0B total incl. per-layer embeddings; ~4B 'effective'."),
    # ================================================================ mid open
    ModelSpec("gpt-oss-20b", "gpt-oss-20b", "mid_open", "vllm",
              "openai/gpt-oss-20b", "openai/gpt-oss-20b", hf_revision="6cee5e81ee83917806bbde320786a8fb61efebee",
              params_total_b=20.9, params_active_b=3.6, license="apache-2.0", quantization="mxfp4 (native)",
              family="gpt-oss"),
    ModelSpec("qwen3.8-27b", "Qwen3.8-27B", "mid_open", "vllm",
              "Qwen/Qwen3.8-27B-FP8", "Qwen/Qwen3.8-27B", hf_revision="c3a462cc3d38cf351f3b26cbd763469eeb74ef81",
              params_total_b=27.8, license="apache-2.0", quantization="fp8 (official build)", family="qwen",
              notes="Official FP8 build so it fits an A100 40GB; hf_revision pins the FP8 repo."),
    ModelSpec("muse-glimmer-30b", "Muse Glimmer 30B", "mid_open", "openrouter",
              "meta/muse-glimmer-30b", "meta-models/Muse-Glimmer-30B",
              params_total_b=29.8, price_in=0.30, price_out=1.20, license="apache-2.0",
              provider="deepinfra/bf16", quantization="bf16", family="muse", enabled=False,
              notes="Optional. Run via API in bf16 rather than 4-bit on Colab."),
    ModelSpec("gemma-4-31b", "Gemma 4 31B-it", "mid_open", "openrouter",
              "google/gemma-4-31b-it", "google/gemma-4-31B-it",
              params_total_b=31.3, price_in=0.14, price_out=0.40, license="apache-2.0",
              provider="novita/bf16", quantization="bf16", family="gemma4", enabled=False,
              notes="Optional. Run via API in bf16 rather than 4-bit on Colab."),
    # ================================================================ large open (OpenRouter)
    ModelSpec("glm-5.3", "GLM-5.3", "large_open", "openrouter",
              "z-ai/glm-5.3", "zai-org/GLM-5.3",
              params_total_b=753, price_in=1.40, price_out=4.40, license="other",
              provider="z-ai/fp8", quantization="fp8", family="glm"),
    ModelSpec("deepseek-v4.1-flash", "DeepSeek V4.1 Flash", "large_open", "openrouter",
              "deepseek/deepseek-v4.1-flash", "deepseek-ai/DeepSeek-V4.1-Flash",
              params_total_b=763, price_in=0.14, price_out=0.42, license="MIT",
              provider="deepinfra/fp8", quantization="fp8", family="deepseek",
              notes="No first-party DeepSeek endpoint on OpenRouter; fp8 is the native release precision."),
    ModelSpec("muse-spark-open", "Muse Spark Open", "large_open", "openrouter",
              "meta/muse-spark-open", "meta-models/Muse-Spark-Open-816B-A42B",
              params_total_b=816, params_active_b=42, price_in=1.25, price_out=4.25, license="other",
              provider="together", quantization="unknown", family="muse",
              notes="HF weights are gated; only 5 endpoints, all at the same price."),
    ModelSpec("qwen3.8-2.4t", "Qwen3.8 2.4T-A95B", "large_open", "openrouter",
              "qwen/qwen3.8-2.4t-a95b", "Qwen/Qwen3.8-2.4T-A95B",
              params_total_b=2400, params_active_b=95, price_in=2.00, price_out=6.00, license="other",
              provider="alibaba", quantization="unknown", family="qwen", enabled=False,
              notes="Optional open Qwen flagship: top of the Qwen size curve."),
    ModelSpec("kimi-k3.1-pro", "Kimi K3.1 Pro", "large_open", "openrouter",
              "moonshotai/kimi-k3.1-pro", "moonshotai/Kimi-K3.1-Pro",
              params_total_b=2780, price_in=2.00, price_out=10.00, license="other",
              provider="moonshotai/mxfp4", quantization="mxfp4", family="kimi",
              notes="High output price: cost is dominated by reasoning/output length."),
    # ================================================================ closed reference
    ModelSpec("qwen3.8-max", "Qwen3.8 Max", "closed_reference", "openrouter",
              "qwen/qwen3.8-max-0902", None,
              price_in=2.00, price_out=6.00, license="proprietary",
              provider="alibaba", quantization="unknown", family="qwen",
              notes="Closed weights (API only). Never counted as open in charts."),
    # ================================================================ frontier (core: Opus 5.5 flagship + 2 cheap)
    ModelSpec("claude-opus-5.5", "Claude Opus 5.5", "frontier", "openrouter",
              "anthropic/claude-opus-5.5", None, price_in=4.00, price_out=20.00, license="proprietary",
              provider="anthropic", temperature_supported=False,
              notes="Anthropic flagship; cascade target."),
    ModelSpec("claude-haiku-5.5", "Claude Haiku 5.5", "frontier", "openrouter",
              "anthropic/claude-haiku-5.5", None, price_in=0.10, price_out=0.50, license="proprietary",
              provider="anthropic", temperature_supported=False, enabled=False,
              notes="Optional: only one Claude model in the core run (Opus 5.5, the cascade target)."),
    ModelSpec("gpt-6-luna", "GPT-6 Luna", "frontier", "openrouter",
              "openai/gpt-6-luna", None, price_in=0.10, price_out=0.50, license="proprietary",
              provider="openai", temperature_supported=False,
              notes="OpenAI's cost-efficient tier (below GPT-6 Sol). `openai/flex` endpoint is half price."),
    ModelSpec("gemini-3.8-flash", "Gemini 3.8 Flash", "frontier", "openrouter",
              "google/gemini-3.8-flash", None, price_in=0.75, price_out=3.75, license="proprietary",
              provider="google-ai-studio",
              notes="AI Studio endpoint (accepts temperature; the Vertex endpoint does not)."),
    # ---------------------------------------------------------------- frontier, optional flagships
    ModelSpec("claude-fable-5.2", "Claude Fable 5.2", "frontier", "openrouter",
              "anthropic/claude-fable-5.2", None, price_in=8.00, price_out=40.00, license="proprietary",
              provider="anthropic", temperature_supported=False, enabled=False,
              notes="Anthropic's most capable model; optional because of price."),
    ModelSpec("gpt-6.2-astra", "GPT-6.2 Astra", "frontier", "openrouter",
              "openai/gpt-6.2-astra", None, price_in=8.00, price_out=40.00, license="proprietary",
              provider="openai", temperature_supported=False, enabled=False,
              notes="OpenAI flagship; optional because of price."),
    ModelSpec("gemini-4-argon", "Gemini 4 Argon", "frontier", "openrouter",
              "google/gemini-4-argon", None, price_in=2.00, price_out=10.00, license="proprietary",
              provider="google-ai-studio", enabled=False,
              notes="Google flagship (Gemini 4 Neon is a mid-priced model, not the flagship)."),
]

REGISTRY: dict[str, ModelSpec] = {m.key: m for m in MODELS}


def get(key: str) -> ModelSpec:
    return REGISTRY[key]


def enabled(tier: str | None = None, backend: str | None = None) -> list[ModelSpec]:
    return [m for m in MODELS if m.enabled and (tier is None or m.tier == tier)
            and (backend is None or m.backend == backend)]


def estimate_run_cost(spec: ModelSpec, n_calls: int, avg_in: float, avg_out: float) -> float:
    """Rough budget: calls x average tokens x price. Replaced by measured pilot tokens in Step 6."""
    return spec.cost_usd(n_calls * avg_in, n_calls * avg_out)


def roster_table():
    """The roster as a DataFrame (for notebooks and the plan)."""
    import pandas as pd
    return pd.DataFrame([{
        "key": m.key, "model": m.display, "tier": m.tier, "status": "core" if m.enabled else "optional",
        "params_B": m.params_total_b, "active_B": m.params_active_b, "open": m.open_weights,
        "backend": m.backend, "pin": m.provider or (m.hf_revision[:10] if m.hf_revision else None),
        "quant": m.quantization, "$in/M": m.price_in, "$out/M": m.price_out, "license": m.license,
    } for m in MODELS])
