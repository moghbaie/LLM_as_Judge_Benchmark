"""Model registry tests (offline)."""
import pytest

from ljbench import models
from ljbench.models import BACKENDS, MODELS, REGISTRY, TIERS, ModelSpec


def test_keys_and_ids_unique():
    assert len({m.key for m in MODELS}) == len(MODELS)
    assert len({m.model_id for m in MODELS}) == len(MODELS)


def test_valid_tier_and_backend():
    for m in MODELS:
        assert m.tier in TIERS, m.key
        assert m.backend in BACKENDS, m.key


def test_no_floating_aliases():
    # '~vendor/...-latest' aliases can silently change the model mid-run
    for m in MODELS:
        assert not m.model_id.startswith("~"), m.key
        assert "latest" not in m.model_id, m.key


def test_open_tiers_have_weights_and_closed_do_not():
    for m in MODELS:
        if m.tier.endswith("_open"):
            assert m.open_weights, f"{m.key} is in an open tier but has no hf_id"
        if m.tier in ("closed_reference", "frontier"):
            assert not m.open_weights, f"{m.key} has open weights but sits in a closed tier"


def test_api_models_have_prices():
    for m in MODELS:
        if m.backend in ("openrouter", "vendor_api"):
            assert m.price_in is not None and m.price_out is not None, m.key


def test_requested_models_registered_and_enabled():
    for key in ["kimi-k3.1-pro", "deepseek-v4.1-flash", "glm-5.3", "qwen3.8-max", "muse-spark-open"]:
        assert REGISTRY[key].enabled
    assert REGISTRY["qwen3.8-max"].tier == "closed_reference"
    assert not REGISTRY["qwen3.8-2.4t"].enabled   # optional, off by default


def test_cost_math():
    m = ModelSpec("x", "X", "large_open", "openrouter", "v/x", "v/X", price_in=1.0, price_out=4.0)
    assert m.cost_usd(1_000_000, 0) == pytest.approx(1.0)
    assert m.cost_usd(500_000, 250_000) == pytest.approx(0.5 + 1.0)
    assert models.estimate_run_cost(m, 1000, 1500, 500) == pytest.approx(1.5 + 2.0)


def test_cost_without_price_raises():
    m = ModelSpec("g", "G", "small_open", "vllm", "Qwen/x", "Qwen/x")
    with pytest.raises(ValueError):
        m.cost_usd(1, 1)
