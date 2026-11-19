"""Model registry tests (offline)."""
import re

import pytest

from ljbench import models
from ljbench.models import BACKENDS, MODELS, REGISTRY, TIERS, ModelSpec

CORE = ["hhem-2.1-open", "minicheck-7b", "qwen3.5-4b", "gemma-4-e4b", "gpt-oss-20b", "qwen3.8-27b",
        "glm-5.3", "deepseek-v4.1-flash", "muse-spark-open", "kimi-k3.1-pro", "qwen3.8-max",
        "claude-opus-5.5", "gpt-6-luna", "gemini-3.8-flash"]
OPTIONAL = ["claude-haiku-5.5", "muse-glimmer-30b", "gemma-4-31b", "qwen3.8-2.4t", "claude-fable-5.2", "gpt-6.2-astra",
            "gemini-4-argon"]


def test_roster_is_exactly_the_agreed_one():
    assert sorted(m.key for m in MODELS if m.enabled) == sorted(CORE)
    assert sorted(m.key for m in MODELS if not m.enabled) == sorted(OPTIONAL)


def test_keys_and_ids_unique():
    assert len({m.key for m in MODELS}) == len(MODELS)
    assert len({m.model_id for m in MODELS}) == len(MODELS)


def test_valid_tier_backend_and_tasks():
    for m in MODELS:
        assert m.tier in TIERS, m.key
        assert m.backend in BACKENDS, m.key
        assert set(m.tasks) <= {"faithfulness", "pairwise"} and m.tasks, m.key
        if m.tier == "specialist":
            assert m.tasks == ("faithfulness",), m.key


def test_no_floating_aliases():
    for m in MODELS:
        assert not m.model_id.startswith("~") and "latest" not in m.model_id, m.key


def test_open_tiers_have_weights_and_closed_do_not():
    for m in MODELS:
        if m.tier.endswith("_open") or m.tier == "specialist":
            assert m.open_weights, f"{m.key} has no hf_id"
        if m.tier in ("closed_reference", "frontier"):
            assert not m.open_weights, f"{m.key} has open weights but sits in a closed tier"


def test_api_models_pinned_with_prices():
    for m in MODELS:
        if m.backend == "openrouter":
            assert m.provider, f"{m.key}: no pinned endpoint"
            assert m.price_in is not None and m.price_out is not None, m.key


def test_colab_models_pinned_to_hf_commit():
    for m in MODELS:
        if m.backend != "openrouter":
            assert m.hf_revision and re.fullmatch(r"[0-9a-f]{40}", m.hf_revision), m.key


def test_size_ordering_matches_tiers():
    for m in MODELS:
        if m.tier == "small_open":
            assert m.params_total_b <= 8.5, m.key
        if m.tier == "mid_open":
            assert 15 <= m.params_total_b <= 35, m.key
        if m.tier == "large_open":
            assert m.params_total_b >= 500, m.key


def test_cost_math():
    m = ModelSpec("x", "X", "large_open", "openrouter", "v/x", "v/X", price_in=1.0, price_out=4.0)
    assert m.cost_usd(1_000_000, 0) == pytest.approx(1.0)
    assert m.cost_usd(500_000, 250_000) == pytest.approx(0.5 + 1.0)
    assert models.estimate_run_cost(m, 1000, 1500, 500) == pytest.approx(1.5 + 2.0)


def test_cost_without_price_raises():
    with pytest.raises(ValueError):
        REGISTRY["qwen3.5-4b"].cost_usd(1, 1)


def test_roster_table():
    t = models.roster_table()
    assert len(t) == 21 and (t["status"] == "core").sum() == 14
