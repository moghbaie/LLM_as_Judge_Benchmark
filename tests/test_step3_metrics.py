"""Step 3 tests: metrics against hand-computed values, sklearn and synthetic data."""
import numpy as np
import pytest

from ljbench import metrics as M

S, U = "SUPPORTED", "UNSUPPORTED"


def test_balanced_accuracy_hand_computed():
    # UNSUPPORTED: 3/4 right, SUPPORTED: 1/2 right -> (0.75 + 0.5) / 2
    t = [U, U, U, U, S, S]
    p = [U, U, U, S, S, U]
    assert M.balanced_accuracy(t, p) == pytest.approx(0.625)
    assert M.accuracy(t, p) == pytest.approx(4 / 6)


def test_parse_failures_count_as_wrong():
    t = [U, U, S, S]
    assert M.balanced_accuracy(t, [U, None, S, None]) == pytest.approx(0.5)
    assert M.parse_failure_rate([U, None, S, None]) == 0.5
    c = M.confusion(t, [U, None, S, None], U)
    assert c == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}


def test_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    t = rng.choice([S, U], 300).tolist()
    p = rng.choice([S, U], 300).tolist()
    assert M.balanced_accuracy(t, p) == pytest.approx(sk.balanced_accuracy_score(t, p))
    assert M.cohen_kappa(t, p) == pytest.approx(sk.cohen_kappa_score(t, p))
    assert M.recall(t, p, U) == pytest.approx(sk.recall_score(t, p, pos_label=U))


def test_kappa_edge_cases():
    assert M.cohen_kappa([S, U, S, U], [S, U, S, U]) == pytest.approx(1.0)
    assert M.cohen_kappa([S, U, S, U], [S, S, S, S]) == pytest.approx(0.0)


def test_precision_at_prevalence():
    # TPR = 0.8, FPR = 0.1  -> at 10%: 0.08 / (0.08 + 0.09)
    t = [U] * 10 + [S] * 10
    p = [U] * 8 + [S] * 2 + [U] * 1 + [S] * 9
    assert M.precision_at_prevalence(t, p, U, 0.10) == pytest.approx(0.08 / 0.17)
    assert M.precision_at_prevalence(t, p, U, 0.50) == pytest.approx(0.8 / 0.9)   # = raw precision on 50/50


def test_pairwise_outcomes():
    gold = ["A", "A", "B", "B"]
    orig = ["A", "A", "B", None]
    swap = ["B", "A", "A", "A"]   # in swapped frame; mapped back: A, B, B, B
    r = M.pairwise_outcomes(gold, orig, swap)
    # pair1 consistent+correct, pair2 inconsistent, pair3 consistent+correct, pair4 failed
    assert r["position_consistency"] == pytest.approx(0.5)
    assert r["acc_both_orders"] == pytest.approx(0.5)
    # orig: A,A,B,None vs gold -> 3/4 right; swap mapped back: A,B,B,B vs gold -> 3/4 right
    assert r["acc_single_order"] == pytest.approx(6 / 8)


def test_always_first_judge_has_zero_consistency():
    gold = ["A", "B"] * 50
    r = M.pairwise_outcomes(gold, ["A"] * 100, ["A"] * 100)
    assert r["position_consistency"] == 0 and r["acc_both_orders"] == 0
    assert r["share_choosing_first"] == 1.0
    assert r["acc_single_order"] == pytest.approx(0.5)   # single-order accuracy hides the bias


def test_bootstrap_ci_contains_point_and_is_deterministic():
    rng = np.random.default_rng(0)
    t = rng.choice([S, U], 400)
    p = np.where(rng.random(400) < 0.8, t, np.where(t == S, U, S))
    a = M.balanced_accuracy_ci(t, p, n_boot=500, seed=3)
    b = M.balanced_accuracy_ci(t, p, n_boot=500, seed=3)
    assert a == b
    point, lo, hi = a
    assert lo < point < hi and 0.03 < hi - lo < 0.12   # ~ +-4 points at n=400


def test_bootstrap_coverage_on_synthetic_data():
    """A 95% CI should contain the true accuracy ~95% of the time (allow 88-99%)."""
    rng = np.random.default_rng(42)
    true_acc, n, sims, hits = 0.75, 300, 120, 0
    for s in range(sims):
        t = np.array([S, U] * (n // 2))
        p = np.where(rng.random(n) < true_acc, t, np.where(t == S, U, S))
        _, lo, hi = M.balanced_accuracy_ci(t, p, n_boot=300, seed=s)
        hits += lo <= true_acc <= hi
    assert 0.88 <= hits / sims <= 0.99


def test_paired_bootstrap_detects_real_difference_only():
    rng = np.random.default_rng(7)
    t = np.array([S, U] * 300)
    flip = lambda acc: np.where(rng.random(len(t)) < acc, t, np.where(t == S, U, S))
    good, bad, bad2 = flip(0.90), flip(0.75), flip(0.75)
    assert M.paired_bootstrap(t, good, bad, n_boot=500)["significant"]
    r = M.paired_bootstrap(t, bad, bad2, n_boot=500)
    assert abs(r["diff"]) < 0.06


def test_cost():
    assert M.cost_per_1000(2.5, 500) == pytest.approx(5.0)
    assert M.gpu_cost_per_1000(2, 1.5, 3000) == pytest.approx(1.0)


def test_summary_keys():
    s = M.summary([U, S, U, S], [U, S, S, S], "faithfulness", n_boot=100)
    assert {"balanced_acc", "ci_lo", "ci_hi", "kappa", "parse_fail", "halluc_recall", "precision@10%"} <= set(s)
