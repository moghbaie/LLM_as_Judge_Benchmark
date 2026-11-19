"""Metrics. A prediction of None (parse failure) always counts as wrong.

Faithfulness convention: the positive class is UNSUPPORTED (= a hallucination was caught).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

POSITIVE = {"faithfulness": "UNSUPPORTED", "pairwise": "A"}


def _arr(x) -> np.ndarray:
    return np.asarray(list(x), dtype=object)


# ------------------------------------------------------------------------ point metrics
def accuracy(y_true, y_pred) -> float:
    t, p = _arr(y_true), _arr(y_pred)
    return float(np.mean(t == p))


def balanced_accuracy(y_true, y_pred) -> float:
    """Mean per-class recall over the classes present in y_true; None predictions are wrong."""
    t, p = _arr(y_true), _arr(y_pred)
    recalls = [np.mean(p[t == c] == c) for c in sorted(set(t.tolist()))]
    return float(np.mean(recalls))


def confusion(y_true, y_pred, positive: str) -> dict:
    """tp/fn/fp/tn for the positive class. A failed parse counts as a miss (fn) on positives
    and as a false alarm (fp) on negatives, so failures can never improve a score."""
    t, p = _arr(y_true), _arr(y_pred)
    pos = t == positive
    hit = p == t
    return {"tp": int(np.sum(pos & hit)), "fn": int(np.sum(pos & ~hit)),
            "fp": int(np.sum(~pos & ~hit)), "tn": int(np.sum(~pos & hit))}


def recall(y_true, y_pred, positive: str) -> float:
    c = confusion(y_true, y_pred, positive)
    return c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")


def specificity(y_true, y_pred, positive: str) -> float:
    c = confusion(y_true, y_pred, positive)
    return c["tn"] / (c["tn"] + c["fp"]) if c["tn"] + c["fp"] else float("nan")


def precision_at_prevalence(y_true, y_pred, positive: str, prevalence: float) -> float:
    """Precision the judge would have if hallucinations made up `prevalence` of traffic.

    Our samples are 50/50, so raw precision is too optimistic. With TPR and FPR from the sample:
        precision = TPR * pi / (TPR * pi + FPR * (1 - pi))
    """
    tpr = recall(y_true, y_pred, positive)
    fpr = 1 - specificity(y_true, y_pred, positive)
    denom = tpr * prevalence + fpr * (1 - prevalence)
    return float(tpr * prevalence / denom) if denom else float("nan")


def cohen_kappa(y_true, y_pred) -> float:
    """Cohen's kappa; parse failures form their own 'INVALID' category (never agreeing)."""
    t = ["INVALID" if v is None else v for v in y_true]
    p = ["INVALID" if v is None else v for v in y_pred]
    cats = sorted(set(t) | set(p))
    n = len(t)
    po = sum(a == b for a, b in zip(t, p)) / n
    pe = sum((t.count(c) / n) * (p.count(c) / n) for c in cats)
    return float((po - pe) / (1 - pe)) if pe != 1 else 1.0


def parse_failure_rate(y_pred) -> float:
    p = list(y_pred)
    return sum(v is None for v in p) / len(p)


# ------------------------------------------------------------------------ pairwise (Task B)
FLIP = {"A": "B", "B": "A"}


def pairwise_outcomes(gold_orig: Sequence, pred_orig: Sequence, pred_swap: Sequence) -> dict:
    """Per pair: gold/pred are in the ORIGINAL order's frame for `orig`, and in the swapped frame for `swap`.

    consistent = the two verdicts pick the same underlying response.
    correct_both = consistent AND that response is the gold one (the plan's headline Task B metric).
    """
    g = list(gold_orig)
    po = list(pred_orig)
    ps = [None if v is None else FLIP[v] for v in pred_swap]   # map back to original frame
    consistent = np.array([a is not None and a == b for a, b in zip(po, ps)])
    correct_both = np.array([c and a == gg for c, a, gg in zip(consistent, po, g)])
    first_pos = np.array([a == "A" for a in po] + [b == "A" for b in pred_swap])
    return {
        "n_pairs": len(g),
        "acc_both_orders": float(correct_both.mean()),
        "acc_single_order": float(np.mean([a == gg for a, gg in zip(po, g)] + [b == gg for b, gg in zip(ps, g)])),
        "position_consistency": float(consistent.mean()),
        "share_choosing_first": float(first_pos.mean()),   # 0.5 = no position bias
    }


# ------------------------------------------------------------------------ uncertainty
def bootstrap_ci(stat: Callable[[np.ndarray], float], n: int, n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap over items. `stat(idx)` computes the metric on rows idx.

    Returns (point estimate, lower, upper)."""
    rng = np.random.default_rng(seed)
    point = stat(np.arange(n))
    boots = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    boots = boots[~np.isnan(boots)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def balanced_accuracy_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 0):
    t, p = _arr(y_true), _arr(y_pred)
    return bootstrap_ci(lambda i: balanced_accuracy(t[i], p[i]), len(t), n_boot, seed)


def paired_bootstrap(y_true, pred_a, pred_b, metric=balanced_accuracy, n_boot: int = 1000, seed: int = 0) -> dict:
    """Difference metric(a) - metric(b) on the SAME items, resampled jointly.

    `p_value` is the two-sided share of resamples whose difference crosses zero."""
    t, a, b = _arr(y_true), _arr(pred_a), _arr(pred_b)
    rng = np.random.default_rng(seed)
    n = len(t)
    diff = metric(t, a) - metric(t, b)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, n, n)
        boots[k] = metric(t[i], a[i]) - metric(t[i], b[i])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return {"diff": float(diff), "lo": float(lo), "hi": float(hi), "p_value": float(min(1.0, p)),
            "significant": bool(lo > 0 or hi < 0)}


# ------------------------------------------------------------------------ cost / speed
def cost_per_1000(total_cost_usd: float, n_judgements: int) -> float:
    return 1000 * total_cost_usd / n_judgements


def gpu_cost_per_1000(gpu_hours: float, usd_per_gpu_hour: float, n_judgements: int) -> float:
    return cost_per_1000(gpu_hours * usd_per_gpu_hour, n_judgements)


def summary(y_true, y_pred, task: str, n_boot: int = 1000, seed: int = 0) -> dict:
    """All headline numbers for one model x method x dataset cell."""
    pos = POSITIVE[task]
    ba, lo, hi = balanced_accuracy_ci(y_true, y_pred, n_boot, seed)
    out = {"n": len(list(y_true)), "balanced_acc": ba, "ci_lo": lo, "ci_hi": hi,
           "kappa": cohen_kappa(y_true, y_pred), "parse_fail": parse_failure_rate(y_pred)}
    if task == "faithfulness":
        out.update({"halluc_recall": recall(y_true, y_pred, pos),
                    "precision@10%": precision_at_prevalence(y_true, y_pred, pos, 0.10),
                    "precision@20%": precision_at_prevalence(y_true, y_pred, pos, 0.20)})
    return out
