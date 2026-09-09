"""
Pure, deterministic metric functions for CI-GCI evaluation.
All functions take Sequence[PredictionRecord] and return computed numbers.
No side effects, no hardcoded lookups, no model dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from cigci_eval.records import PredictionRecord

RNG_SEED = 42


# ----------------------------------------------------------- core VQA metrics


def accuracy(records: Sequence[PredictionRecord]) -> float:
    """Exact match accuracy over records."""
    if not records:
        raise ValueError("empty records")
    return float(np.mean([r.correct for r in records]))


def macro_f1(records: Sequence[PredictionRecord]) -> float:
    """Macro-averaged F1 score across answer classes."""
    if not records:
        raise ValueError("empty records")
    golds = [r.gold for r in records]
    preds = [r.pred for r in records]
    return float(f1_score(golds, preds, average="macro", zero_division=0))


def weighted_f1(records: Sequence[PredictionRecord]) -> float:
    """Weighted F1 score across answer classes."""
    if not records:
        raise ValueError("empty records")
    golds = [r.gold for r in records]
    preds = [r.pred for r in records]
    return float(f1_score(golds, preds, average="weighted", zero_division=0))


# ---------------------------------------------------- calibration uncertainty


def brier_score(records: Sequence[PredictionRecord]) -> float:
    """
    Multiclass Brier score: mean squared error between full probability vector
    and one-hot ground-truth target. Requires prob_vector and answer_space.
    Falls back to binary (1 - conf)^2 for correct and conf^2 for incorrect if vector absent.
    """
    if not records:
        raise ValueError("empty records")
    scores = []
    for r in records:
        if r.prob_vector is not None and r.answer_space is not None:
            ans_map = {ans: i for i, ans in enumerate(r.answer_space)}
            gold_idx = ans_map.get(r.gold)
            vec = np.asarray(r.prob_vector, dtype=float)
            target = np.zeros_like(vec)
            if gold_idx is not None and gold_idx < len(target):
                target[gold_idx] = 1.0
            scores.append(float(np.sum((vec - target) ** 2)))
        else:
            # Top-1 fallback
            target = 1.0 if r.correct else 0.0
            scores.append(float((r.confidence - target) ** 2))
    return float(np.mean(scores))


def negative_log_likelihood(records: Sequence[PredictionRecord], eps: float = 1e-12) -> float:
    """Negative log-likelihood of ground-truth target."""
    if not records:
        raise ValueError("empty records")
    nlls = []
    for r in records:
        if r.prob_vector is not None and r.answer_space is not None:
            ans_map = {ans: i for i, ans in enumerate(r.answer_space)}
            gold_idx = ans_map.get(r.gold)
            if gold_idx is not None and gold_idx < len(r.prob_vector):
                prob = max(eps, float(r.prob_vector[gold_idx]))
            else:
                prob = eps
        else:
            prob = max(eps, r.confidence if r.correct else (1.0 - r.confidence))
        nlls.append(-np.log(prob))
    return float(np.mean(nlls))


def ece(
    records: Sequence[PredictionRecord],
    n_bins: int = 10,
    equal_frequency: bool = False
) -> float:
    """
    Expected Calibration Error across top-1 confidence predictions.
    Computes absolute difference between mean confidence and empirical accuracy in each bin.
    """
    if not records:
        raise ValueError("empty records")
    confs = np.array([r.confidence for r in records], dtype=float)
    corrects = np.array([r.correct for r in records], dtype=float)
    n = len(records)

    if equal_frequency:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(confs, quantiles)
        bin_edges[0] -= 1e-7
        bin_edges[-1] += 1e-7
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece_val = 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confs >= low) & (confs <= high)
        else:
            mask = (confs >= low) & (confs < high)
        n_bin = np.sum(mask)
        if n_bin > 0:
            bin_acc = float(np.mean(corrects[mask]))
            bin_conf = float(np.mean(confs[mask]))
            ece_val += (n_bin / n) * abs(bin_acc - bin_conf)

    return float(ece_val)


def mce(records: Sequence[PredictionRecord], n_bins: int = 10) -> float:
    """Maximum Calibration Error across confidence bins."""
    if not records:
        raise ValueError("empty records")
    confs = np.array([r.confidence for r in records], dtype=float)
    corrects = np.array([r.correct for r in records], dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    max_err = 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (confs >= low) & (confs <= high) if i == n_bins - 1 else (confs >= low) & (confs < high)
        n_bin = np.sum(mask)
        if n_bin > 0:
            bin_acc = float(np.mean(corrects[mask]))
            bin_conf = float(np.mean(confs[mask]))
            max_err = max(max_err, abs(bin_acc - bin_conf))

    return float(max_err)


# ------------------------------------------------ selective abstention / triage


@dataclass
class RiskCoveragePoint:
    threshold: float
    coverage: float
    risk: float
    n_covered: int


def risk_coverage_curve(records: Sequence[PredictionRecord]) -> list[RiskCoveragePoint]:
    """
    Computes empirical risk-coverage trade-off curve across all confidence values.
    Sorted descending by confidence.
    """
    if not records:
        raise ValueError("empty records")
    sorted_recs = sorted(records, key=lambda r: r.confidence, reverse=True)
    n_total = len(sorted_recs)

    out: list[RiskCoveragePoint] = []
    errors = 0
    for i, r in enumerate(sorted_recs, start=1):
        if not r.correct:
            errors += 1
        out.append(
            RiskCoveragePoint(
                threshold=float(r.confidence),
                coverage=float(i / n_total),
                risk=float(errors / i),
                n_covered=i,
            )
        )
    return out


def select_threshold_for_coverage(
    val_records: Sequence[PredictionRecord], target_coverage: float
) -> float:
    """
    Pick tau on VALIDATION to hit a target coverage. Freeze it, then evaluate on test.
    Selecting tau on test inflates the reported risk-coverage tradeoff and is the
    first thing a reviewer will check.
    """
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1]")
    conf = np.sort(np.array([r.confidence for r in val_records], dtype=float))[::-1]
    k = max(1, int(round(target_coverage * len(conf))))
    return float(conf[k - 1])


def risk_at_threshold(
    records: Sequence[PredictionRecord], tau: float
) -> tuple[float, float, int]:
    """Returns (risk, coverage, n_covered) on the held-out set at a frozen tau."""
    covered = [r for r in records if r.confidence >= tau]
    if not covered:
        return float("nan"), 0.0, 0
    risk = 1.0 - float(np.mean([r.correct for r in covered]))
    return risk, len(covered) / len(records), len(covered)


# ------------------------------------------------------------------ POPE probes


@dataclass
class PopeResult:
    precision: float
    recall: float
    f1: float
    accuracy: float
    yes_ratio: float
    hallucination_rate: float
    n: int


def pope(records: Sequence[PredictionRecord], yes: str = "yes") -> PopeResult:
    """
    Records are yes/no probes. gold == "no" means the finding is documented absent;
    predicting "yes" there is a hallucination.

    Positive class = "yes" (finding present), following the original POPE convention.
    """
    g = np.array([r.gold.strip().lower() == yes for r in records])
    p = np.array([r.pred.strip().lower() == yes for r in records])
    if len(g) == 0:
        raise ValueError("empty probe set")

    absent = ~g
    if absent.sum() == 0:
        raise ValueError("probe set contains no absent-finding items")

    return PopeResult(
        precision=float(precision_score(g, p, zero_division=0)),
        recall=float(recall_score(g, p, zero_division=0)),
        f1=float(f1_score(g, p, zero_division=0)),
        accuracy=float(np.mean(g == p)),
        yes_ratio=float(p.mean()),
        hallucination_rate=float(p[absent].mean()),
        n=len(g),
    )


# ----------------------------------------------------- uncertainty + comparison


def bootstrap_ci(
    records: Sequence[PredictionRecord],
    stat: Callable[[Sequence[PredictionRecord]], float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """Percentile bootstrap over samples. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(records)
    point = stat(records)
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        draws[i] = stat([records[j] for j in idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def aggregate_over_seeds(values: Sequence[float]) -> dict[str, float | int]:
    a = np.asarray(values, dtype=float)
    if a.size < 2:
        raise ValueError(
            "at least 2 seeds required; a single run cannot support a +/- claim"
        )
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)),
        "min": float(a.min()),
        "max": float(a.max()),
        "n_seeds": int(a.size),
    }


def mcnemar(
    a: Sequence[PredictionRecord], b: Sequence[PredictionRecord]
) -> tuple[float, int, int]:
    """
    Paired accuracy comparison on the same samples. Returns (p_value, n01, n10).
    Exact binomial; appropriate for the small discordant counts you will see on
    a 451-sample test split.
    """
    from scipy.stats import binomtest

    ba = {r.sample_id: r.correct for r in a}
    bb = {r.sample_id: r.correct for r in b}
    shared = sorted(set(ba) & set(bb))
    if len(shared) != len(ba) or len(shared) != len(bb):
        raise ValueError("McNemar requires identical sample sets on both sides")
    n01 = sum(1 for s in shared if not ba[s] and bb[s])
    n10 = sum(1 for s in shared if ba[s] and not bb[s])
    if n01 + n10 == 0:
        return 1.0, 0, 0
    p = binomtest(n10, n01 + n10, 0.5).pvalue
    return float(p), n01, n10


def paired_bootstrap_diff(
    a: Sequence[PredictionRecord],
    b: Sequence[PredictionRecord],
    stat: Callable[[Sequence[PredictionRecord]], float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """CI on stat(a) - stat(b) with paired resampling. Use for ECE, Brier, etc."""
    ia = {r.sample_id: r for r in a}
    ib = {r.sample_id: r for r in b}
    shared = sorted(set(ia) & set(ib))
    if not shared:
        raise ValueError("no shared sample ids")
    rng = np.random.default_rng(seed)
    point = stat([ia[s] for s in shared]) - stat([ib[s] for s in shared])
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, len(shared), size=len(shared))
        ss = [shared[j] for j in idx]
        draws[i] = stat([ia[s] for s in ss]) - stat([ib[s] for s in ss])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """
    Correct across the whole family of comparisons reported in the paper.
    Returns per-comparison adjusted p and reject decision.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[name] = {"p_raw": float(p), "p_adj": float(adj), "reject": bool(adj < alpha)}
    return out


# ------------------------------------------------------------------- derived


def relative_reduction(new: float, old: float) -> float:
    """(old - new) / old. Always name the comparator at the call site."""
    if old == 0:
        raise ZeroDivisionError("comparator is zero")
    return (old - new) / old
