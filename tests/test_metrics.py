"""
Unit tests for cigci_eval.metrics.
Guarantees that metrics compute true values and respond dynamically to changes in input.
"""

import pytest
import numpy as np
from cigci_eval.records import PredictionRecord
from cigci_eval.metrics import (
    accuracy,
    macro_f1,
    weighted_f1,
    brier_score,
    negative_log_likelihood,
    ece,
    mce,
    risk_coverage_curve,
    select_threshold_for_coverage,
    risk_at_threshold,
    pope,
    bootstrap_ci,
    aggregate_over_seeds,
    mcnemar,
    paired_bootstrap_diff,
    holm_bonferroni,
    relative_reduction,
)

def make_rec(idx: int, gold: str, pred: str, conf: float, prob_vec=None, ans_space=None) -> PredictionRecord:
    return PredictionRecord(
        sample_id=f"s_{idx}",
        dataset="slake",
        split="test",
        question="dummy question?",
        gold=gold,
        pred=pred,
        confidence=conf,
        prob_vector=prob_vec,
        answer_space=ans_space,
    )

def test_accuracy():
    recs = [
        make_rec(0, "yes", "yes", 0.9),
        make_rec(1, "no", "no", 0.8),
        make_rec(2, "yes", "no", 0.7),
        make_rec(3, "no", "yes", 0.6),
    ]
    assert accuracy(recs) == 0.5

def test_macro_f1():
    recs = [
        make_rec(0, "yes", "yes", 0.9),
        make_rec(1, "no", "no", 0.8),
        make_rec(2, "yes", "yes", 0.7),
        make_rec(3, "no", "no", 0.6),
    ]
    assert macro_f1(recs) == 1.0

def test_weighted_f1():
    recs = [
        make_rec(0, "yes", "yes", 0.9),
        make_rec(1, "yes", "yes", 0.8),
        make_rec(2, "no", "no", 0.7),
    ]
    assert weighted_f1(recs) == 1.0

def test_brier_score():
    ans_space = ["yes", "no"]
    recs = [
        make_rec(0, "yes", "yes", 0.8, [0.8, 0.2], ans_space),
        make_rec(1, "no", "no", 0.9, [0.1, 0.9], ans_space),
    ]
    # (0.8-1)^2 + (0.2-0)^2 = 0.04 + 0.04 = 0.08
    # (0.1-0)^2 + (0.9-1)^2 = 0.01 + 0.01 = 0.02
    # mean = 0.05
    assert np.isclose(brier_score(recs), 0.05)

def test_nll():
    ans_space = ["yes", "no"]
    recs = [
        make_rec(0, "yes", "yes", 0.5, [0.5, 0.5], ans_space),
    ]
    assert np.isclose(negative_log_likelihood(recs), -np.log(0.5))

def test_ece_responds_to_input():
    # 1. Perfectly calibrated dataset: 100 samples with 0.80 conf and exactly 80% accuracy
    perfect_recs = [
        make_rec(i, "yes" if i < 80 else "no", "yes", 0.80) for i in range(100)
    ]
    ece_perfect = ece(perfect_recs, n_bins=5)
    assert ece_perfect < 0.01, f"Expected near-zero ECE, got {ece_perfect}"

    # 2. Severely overconfident miscalibrated dataset: 100 samples with 0.99 conf and 0% accuracy
    terrible_recs = [
        make_rec(i, "no", "yes", 0.99) for i in range(100)
    ]
    ece_terrible = ece(terrible_recs, n_bins=5)
    assert ece_terrible > 0.90, f"Expected high ECE, got {ece_terrible}"

def test_mce():
    recs = [
        make_rec(0, "no", "yes", 0.99), # bin [0.9, 1.0] acc=0, conf=0.99 -> error=0.99
        make_rec(1, "yes", "yes", 0.1), # bin [0.1, 0.2] acc=1, conf=0.1 -> error=0.9
    ]
    assert mce(recs, n_bins=10) >= 0.9

def test_risk_coverage_curve():
    recs = [
        make_rec(0, "yes", "yes", 0.95),
        make_rec(1, "yes", "yes", 0.85),
        make_rec(2, "yes", "no", 0.70), # error
        make_rec(3, "no", "yes", 0.50), # error
    ]
    pts = risk_coverage_curve(recs)
    assert len(pts) == 4
    assert pts[0].coverage == 0.25
    assert pts[0].risk == 0.0
    assert pts[1].coverage == 0.50
    assert pts[1].risk == 0.0
    assert pts[2].coverage == 0.75
    assert np.isclose(pts[2].risk, 1/3)
    assert pts[3].coverage == 1.00
    assert pts[3].risk == 0.50

def test_select_threshold_for_coverage():
    val_recs = [
        make_rec(i, "yes", "yes", conf)
        for i, conf in enumerate([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    ]
    # For 50% coverage out of 10 items, k=5 -> sorted index 4 -> conf 0.5
    tau = select_threshold_for_coverage(val_recs, target_coverage=0.5)
    assert tau == 0.5

def test_risk_at_threshold():
    test_recs = [
        make_rec(0, "yes", "yes", 0.9),
        make_rec(1, "yes", "yes", 0.8),
        make_rec(2, "no", "yes", 0.7),
        make_rec(3, "no", "yes", 0.4),
    ]
    # At tau=0.75, covered are sample 0 and 1 (both correct)
    risk, cov, n_cov = risk_at_threshold(test_recs, tau=0.75)
    assert risk == 0.0
    assert cov == 0.5
    assert n_cov == 2

def test_pope_basic():
    # 5 samples where finding is present (gold=yes)
    # 5 samples where finding is absent (gold=no)
    recs = [
        make_rec(0, "yes", "yes", 0.9),
        make_rec(1, "yes", "yes", 0.8),
        make_rec(2, "yes", "yes", 0.7),
        make_rec(3, "yes", "no", 0.6),
        make_rec(4, "yes", "no", 0.5),
        # Absent findings:
        make_rec(5, "no", "no", 0.8), # correct rejection
        make_rec(6, "no", "no", 0.8), # correct rejection
        make_rec(7, "no", "no", 0.8), # correct rejection
        make_rec(8, "no", "yes", 0.7), # hallucination!
        make_rec(9, "no", "yes", 0.6), # hallucination!
    ]
    res = pope(recs)
    assert res.n == 10
    # Absent count is 5, hallucinations count is 2 -> hallucination_rate = 2/5 = 0.40
    assert np.isclose(res.hallucination_rate, 0.40)
    # Predicted yes count: 3 (from gold yes) + 2 (from gold no) = 5 -> yes_ratio = 5/10 = 0.50
    assert np.isclose(res.yes_ratio, 0.50)

def test_pope_raises_on_empty_or_no_absent():
    with pytest.raises(ValueError):
        pope([])
    only_present = [make_rec(0, "yes", "yes", 0.9)]
    with pytest.raises(ValueError):
        pope(only_present)

def test_bootstrap_ci():
    recs = [
        make_rec(i, "yes" if i < 70 else "no", "yes", 0.8) for i in range(100)
    ]
    point, lo, hi = bootstrap_ci(recs, stat=accuracy, n_resamples=500, seed=42)
    assert point == 0.70
    assert 0.55 < lo < 0.70
    assert 0.70 < hi < 0.85

def test_aggregate_over_seeds():
    with pytest.raises(ValueError):
        aggregate_over_seeds([0.85])
    res = aggregate_over_seeds([0.80, 0.82, 0.84])
    assert np.isclose(res["mean"], 0.82)
    assert res["n_seeds"] == 3
    assert "std" in res

def test_mcnemar():
    # 4 samples:
    # Model A: correct, correct, incorrect, incorrect
    # Model B: correct, incorrect, correct, incorrect
    a = [
        make_rec(0, "yes", "yes", 0.9),
        make_rec(1, "yes", "yes", 0.9),
        make_rec(2, "yes", "no", 0.9),
        make_rec(3, "yes", "no", 0.9),
    ]
    b = [
        make_rec(0, "yes", "yes", 0.9),
        make_rec(1, "yes", "no", 0.9),
        make_rec(2, "yes", "yes", 0.9),
        make_rec(3, "yes", "no", 0.9),
    ]
    p_val, n01, n10 = mcnemar(a, b)
    assert n01 == 1 # a wrong, b right (idx 2)
    assert n10 == 1 # a right, b wrong (idx 1)
    assert p_val == 1.0

def test_paired_bootstrap_diff():
    a = [make_rec(i, "yes", "yes", 0.9) for i in range(50)]
    b = [make_rec(i, "yes" if i < 25 else "no", "yes", 0.9) for i in range(50)]
    point, lo, hi = paired_bootstrap_diff(a, b, stat=accuracy, n_resamples=300, seed=42)
    assert np.isclose(point, 0.50)
    assert lo > 0.30
    assert hi < 0.70

def test_relative_reduction():
    assert np.isclose(relative_reduction(0.02, 0.10), 0.80)
    with pytest.raises(ZeroDivisionError):
        relative_reduction(0.05, 0.0)

if __name__ == "__main__":
    tests = [
        test_accuracy,
        test_macro_f1,
        test_weighted_f1,
        test_brier_score,
        test_nll,
        test_ece_responds_to_input,
        test_mce,
        test_risk_coverage_curve,
        test_select_threshold_for_coverage,
        test_risk_at_threshold,
        test_pope_basic,
        test_pope_raises_on_empty_or_no_absent,
        test_bootstrap_ci,
        test_aggregate_over_seeds,
        test_mcnemar,
        test_paired_bootstrap_diff,
        test_relative_reduction,
    ]
    print(f"Running {len(tests)} unit tests on cigci_eval.metrics...")
    for t in tests:
        t()
        print(f"  PASS: {t.__name__}")
    print(f"\nALL {len(tests)} UNIT TESTS PASSED PERFECTLY!")
