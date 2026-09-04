"""
Unit tests for cigci_eval.fidelity.
Verifies background preservation, target attenuation, TOST equivalence testing, and realism discriminator AUC.
"""

import numpy as np
import pytest
from cigci_eval.fidelity import (
    compute_background_preservation,
    compute_target_attenuation,
    tost_equivalence_test,
    compute_realism_discriminator,
)

def test_compute_background_preservation():
    # Identity background
    orig = np.ones((64, 64, 3), dtype=float) * 0.5
    cf = np.ones((64, 64, 3), dtype=float) * 0.5
    # Mask center 16x16 with 1.0 (lesion)
    mask = np.zeros((64, 64), dtype=float)
    mask[24:40, 24:40] = 1.0
    # Inpainted center changed
    cf[24:40, 24:40] = 0.2

    res = compute_background_preservation(orig, cf, mask)
    assert res["is_sanity_check"] is True
    # Background outside mask is exact identity -> MSE == 0, PSNR == inf or very high
    assert res["mse_bg"] == 0.0
    assert res["psnr_db"] > 100.0 or np.isinf(res["psnr_db"])
    assert res["ssim_bg"] > 0.90

def test_compute_target_attenuation():
    # 10 samples
    # Original probabilities for target class (col 0): ~0.85
    # Counterfactual probabilities for target class: ~0.15
    orig_probs = np.zeros((10, 3))
    orig_probs[:, 0] = 0.85
    orig_probs[:, 1:] = 0.075

    cf_probs = np.zeros((10, 3))
    cf_probs[:, 0] = 0.15
    cf_probs[:, 1:] = 0.425

    res = compute_target_attenuation(orig_probs, cf_probs, target_class_idx=0)
    assert np.isclose(res["orig_prob_mean"], 0.85)
    assert np.isclose(res["cf_prob_mean"], 0.15)
    assert np.isclose(res["attenuation_drop_mean"], 0.70)

def test_tost_equivalence():
    # Differences tightly centered at zero with tiny variance -> equivalent within 0.05
    diffs = np.random.normal(loc=0.001, scale=0.005, size=50)
    res = tost_equivalence_test(diffs, margin=0.05, alpha=0.05)
    assert res["is_equivalent"] is True
    assert res["tost_p_value"] < 0.05

    # Differences centered at 0.10 -> NOT equivalent within margin 0.05
    bad_diffs = np.random.normal(loc=0.10, scale=0.01, size=50)
    bad_res = tost_equivalence_test(bad_diffs, margin=0.05, alpha=0.05)
    assert bad_res["is_equivalent"] is False

def test_realism_discriminator():
    # Real and synthetic features from same distribution -> AUC ~ 0.50
    rng = np.random.default_rng(42)
    real_feats = rng.normal(loc=0.0, scale=1.0, size=(40, 16))
    syn_feats = rng.normal(loc=0.0, scale=1.0, size=(40, 16))

    res = compute_realism_discriminator(real_feats, syn_feats, cv=3, seed=42)
    assert 0.35 <= res["discriminator_auc_mean"] <= 0.65
    assert res["ideal_target_auc"] == 0.50
