"""
Counterfactual inpainting fidelity evaluation on real image arrays and classifiers.
Computes:
1. Out-of-mask background preservation (PSNR, SSIM) as a sanity check.
2. Target pathology attenuation via independent classifier.
3. Non-target finding stability via TOST equivalence testing.
4. Patch realism via real-vs-synthetic discriminator AUC (appropriate for n < 2048).
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from skimage.metrics import structural_similarity


def compute_background_preservation(
    orig_img: np.ndarray,
    cf_img: np.ndarray,
    mask: np.ndarray,
    mask_threshold: float = 0.05
) -> dict[str, float]:
    """
    Evaluates background preservation outside the ROI mask (where mask <= threshold).
    Images are assumed in [0, 1] range with shape (H, W, C) or (H, W).
    """
    orig = np.asarray(orig_img, dtype=float)
    cf = np.asarray(cf_img, dtype=float)
    m = np.asarray(mask, dtype=float)
    if m.ndim == 2 and orig.ndim == 3:
        m_2d = m
        bg_mask = np.broadcast_to(m_2d[..., None] <= mask_threshold, orig.shape)
    else:
        bg_mask = (m <= mask_threshold)

    if not np.any(bg_mask):
        raise ValueError("no background pixels outside mask threshold")

    # Outside-mask difference
    diff = (orig - cf) * bg_mask
    n_bg_pixels = np.sum(bg_mask)
    mse = float(np.sum(diff ** 2) / (n_bg_pixels + 1e-12))
    psnr = float(10.0 * np.log10(1.0 / (mse + 1e-12))) if mse > 0 else float("inf")

    # Outside-mask SSIM
    _, ssim_map = structural_similarity(
        orig, cf, data_range=1.0, channel_axis=-1 if orig.ndim == 3 else None, full=True
    )
    bg_ssim = float(np.mean(ssim_map[bg_mask]))

    return {
        "psnr_db": psnr,
        "ssim_bg": bg_ssim,
        "mse_bg": mse,
        "is_sanity_check": True,
    }


def compute_target_attenuation(
    orig_probs: np.ndarray,
    cf_probs: np.ndarray,
    target_class_idx: int | Sequence[int]
) -> dict[str, float]:
    """
    Computes attenuation of target class probability under counterfactual intervention:
    P(Y_target | I) - P(Y_target | I_cf).
    """
    orig_p = np.asarray(orig_probs, dtype=float)
    cf_p = np.asarray(cf_probs, dtype=float)

    if isinstance(target_class_idx, int):
        target_indices = [target_class_idx]
    else:
        target_indices = list(target_class_idx)

    p_orig_target = orig_p[..., target_indices].sum(axis=-1)
    p_cf_target = cf_p[..., target_indices].sum(axis=-1)
    drop = p_orig_target - p_cf_target

    return {
        "orig_prob_mean": float(np.mean(p_orig_target)),
        "cf_prob_mean": float(np.mean(p_cf_target)),
        "attenuation_drop_mean": float(np.mean(drop)),
        "attenuation_drop_std": float(np.std(drop, ddof=1)) if len(drop) > 1 else 0.0,
    }


def tost_equivalence_test(
    diffs: np.ndarray,
    margin: float = 0.05,
    alpha: float = 0.05
) -> dict[str, float | bool]:
    """
    Two One-Sided Tests (TOST) for equivalence:
    H01: diff <= -margin  vs  H11: diff > -margin
    H02: diff >= margin   vs  H12: diff < margin
    Rejects non-equivalence if both one-sided p-values < alpha.
    """
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n < 3:
        raise ValueError("at least 3 samples required for TOST equivalence test")

    mean_d = float(np.mean(d))
    se = float(stats.sem(d))
    df = n - 1

    t1 = (mean_d - (-margin)) / (se + 1e-12)
    p1 = float(1.0 - stats.t.cdf(t1, df))

    t2 = (mean_d - margin) / (se + 1e-12)
    p2 = float(stats.t.cdf(t2, df))

    p_tost = max(p1, p2)
    is_equivalent = bool(p_tost < alpha)

    return {
        "mean_difference": mean_d,
        "margin": margin,
        "tost_p_value": p_tost,
        "p1_lower": p1,
        "p2_upper": p2,
        "is_equivalent": is_equivalent,
    }


def compute_realism_discriminator(
    real_features: np.ndarray,
    synthetic_features: np.ndarray,
    cv: int = 5,
    seed: int = 42
) -> dict[str, float]:
    """
    Evaluates distributional realism via a real-vs-synthetic linear discriminator AUC.
    An AUC ~ 0.50 indicates the synthetic patches are distributionally indistinguishable
    from real healthy patches. Avoids small-sample FID instability (where n < 2048).
    """
    n_real = len(real_features)
    n_syn = len(synthetic_features)
    if n_real < 10 or n_syn < 10:
        raise ValueError("at least 10 samples per class required for discriminator AUC")

    X = np.vstack([real_features, synthetic_features])
    y = np.hstack([np.ones(n_real), np.zeros(n_syn)])

    clf = LogisticRegression(random_state=seed, max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

    return {
        "discriminator_auc_mean": float(np.mean(scores)),
        "discriminator_auc_std": float(np.std(scores, ddof=1)),
        "ideal_target_auc": 0.50,
        "n_real": n_real,
        "n_synthetic": n_syn,
    }
