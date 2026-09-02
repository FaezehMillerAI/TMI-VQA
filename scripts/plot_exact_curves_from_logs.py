import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Workspace setup
workspace_dir = "/Users/faezeh/Desktop/PhD August/CQC2"
sys.path.insert(0, workspace_dir)
os.chdir(workspace_dir)

# Set IEEE TMI styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 14

def plot_exact_curves_for_dataset(dataset_name="slake", log_dir="outputs/logs", out_dir="outputs/paper_figures"):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"per_sample_predictions_{dataset_name}.csv")

    if not os.path.exists(csv_path):
        print(f"Warning: Log file not found at {csv_path}. Please run export_detailed_predictions.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} per-sample predictions from {csv_path}")

    # Extract vectors
    base_confs = df["baseline_confidence"].to_numpy()
    base_correct = df["baseline_correct"].to_numpy()
    cqc_confs = df["cqc_confidence"].to_numpy()
    cqc_correct = df["cqc_correct"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=300)

    # -------------------------------------------------------------
    # 1. Exact ROC Curves (Confidence vs Correctness)
    # -------------------------------------------------------------
    fpr_base, tpr_base, _ = roc_curve(base_correct, base_confs)
    roc_auc_base = auc(fpr_base, tpr_base)

    fpr_cqc, tpr_cqc, _ = roc_curve(cqc_correct, cqc_confs)
    roc_auc_cqc = auc(fpr_cqc, tpr_cqc)

    axes[0, 0].plot(fpr_base, tpr_base, color='#E74C3C', linestyle='--', linewidth=2, label=f'Baseline VLM (AUROC = {roc_auc_base:.3f})')
    axes[0, 0].plot(fpr_cqc, tpr_cqc, color='#2ECC71', linewidth=2.5, label=f'CI-GCI CQC-Net (AUROC = {roc_auc_cqc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k:', linewidth=1.2)
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].set_ylim([0, 1.02])
    axes[0, 0].set_xlabel("False Positive Rate (FPR)")
    axes[0, 0].set_ylabel("True Positive Rate (TPR)")
    axes[0, 0].set_title(f"(a) Empirical ROC Curve ({dataset_name.upper()})")
    axes[0, 0].legend(loc="lower right", frameon=True)
    axes[0, 0].grid(True, linestyle=':', alpha=0.5)

    # -------------------------------------------------------------
    # 2. Exact Precision-Recall Curves
    # -------------------------------------------------------------
    prec_base, rec_base, _ = precision_recall_curve(base_correct, base_confs)
    ap_base = average_precision_score(base_correct, base_confs)

    prec_cqc, rec_cqc, _ = precision_recall_curve(cqc_correct, cqc_confs)
    ap_cqc = average_precision_score(cqc_correct, cqc_confs)

    axes[0, 1].plot(rec_base, prec_base, color='#E74C3C', linestyle='--', linewidth=2, label=f'Baseline VLM (AUPRC = {ap_base:.3f})')
    axes[0, 1].plot(rec_cqc, prec_cqc, color='#2ECC71', linewidth=2.5, label=f'CI-GCI CQC-Net (AUPRC = {ap_cqc:.3f})')
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].set_ylim([0, 1.02])
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title(f"(b) Precision-Recall Curve ({dataset_name.upper()})")
    axes[0, 1].legend(loc="lower left", frameon=True)
    axes[0, 1].grid(True, linestyle=':', alpha=0.5)

    # -------------------------------------------------------------
    # 3. Exact Risk-Coverage Trade-off Curve (Selective Abstention)
    # -------------------------------------------------------------
    thresholds = np.linspace(0.0, 1.0, 100)
    base_coverages, base_risks = [], []
    cqc_coverages, cqc_risks = [], []

    for tau in thresholds:
        # Baseline
        accepted_base = base_confs >= tau
        cov_b = np.mean(accepted_base)
        risk_b = 1.0 - np.mean(base_correct[accepted_base]) if np.sum(accepted_base) > 0 else 0.0
        base_coverages.append(cov_b)
        base_risks.append(risk_b)

        # CQC-Net
        accepted_cqc = cqc_confs >= tau
        cov_c = np.mean(accepted_cqc)
        risk_c = 1.0 - np.mean(cqc_correct[accepted_cqc]) if np.sum(accepted_cqc) > 0 else 0.0
        cqc_coverages.append(cov_c)
        cqc_risks.append(risk_c)

    axes[1, 0].plot(np.array(base_coverages) * 100, np.array(base_risks) * 100, color='#E74C3C', linestyle='--', linewidth=2, label='Baseline VLM')
    axes[1, 0].plot(np.array(cqc_coverages) * 100, np.array(cqc_risks) * 100, color='#2ECC71', linewidth=2.5, label='CI-GCI CQC-Net')
    axes[1, 0].set_xlabel("Coverage Percentage (%)")
    axes[1, 0].set_ylabel("Clinical Error / Risk (%)")
    axes[1, 0].set_title(f"(c) Selective Abstention Risk-Coverage")
    axes[1, 0].legend(loc="upper left", frameon=True)
    axes[1, 0].grid(True, linestyle=':', alpha=0.5)

    # -------------------------------------------------------------
    # 4. Exact 10-Bin Reliability Calibration Histogram (ECE)
    # -------------------------------------------------------------
    num_bins = 10
    bins = np.linspace(0, 1, num_bins + 1)
    cal_accs, cal_confs_bin = [], []

    for i in range(num_bins):
        in_bin = (cqc_confs >= bins[i]) & (cqc_confs < bins[i+1])
        if i == num_bins - 1:
            in_bin = in_bin | (cqc_confs == bins[i+1])
        if np.sum(in_bin) > 0:
            cal_accs.append(np.mean(cqc_correct[in_bin]))
            cal_confs_bin.append(np.mean(cqc_confs[in_bin]))
        else:
            cal_accs.append(0.0)
            cal_confs_bin.append((bins[i] + bins[i+1]) / 2.0)

    # Calculate exact empirical ECE
    ece = 0.0
    for i in range(num_bins):
        in_bin = (cqc_confs >= bins[i]) & (cqc_confs < bins[i+1])
        if i == num_bins - 1:
            in_bin = in_bin | (cqc_confs == bins[i+1])
        weight = np.sum(in_bin) / len(cqc_confs)
        if np.sum(in_bin) > 0:
            ece += weight * np.abs(np.mean(cqc_correct[in_bin]) - np.mean(cqc_confs[in_bin]))

    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    axes[1, 1].bar(bin_centers, cal_accs, width=0.08, color='#2ECC71', alpha=0.8, edgecolor='#1E8449', label=f'CI-GCI Output (ECE = {ece:.4f})')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1.05])
    axes[1, 1].set_xlabel("Confidence $\hat{P}$")
    axes[1, 1].set_ylabel("Empirical Accuracy")
    axes[1, 1].set_title(f"(d) Reliability Calibration Diagram")
    axes[1, 1].legend(loc="upper left", frameon=True)
    axes[1, 1].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"exact_curves_{dataset_name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(f"Manuscript TMI/exact_curves_{dataset_name}.png", bbox_inches='tight')
    plt.close()
    print(f"-> Successfully plotted exact empirical curves to:\n   {save_path}\n   Manuscript TMI/exact_curves_{dataset_name}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="slake", choices=["slake", "vqa_rad", "pathvqa", "kvasir"])
    args = parser.parse_args()
    plot_exact_curves_for_dataset(dataset_name=args.dataset)
