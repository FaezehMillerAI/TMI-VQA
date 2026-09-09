"""
Dynamic Publication Table Generator for IEEE TMI CI-GCI Manuscript.
Ingests verified canonical evaluation outputs (outputs/canonical.json)
and per-sample inference logs to dynamically construct all paper tables.
Zero hardcoded values.
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_canonical_data(canonical_path: str = "outputs/canonical.json") -> dict:
    """Load canonical evaluation metrics aggregated over multiple random seeds."""
    p = Path(canonical_path)
    if not p.exists():
        # Check parent directory fallback
        alt = Path(__file__).resolve().parents[1] / canonical_path
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(
                f"Canonical results file '{canonical_path}' not found.\n"
                "Please run 'python scripts/build_canonical.py' or 'python run_kaggle_pipeline.py' first."
            )
    with open(p, "r") as f:
        return json.load(f)


def generate_table_1_main_results(output_dir: str, canonical_data: dict):
    """
    Table 1 (Table III in Manuscript):
    Main comparison across multi-center datasets: VQA-RAD, SLAKE, PathVQA, Kvasir-VQA-x1.
    Reports Open, Closed, and Overall Accuracies with ECE.
    """
    print("[Tables] Dynamically building Main VQA Comparison Table...")
    rows = []
    models = canonical_data.get("models", {})

    dataset_keys = [("vqa_rad", "VQA-RAD"), ("slake", "SLAKE"), ("pathvqa", "PathVQA"), ("kvasir_x1", "Kvasir-VQA-x1")]
    model_keys = [
        ("baseline_1", "Baseline-1 (ResNet + LSTM)"),
        ("baseline_2", "Baseline-2 (ViT + PubMedBERT)"),
        ("ci_gci", "Proposed CI-GCI (APD-CC)")
    ]

    for m_key, m_label in model_keys:
        row = {"Model": m_label}
        for d_key, d_label in dataset_keys:
            d_info = models.get(d_key, {}).get(m_key, {})
            if not d_info:
                row[f"{d_label} Overall Acc"] = "--"
                row[f"{d_label} Closed Acc"] = "--"
                row[f"{d_label} Open Acc"] = "--"
                row[f"{d_label} ECE ↓"] = "--"
                continue

            overall = d_info.get("overall", {})
            closed = d_info.get("closed", {})
            open_sp = d_info.get("open", {})

            if "acc" in overall:
                m_val = overall['acc']['mean'] * 100
                s_val = overall['acc']['std'] * 100
                row[f"{d_label} Overall Acc"] = f"{m_val:.2f} ± {s_val:.2f}%"
            else:
                row[f"{d_label} Overall Acc"] = "--"

            if "acc" in closed:
                m_val = closed['acc']['mean'] * 100
                s_val = closed['acc']['std'] * 100
                row[f"{d_label} Closed Acc"] = f"{m_val:.2f} ± {s_val:.2f}%"
            else:
                row[f"{d_label} Closed Acc"] = "--"

            if "acc" in open_sp:
                m_val = open_sp['acc']['mean'] * 100
                s_val = open_sp['acc']['std'] * 100
                row[f"{d_label} Open Acc"] = f"{m_val:.2f} ± {s_val:.2f}%"
            else:
                row[f"{d_label} Open Acc"] = "--"

            if "ece" in overall:
                row[f"{d_label} ECE ↓"] = f"{overall['ece']['mean']:.4f}"
            else:
                row[f"{d_label} ECE ↓"] = "--"

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_out = os.path.join(output_dir, "table_1_main_comparison.csv")
    md_out = os.path.join(output_dir, "table_1_main_comparison.md")
    df.to_csv(csv_out, index=False)

    with open(md_out, "w") as f:
        f.write("### Table 1: Main Multi-Center Benchmark Comparison (Canonical Seeds)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"-> Emitted {csv_out}")


def generate_table_2_kvasir_breakdown(output_dir: str, canonical_data: dict):
    """Table 2: Kvasir-VQA-x1 Performance Stratification."""
    print("[Tables] Dynamically building Kvasir Breakdown Table...")
    kvasir_models = canonical_data.get("models", {}).get("kvasir_x1", {})
    rows = []

    for m_key, m_label in [("baseline_1", "Baseline-1"), ("baseline_2", "Baseline-2"), ("ci_gci", "Proposed CI-GCI")]:
        m_info = kvasir_models.get(m_key, {})
        if not m_info:
            continue
        overall = m_info.get("overall", {})
        closed = m_info.get("closed", {})
        open_sp = m_info.get("open", {})

        row = {
            "Model": m_label,
            "Overall Acc": f"{overall.get('acc', {}).get('mean', 0.0)*100:.2f}%" if "acc" in overall else "--",
            "Closed Acc": f"{closed.get('acc', {}).get('mean', 0.0)*100:.2f}%" if "acc" in closed else "--",
            "Open Acc": f"{open_sp.get('acc', {}).get('mean', 0.0)*100:.2f}%" if "acc" in open_sp else "--",
            "Macro-F1": f"{overall.get('f1', {}).get('mean', 0.0):.4f}" if "f1" in overall else "--",
            "ECE ↓": f"{overall.get('ece', {}).get('mean', 0.0):.4f}" if "ece" in overall else "--",
            "Brier ↓": f"{overall.get('brier', {}).get('mean', 0.0):.4f}" if "brier" in overall else "--",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_out = os.path.join(output_dir, "table_2_reasoning_breakdown.csv")
    md_out = os.path.join(output_dir, "table_2_reasoning_breakdown.md")
    df.to_csv(csv_out, index=False)
    with open(md_out, "w") as f:
        f.write("### Table 2: Kvasir-VQA-x1 Diagnostic Breakdown\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"-> Emitted {csv_out}")


def generate_table_3_hallucination_and_calibration(output_dir: str, canonical_data: dict):
    """Table 3 / Table 6: Calibration and Reliability Error Across Benchmarks."""
    print("[Tables] Dynamically building Calibration & Reliability Table...")
    rows = []
    models = canonical_data.get("models", {})

    for d_key, d_label in [("vqa_rad", "VQA-RAD"), ("slake", "SLAKE"), ("pathvqa", "PathVQA"), ("kvasir_x1", "Kvasir-VQA-x1")]:
        for m_key, m_label in [("baseline_2", f"{d_label} - Baseline VLM"), ("ci_gci", f"{d_label} - Proposed CI-GCI")]:
            m_info = models.get(d_key, {}).get(m_key, {})
            if not m_info:
                continue
            overall = m_info.get("overall", {})
            row = {
                "Cohort & Model": m_label,
                "Accuracy": f"{overall.get('acc', {}).get('mean', 0.0)*100:.2f}%" if "acc" in overall else "--",
                "ECE ↓": f"{overall.get('ece', {}).get('mean', 0.0):.4f}" if "ece" in overall else "--",
                "Brier Score ↓": f"{overall.get('brier', {}).get('mean', 0.0):.4f}" if "brier" in overall else "--",
                "Macro F1": f"{overall.get('f1', {}).get('mean', 0.0):.4f}" if "f1" in overall else "--",
                "Number of Seeds": m_info.get("n_seeds", "--")
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_out = os.path.join(output_dir, "table_6_calibration_abstention.csv")
    md_out = os.path.join(output_dir, "table_6_calibration_abstention.md")
    df.to_csv(csv_out, index=False)
    with open(md_out, "w") as f:
        f.write("### Table 6: Calibration and Reliability Metrics Across Benchmark Cohorts\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"-> Emitted {csv_out}")


def generate_table_7_ablation_modes(output_dir: str, canonical_data: dict):
    """
    Table 7 / Table VIII: Decisive Ablation Study of Counterfactual Intervention Mechanisms.
    Compares: Zero Masking, Gaussian Blur, Nearest Neighbor, Generative Inpainting.
    """
    print("[Tables] Dynamically building Intervention Mode Ablation Table...")
    rows = []
    models = canonical_data.get("models", {})

    mode_map = [
        ("ablation_zero", "Zero / Black-Box Masking (OOD Artifacts)"),
        ("ablation_blur", "Gaussian Blur (Border Discontinuities)"),
        ("ablation_nearest", "Nearest-Neighbor Patch Inpainting"),
        ("ci_gci", "Generative Anatomical Inpainting (Proposed CI-GCI)")
    ]

    for m_key, m_desc in mode_map:
        slake_info = models.get("slake", {}).get(m_key, {}).get("overall", {})
        vqa_info = models.get("vqa_rad", {}).get(m_key, {}).get("overall", {})

        row = {
            "Intervention Mechanism": m_desc,
            "SLAKE Acc": f"{slake_info.get('acc', {}).get('mean', 0.0)*100:.2f}%" if "acc" in slake_info else "--",
            "SLAKE ECE ↓": f"{slake_info.get('ece', {}).get('mean', 0.0):.4f}" if "ece" in slake_info else "--",
            "VQA-RAD Acc": f"{vqa_info.get('acc', {}).get('mean', 0.0)*100:.2f}%" if "acc" in vqa_info else "--",
            "VQA-RAD ECE ↓": f"{vqa_info.get('ece', {}).get('mean', 0.0):.4f}" if "ece" in vqa_info else "--",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_out = os.path.join(output_dir, "ablation_1_modules.csv")
    md_out = os.path.join(output_dir, "ablation_1_modules.md")
    df.to_csv(csv_out, index=False)
    with open(md_out, "w") as f:
        f.write("### Table 7: Decisive Ablation of Counterfactual Intervention Modes\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"-> Emitted {csv_out}")


def generate_all_tables(
    output_dir: str = "outputs/tables",
    canonical_path: str = "outputs/canonical.json"
):
    """Orchestrate dynamic generation of all publication-grade benchmark tables."""
    os.makedirs(output_dir, exist_ok=True)
    canonical_data = load_canonical_data(canonical_path)

    generate_table_1_main_results(output_dir, canonical_data)
    generate_table_2_kvasir_breakdown(output_dir, canonical_data)
    generate_table_3_hallucination_and_calibration(output_dir, canonical_data)
    generate_table_7_ablation_modes(output_dir, canonical_data)

    print(f"\n[Tables] Successfully generated all dynamic publication tables in '{output_dir}'.")


if __name__ == "__main__":
    generate_all_tables()
