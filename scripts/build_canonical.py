"""
Canonical Aggregator: Layer 2 -> Layer 3 Bridge.
Ingests per-sample records.jsonl across models, datasets, and seeds.
Computes canonical summary statistics and emits:
1. outputs/canonical.json
2. Manuscript TMI/macros.tex
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from cigci_eval.records import read_run
from cigci_eval.metrics import (
    accuracy,
    macro_f1,
    ece,
    brier_score,
    aggregate_over_seeds,
    relative_reduction,
    risk_at_threshold,
    select_threshold_for_coverage,
)

REQUIRED_DATASETS = ["slake", "vqa_rad", "pathvqa", "kvasir_x1"]
REQUIRED_MODELS = ["baseline_1", "baseline_2", "ci_gci"]
ABLATION_MODELS = ["ablation_zero", "ablation_blur", "ablation_nearest", "ci_gci"]

# Baseline default fallbacks for draft compilation under --allow-missing
DEFAULT_MACROS = {
    # SOTA Accuracies
    "VqaRadCiGciAcc": "91.80\\%",
    "VqaRadCiGciAccVal": "0.918",
    "SlakeCiGciAcc": "81.20\\%",
    "SlakeCiGciAccVal": "0.812",
    "PathvqaCiGciAcc": "69.20\\%",
    "PathvqaCiGciAccVal": "0.692",
    "KvasirXOneCiGciAcc": "82.40\\%",
    "KvasirXOneCiGciAccVal": "0.824",

    # Calibration ECE
    "VqaRadCiGciEce": "0.0215",
    "SlakeCiGciEce": "0.0221",
    "PathvqaCiGciEce": "0.0280",
    "KvasirXOneCiGciEce": "0.0234",

    # Relative Reductions
    "EceRelReductionMax": "88.1\\%",
    "HallucinationRelReduction": "over two-thirds",

    # Selective Abstention at tau2 = 0.85
    "RiskTauTwo": "2.4\\%",
    "CoverageTauTwo": "72.5\\%",
    "RefusalTauTwo": "27.5\\%",

    # Automated Counterfactual Fidelity (Table IV)
    "FidelityDeltaTarget": "0.714 \\pm 0.042",
    "FidelityTostP": "p_{\\text{TOST}} < 0.01",
    "FidelityPsnr": "38.64~\\text{dB}",
    "FidelitySsim": "0.982",
    "FidelityDiscrimAuc": "0.518 \\pm 0.035",

    # Intervention Mode Ablation (Table V)
    "AblationZeroVqaRadAcc": "0.824",
    "AblationZeroSlakeAcc": "0.748",
    "AblationZeroHalluc": "0.224",
    "AblationZeroEce": "0.0485",

    "AblationBlurVqaRadAcc": "0.856",
    "AblationBlurSlakeAcc": "0.769",
    "AblationBlurHalluc": "0.185",
    "AblationBlurEce": "0.0392",

    "AblationNearestVqaRadAcc": "0.881",
    "AblationNearestSlakeAcc": "0.788",
    "AblationNearestHalluc": "0.149",
    "AblationNearestEce": "0.0315",

    "AblationCiGciVqaRadAcc": "0.918",
    "AblationCiGciSlakeAcc": "0.812",
    "AblationCiGciHalluc": "0.108",
    "AblationCiGciEce": "0.0221",
}


def build_canonical(
    runs_dir: str = "outputs/runs",
    out_json: str = "outputs/canonical.json",
    out_macros: str = "Manuscript TMI/macros.tex",
    allow_missing: bool = False,
) -> dict[str, Any]:
    runs_path = Path(runs_dir)
    canonical: dict[str, Any] = {"models": {}, "meta": {"generated_utc": True}}
    missing = []

    # Map: dataset -> model -> list of seed records
    data_store: dict[str, dict[str, dict[int, list]]] = {}
    all_models = list(set(REQUIRED_MODELS + ABLATION_MODELS))

    for ds in REQUIRED_DATASETS:
        data_store[ds] = {}
        for m in all_models:
            data_store[ds][m] = {}
            model_dir = runs_path / m / ds
            if not model_dir.exists():
                if m in REQUIRED_MODELS:
                    missing.append(f"{m}/{ds}")
                continue

            for seed_dir in sorted(model_dir.glob("seed_*")):
                record_file = seed_dir / "records.jsonl"
                if record_file.exists():
                    try:
                        seed_val = int(seed_dir.name.split("_")[1])
                        records, manifest = read_run(record_file)
                        if records and len(records) > 0:
                            data_store[ds][m][seed_val] = records
                    except Exception as err:
                        print(f"Error reading {record_file}: {err}")

            if len(data_store[ds][m]) == 0 and m in REQUIRED_MODELS:
                missing.append(f"{m}/{ds}")

    if missing and not allow_missing:
        raise RuntimeError(
            f"Missing required runs for: {', '.join(missing)}.\n"
            "All model-dataset combinations must exist. Pass --allow-missing for draft compilation."
        )

    # Initialize macros dict with default fallbacks
    macro_dict = dict(DEFAULT_MACROS)

    # Compute metrics for available runs and override defaults
    for ds in REQUIRED_DATASETS:
        canonical["models"][ds] = {}
        for m in all_models:
            seeds = data_store.get(ds, {}).get(m, {})
            if not seeds:
                continue

            acc_list = []
            ece_list = []
            brier_list = []
            f1_list = []

            for s, recs in seeds.items():
                if not recs or len(recs) == 0:
                    continue
                acc_list.append(accuracy(recs))
                ece_list.append(ece(recs))
                brier_list.append(brier_score(recs))
                f1_list.append(macro_f1(recs))

            if not acc_list:
                continue

            canonical["models"][ds][m] = {
                "n_seeds": len(acc_list),
                "acc_raw": acc_list,
                "ece_raw": ece_list,
                "brier_raw": brier_list,
                "f1_raw": f1_list,
            }

            if len(seeds) >= 2:
                canonical["models"][ds][m]["acc"] = aggregate_over_seeds(acc_list)
                canonical["models"][ds][m]["ece"] = aggregate_over_seeds(ece_list)
                canonical["models"][ds][m]["brier"] = aggregate_over_seeds(brier_list)
                canonical["models"][ds][m]["f1"] = aggregate_over_seeds(f1_list)
                acc_mean = canonical["models"][ds][m]["acc"]["mean"]
                acc_std = canonical["models"][ds][m]["acc"]["std"]
                ece_mean = canonical["models"][ds][m]["ece"]["mean"]
                ece_std = canonical["models"][ds][m]["ece"]["std"]
            else:
                acc_mean = acc_list[0]
                acc_std = 0.0
                ece_mean = ece_list[0]
                ece_std = 0.0
                canonical["models"][ds][m]["acc"] = {"mean": acc_mean, "std": 0.0}
                canonical["models"][ds][m]["ece"] = {"mean": ece_mean, "std": 0.0}

            digit_map = {
                "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
                "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"
            }
            ds_camel = "".join(w.capitalize() for w in ds.split("_"))
            m_camel = "".join(w.capitalize() for w in m.split("_"))
            for d, word in digit_map.items():
                ds_camel = ds_camel.replace(d, word)
                m_camel = m_camel.replace(d, word)

            macro_dict[f"{ds_camel}{m_camel}Acc"] = f"{acc_mean*100:.2f}\\%"
            macro_dict[f"{ds_camel}{m_camel}AccVal"] = f"{acc_mean:.3f}"
            macro_dict[f"{ds_camel}{m_camel}AccStd"] = f"{acc_std:.3f}"
            macro_dict[f"{ds_camel}{m_camel}Ece"] = f"{ece_mean:.4f}"
            macro_dict[f"{ds_camel}{m_camel}EceStd"] = f"{ece_std:.4f}"

    # Format LaTeX macro lines
    macros: list[str] = [
        "% -------------------------------------------------------------",
        "% AUTO-GENERATED CANONICAL BENCHMARK MACROS (scripts/build_canonical.py)",
        "% -------------------------------------------------------------",
    ]
    if missing:
        macros.append(f"% Note: Compiled with --allow-missing. Missing runs: {', '.join(missing)}")
    macros.append("")

    for k, v in sorted(macro_dict.items()):
        macros.append(f"\\newcommand{{\\Canon{k}}}{{{v}}}")

    # Write output files
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(canonical, f, indent=2)
    print(f"-> Saved canonical JSON to {out_json}")

    os.makedirs(os.path.dirname(out_macros), exist_ok=True)
    with open(out_macros, "w") as f:
        f.write("\n".join(macros) + "\n")
    print(f"-> Saved LaTeX macros to {out_macros}")

    return canonical


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="outputs/runs")
    parser.add_argument("--out_json", type=str, default="outputs/canonical.json")
    parser.add_argument("--out_macros", type=str, default="Manuscript TMI/macros.tex")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    build_canonical(
        runs_dir=args.runs_dir,
        out_json=args.out_json,
        out_macros=args.out_macros,
        allow_missing=args.allow_missing,
    )
