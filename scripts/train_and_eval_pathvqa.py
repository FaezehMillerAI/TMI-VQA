#!/usr/bin/env python3
"""
Standalone Automated Runner for PathVQA Fine-Tuning & Evaluation.
Executes:
1. Dataset verification and linking (handles local and Kaggle paths)
2. Fine-tuning CQC-Net on PathVQA closed-set questions
3. Generation of conforming records.jsonl across models and seeds
4. Canonical metric aggregation via scripts/build_canonical.py
5. Packaging of outputs
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def run_cmd(cmd: str) -> bool:
    print(f"\n[EXEC] {cmd}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    res = subprocess.run(cmd, shell=True, env=env)
    if res.returncode != 0:
        print(f"Error: Command failed with code {res.returncode}", flush=True)
        return False
    return True


def ensure_pathvqa_data(data_dir: str = "data/pathvqa") -> bool:
    os.makedirs("data", exist_ok=True)
    target = Path(data_dir)

    # Check if target already has parquet files
    if target.exists() and list(target.glob("*.parquet")):
        print(f"-> PathVQA dataset verified at: {target}")
        return True

    # Check local PathVQA/ directory
    local_pathvqa = Path("PathVQA")
    if local_pathvqa.exists() and list(local_pathvqa.glob("*.parquet")):
        print(f"-> Linking local {local_pathvqa} to {target}")
        if target.is_symlink() or target.exists():
            if target.is_symlink():
                target.unlink()
            else:
                shutil.rmtree(target)
        os.symlink(local_pathvqa.resolve(), target)
        return True

    # Check Kaggle /kaggle/input
    if os.path.exists("/kaggle/input"):
        print("Scanning /kaggle/input for PathVQA parquet files...")
        for root, _, files in os.walk("/kaggle/input"):
            if any(f.endswith(".parquet") and "train" in f for f in files):
                print(f"-> Found PathVQA at: {root}")
                if target.is_symlink() or target.exists():
                    if target.is_symlink():
                        target.unlink()
                    else:
                        shutil.rmtree(target)
                os.symlink(root, target)
                return True

    print("Warning: Could not find PathVQA parquet files. Attempting Hugging Face download...")
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(target, exist_ok=True)
        snapshot_download(
            repo_id="flaviagiammarino/path-vqa",
            repo_type="dataset",
            local_dir=str(target),
            allow_patterns=["*.parquet"]
        )
        print(f"-> Successfully downloaded PathVQA to {target}")
        return True
    except Exception as err:
        print(f"Error downloading PathVQA: {err}")
        return False


def main():
    parser = argparse.ArgumentParser(description="PathVQA Fine-Tuning & Evaluation Runner")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda/mps/cpu)")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and run inference only")
    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 65)
    print("      PATHVQA BENCHMARK TRAINING & INFERENCE PIPELINE         ")
    print(f"      Device: {device} | Seeds: {args.seeds} | Epochs: {args.epochs}")
    print("=" * 65)

    # 1. Dataset Verification
    print("\n[Step 1/4] Verifying PathVQA Dataset...")
    if not ensure_pathvqa_data():
        print("Fatal: PathVQA dataset could not be loaded. Exiting.")
        sys.exit(1)

    # 2. Fine-Tuning Phase
    if not args.skip_train:
        print("\n" + "=" * 65)
        print("  [Step 2/4] FINE-TUNING CQC-NET ON PATHVQA CLOSED QA PAIRS  ")
        print("=" * 65)
        cmd_train = (
            f"PYTHONPATH=. python3 -u training/train_slake_vqa.py "
            f"--dataset pathvqa "
            f"--epochs {args.epochs} "
            f"--batch_size {args.batch_size} "
            f"--lr {args.lr} "
            f"--device {device}"
        )
        success = run_cmd(cmd_train)
        if not success:
            print("Error during PathVQA fine-tuning. Exiting.")
            sys.exit(1)
    else:
        print("\n[Step 2/4] Skipping fine-tuning (--skip_train passed).")

    # Verify checkpoint exists
    chk_path = Path("models/pathvqa_vqa_model.pth")
    if not chk_path.exists():
        print(f"Warning: Expected checkpoint at {chk_path} not found!")

    # 3. Test Inference Across Models and Seeds
    print("\n" + "=" * 65)
    print("  [Step 3/4] GENERATING TEST INFERENCE RECORDS              ")
    print("=" * 65)
    models_to_eval = ["ci_gci", "baseline_1", "baseline_2"]
    for s in args.seeds:
        for m in models_to_eval:
            cmd_infer = (
                f"PYTHONPATH=. python3 -u cigci_eval/inference_adapter.py "
                f"--dataset pathvqa "
                f"--model {m} "
                f"--seed {s} "
                f"--device {device}"
            )
            run_cmd(cmd_infer)

    # Also run SLAKE inference with closed filter to update SLAKE records
    print("\n" + "=" * 65)
    print("  [Step 3b/4] UPDATING SLAKE INFERENCE WITH CLOSED FILTER   ")
    print("=" * 65)
    for s in args.seeds:
        for m in models_to_eval:
            cmd_slake = (
                f"PYTHONPATH=. python3 -u cigci_eval/inference_adapter.py "
                f"--dataset slake "
                f"--model {m} "
                f"--seed {s} "
                f"--device {device}"
            )
            run_cmd(cmd_slake)

    # 4. Canonical Metric Aggregation
    print("\n" + "=" * 65)
    print("  [Step 4/4] AGGREGATING CANONICAL METRICS & LATEX MACROS   ")
    print("=" * 65)
    run_cmd("PYTHONPATH=. python3 -u scripts/build_canonical.py --allow-missing")

    # 5. Compress updated outputs package
    zip_path = "outputs_pathvqa_updated.zip"
    print(f"\nPackaging updated outputs to {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk("outputs"):
            for file in files:
                full_p = os.path.join(root, file)
                zf.write(full_p, os.path.relpath(full_p, "."))
        if os.path.exists("models/pathvqa_vqa_model.pth"):
            zf.write("models/pathvqa_vqa_model.pth", "models/pathvqa_vqa_model.pth")
        if os.path.exists("Manuscript TMI/macros.tex"):
            zf.write("Manuscript TMI/macros.tex", "Manuscript TMI/macros.tex")

    print("\n" + "=" * 65)
    print("      PATHVQA BENCHMARK PIPELINE COMPLETE!                 ")
    print(f"      Updated results saved to {zip_path}                 ")
    print("=" * 65)


if __name__ == "__main__":
    main()
