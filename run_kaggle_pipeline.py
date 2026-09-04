"""
Unified Kaggle GPU Execution Pipeline for CI-GCI (IEEE TMI)
Executes:
1. Data setup and symlinking from /kaggle/input or Hugging Face
2. Multi-seed fine-tuning on GPU
3. Per-sample prediction record generation (cigci_eval contract)
4. Automated counterfactual fidelity & preservation evaluation on real pixels
5. Canonical metrics aggregation and LaTeX macro generation (macros.tex)
"""

import os
import subprocess
import sys
import shutil
import argparse
import zipfile


def run_command(cmd_str):
    print(f"\n[EXEC] {cmd_str}")
    process = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"Error: Command failed with exit code {process.returncode}")
        return False
    return True


def setup_symlink(dataset_name, search_filename, target_link, hf_repo_id=None, path_filter=None):
    os.makedirs("data", exist_ok=True)
    if os.path.exists(target_link) and len(os.listdir(target_link)) > 0:
        print(f"-> {dataset_name} is already prepared at: {target_link}")
        return True

    found_dir = None
    if os.path.exists("/kaggle/input"):
        print(f"Scanning /kaggle/input for {dataset_name} ({search_filename})...")
        for root, dirs, files in os.walk("/kaggle/input"):
            if path_filter and path_filter not in root.lower():
                continue
            if search_filename in files:
                found_dir = root
                break

    if found_dir:
        print(f"-> Found {dataset_name} in /kaggle/input at: {found_dir}")
        if os.path.exists(target_link):
            if os.path.islink(target_link):
                os.unlink(target_link)
            else:
                shutil.rmtree(target_link)
        os.symlink(found_dir, target_link)
        print(f"-> Successfully linked to {target_link}")
        return True

    if hf_repo_id:
        print(f"-> {dataset_name} not found in /kaggle/input. Auto-downloading from Hugging Face ({hf_repo_id})...")
        os.makedirs(target_link, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                local_dir=target_link,
                allow_patterns=["*.parquet", "*.parquet.gzip", "*.json", "*.csv", "*.txt", "*.zip"]
            )
            print(f"-> Successfully downloaded {dataset_name}")
            return True
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            return False
    return False


def main():
    parser = argparse.ArgumentParser(description="Kaggle GPU Pipeline Runner for CI-GCI")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--skip_train", action="store_true", help="Skip training and run inference only")
    args = parser.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    print("==================================================")
    print("  STARTING CI-GCI KAGGLE BENCHMARK PIPELINE       ")
    print(f"  Device: {device} | Seeds: {args.seeds} | Epochs: {args.epochs}")
    print("==================================================")

    # 1. Setup Datasets
    print("\n[Step 1/5] Setting up multi-center benchmark datasets...")
    setup_symlink("SLAKE", "test.json", "data/slake", hf_repo_id="BoKelvin/SLAKE", path_filter="slake")
    setup_symlink("VQA-RAD", "VQA_RAD Dataset Public.json", "data/VQA-RAD", hf_repo_id="flaviagiammarino/vqa-rad")
    setup_symlink("PathVQA", "train-00000-of-00007-f2d0e9ef9f022d38.parquet", "data/pathvqa", hf_repo_id="flaviagiammarino/path-vqa", path_filter="pathvqa")
    setup_symlink("Kvasir-VQA", "train-00000-of-00001.parquet", "data/kvasir", hf_repo_id="SimulaMet/Kvasir-VQA-x1", path_filter="kvasir")
    setup_symlink("MS-CXR", "MS_CXR_Local_Alignment_v1.1.0.json", "data/ms-cxr")

    # 2. Model Training / Fine-tuning
    if not args.skip_train:
        print("\n[Step 2/5] Training inpainter and fine-tuning models across seeds...")
        run_command(f"PYTHONPATH=. python3 training/train_inpainter.py --epochs 5 --batch_size {args.batch_size} --device {device}")
        for s in args.seeds:
            for ds in ["slake", "vqa_rad", "pathvqa", "kvasir"]:
                run_command(f"PYTHONPATH=. python3 training/train_slake_vqa.py --dataset {ds} --epochs {args.epochs} --batch_size {args.batch_size} --device {device}")

    # 3. Layer 1 Inference: Emit conforming records.jsonl
    print("\n[Step 3/5] Running Layer 1 inference and emitting per-sample PredictionRecords...")
    for s in args.seeds:
        for ds in ["slake", "vqa_rad", "pathvqa", "kvasir_x1"]:
            for m in ["ci_gci", "baseline_1", "baseline_2"]:
                run_command(f"PYTHONPATH=. python3 cigci_eval/inference_adapter.py --dataset {ds} --model {m} --seed {s} --device {device}")

    # Decisive Inpainting Mode Ablation (Task 8: Diffusion vs Black-Box vs Gaussian Blur vs Nearest)
    print("\n[Step 3b/5] Running Decisive Inpainting Mode Ablations...")
    for m_abl in ["ablation_zero", "ablation_blur", "ablation_nearest"]:
        for ds in ["slake", "vqa_rad"]:
            run_command(f"PYTHONPATH=. python3 cigci_eval/inference_adapter.py --dataset {ds} --model {m_abl} --seed 42 --device {device}")

    # 4. Layer 2: Automated Counterfactual Fidelity Evaluation
    print("\n[Step 4/5] Evaluating image-domain counterfactual fidelity...")
    # Computes real PSNR/SSIM outside mask and independent classifier attenuation
    run_command("PYTHONPATH=. pytest -v tests/test_fidelity.py")

    # 5. Layer 3: Build Canonical Metrics & LaTeX Macros
    print("\n[Step 5/5] Aggregating canonical metrics and compiling macros.tex...")
    run_command("PYTHONPATH=. python3 scripts/build_canonical.py --allow-missing")

    # Package outputs
    zip_path = "outputs_cigci_verified.zip"
    print(f"\nCompressing outputs to {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk("outputs"):
            for file in files:
                full_p = os.path.join(root, file)
                zf.write(full_p, os.path.relpath(full_p, "."))
        if os.path.exists("Manuscript TMI/macros.tex"):
            zf.write("Manuscript TMI/macros.tex", "Manuscript TMI/macros.tex")

    print("\n==================================================")
    print("  CI-GCI PIPELINE COMPLETED SUCCESSFULLY!         ")
    print(f"  All outputs bundled into: {zip_path}")
    print("==================================================")


if __name__ == "__main__":
    main()
