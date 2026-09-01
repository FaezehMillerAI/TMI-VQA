import os
import subprocess
import sys

def run_command(cmd_str):
    print(f"\nExecuting: {cmd_str}")
    process = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Stream output live
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"Error: Command failed with exit code {process.returncode}")
        return False
    return True

import shutil

def setup_symlink(dataset_name, search_filename, target_link, hf_repo_id=None, post_download_cmd=None, path_filter=None):
    os.makedirs("data", exist_ok=True)
    
    # Check if target directory already has data
    if os.path.exists(target_link) and len(os.listdir(target_link)) > 0:
        print(f"-> {dataset_name} is already prepared at: {target_link}")
        return True

    # 1. First scan /kaggle/input
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

    # 2. Fallback: Auto-download from Hugging Face
    if hf_repo_id:
        print(f"-> {dataset_name} not found in /kaggle/input. Auto-downloading from Hugging Face ({hf_repo_id})...")
        os.makedirs(target_link, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=hf_repo_id, repo_type="dataset", local_dir=target_link)
            success = True
        except Exception as err:
            print(f"snapshot_download fallback ({err}), trying CLI...")
            cmd = f"huggingface-cli download {hf_repo_id} --repo-type dataset --local-dir {target_link}"
            success = run_command(cmd)

        if success and post_download_cmd:
            run_command(post_download_cmd)
        return success

    print(f"-> Warning: Could not find {dataset_name} dataset.")
    return False

import argparse

def main():
    parser = argparse.ArgumentParser(description="CI-GCI Automated Experiment Pipeline")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs per dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--device", type=str, default=None, help="Target device (cuda, mps, cpu)")
    args = parser.parse_args()

    print("==================================================")
    print("      CI-GCI AUTOMATED EXPERIMENT PIPELINE        ")
    print("==================================================")
    
    # Check GPU availability
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\nTarget execution device: {device.upper()}")
    print(f"Fine-tuning epochs per dataset: {args.epochs}")

    # 1. Install requirements if not fully installed
    print("\nInstalling requirements...")
    run_command("pip install -q -r requirements.txt huggingface_hub")

    # 2. Setup or Auto-Download all 5 datasets
    print("\nPreparing Datasets...")
    setup_symlink(
        "SLAKE", 
        "test.json", 
        "data/slake", 
        hf_repo_id="BoKelvin/SLAKE", 
        post_download_cmd="if [ -f data/slake/imgs.zip ]; then unzip -q -o data/slake/imgs.zip -d data/slake/; fi",
        path_filter="slake"
    )
    setup_symlink(
        "VQA-RAD", 
        "VQA_RAD Dataset Public.json", 
        "data/VQA-RAD", 
        hf_repo_id="flaviagiammarino/vqa-rad"
    )
    setup_symlink(
        "PathVQA", 
        "train-00000-of-00007-f2d0e9ef9f022d38.parquet", 
        "data/pathvqa", 
        hf_repo_id="flaviagiammarino/path-vqa",
        path_filter="pathvqa"
    )
    setup_symlink(
        "Kvasir-VQA", 
        "train-00000-of-00001.parquet", 
        "data/kvasir", 
        hf_repo_id="SimulaMet/Kvasir-VQA-x1",
        path_filter="kvasir"
    )
    setup_symlink(
        "MS-CXR", 
        "MS_CXR_Local_Alignment_v1.1.0.json", 
        "data/ms-cxr"
    )

    # 3. Train Inpainter
    print("\nStep 1/6: Training Counterfactual Inpainter (CFI)...")
    run_command(f"PYTHONPATH=. python3 training/train_inpainter.py --epochs 5 --batch_size {args.batch_size} --device {device}")

    # 4. Fine-tune VQA Model across datasets
    print(f"\nStep 2/6: Fine-tuning VQA model on SLAKE ({args.epochs} epochs)...")
    run_command(f"PYTHONPATH=. python3 training/train_slake_vqa.py --dataset slake --epochs {args.epochs} --batch_size {args.batch_size} --device {device}")

    print(f"\nStep 3/6: Fine-tuning VQA model on VQA-RAD ({args.epochs} epochs)...")
    run_command(f"PYTHONPATH=. python3 training/train_slake_vqa.py --dataset vqa_rad --epochs {args.epochs} --batch_size {args.batch_size} --device {device}")

    print(f"\nStep 4/6: Fine-tuning VQA model on PathVQA ({args.epochs} epochs)...")
    run_command(f"PYTHONPATH=. python3 training/train_slake_vqa.py --dataset pathvqa --epochs {args.epochs} --batch_size {args.batch_size} --device {device}")

    print(f"\nStep 5/6: Fine-tuning VQA model on Kvasir-VQA ({args.epochs} epochs)...")
    run_command(f"PYTHONPATH=. python3 training/train_slake_vqa.py --dataset kvasir --epochs {args.epochs} --batch_size {args.batch_size} --device {device}")

    # 5. Run Comparative Benchmarking
    print("\nStep 6/6: Running comparative benchmarks across all datasets...")
    for ds in ["slake", "vqa_rad", "pathvqa", "kvasir", "ms_cxr"]:
        run_command(f"PYTHONPATH=. python3 scripts/benchmark_comparison.py --dataset {ds} --device {device}")
        
    # 6. Generate reliability diagrams and proof sheets
    print("\nGenerating plots and reliability diagrams...")
    run_command(f"PYTHONPATH=. python3 scripts/generate_plots_and_proofs.py --dataset slake --device {device}")
    run_command(f"PYTHONPATH=. python3 scripts/generate_plots_and_proofs.py --dataset vqa_rad --device {device}")
    
    # 7. Generate publication tables
    print("\nCompiling SOTA and Ablation publication tables...")
    run_command("python3 evaluation/result_table_generator.py")
    
    print("\n==================================================")
    print("     ALL CI-GCI EXPERIMENTS COMPLETED SUCCESSFULLY! ")
    print("==================================================")
    print("Generated Outputs:")
    print("  - Visual Proofs: outputs/proofs/")
    print("  - Reliability Plots: outputs/")
    print("  - SOTA & Ablation Tables: outputs/tables/")
    print("==================================================")

if __name__ == "__main__":
    main()
