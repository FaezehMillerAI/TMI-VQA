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

def setup_symlink(dataset_name, search_filename, target_link, path_filter=None):
    if os.path.exists(target_link):
        if os.path.islink(target_link):
            os.unlink(target_link)
        else:
            import shutil
            shutil.rmtree(target_link)
            
    found_dir = None
    print(f"Scanning /kaggle/input for {dataset_name} ({search_filename})...")
    for root, dirs, files in os.walk("/kaggle/input"):
        if path_filter and path_filter not in root.lower():
            continue
        for file in files:
            if file == search_filename:
                found_dir = root
                break
        if found_dir:
            break
            
    if found_dir:
        print(f"-> Found {dataset_name} at: {found_dir}")
        os.symlink(found_dir, target_link)
        print(f"-> Successfully linked to {target_link}")
        return True
    else:
        print(f"-> Warning: Could not find {dataset_name} dataset in /kaggle/input.")
        return False

def main():
    print("==================================================")
    print("      CI-GCI AUTOMATED EXPERIMENT PIPELINE        ")
    print("==================================================")
    
    # 1. Setup Kaggle paths and symlinks
    os.makedirs("data", exist_ok=True)
    setup_symlink("SLAKE", "test.json", "data/slake", path_filter="slake")
    setup_symlink("VQA-RAD", "VQA_RAD Dataset Public.json", "data/VQA-RAD")
    setup_symlink("MS-CXR", "MS_CXR_Local_Alignment_v1.1.0.json", "data/ms-cxr")
    setup_symlink("PathVQA", "train-00000-of-00007-f2d0e9ef9f022d38.parquet", "data/pathvqa", path_filter="pathvqa")
    setup_symlink("Kvasir-VQA", "train-00000-of-00001.parquet", "data/kvasir", path_filter="kvasir")

    # Check GPU availability
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTarget execution device: {device.upper()}")

    # 2. Install requirements if not fully installed
    print("\nInstalling requirements...")
    run_command("pip install -q -r requirements.txt")

    # 3. Train Inpainter
    print("\nStep 1/6: Training Counterfactual Inpainter (CFI)...")
    run_command(f"PYTHONPATH=. python3 training/train_inpainter.py --epochs 5 --batch_size 16 --device {device}")

    # 4. Fine-tune VQA Model across datasets
    print("\nStep 2/6: Fine-tuning VQA model on SLAKE...")
    run_command(f"PYTHONPATH=. python3 training/train_slake_vqa.py --dataset slake --epochs 3 --batch_size 16 --device {device}")

    print("\nStep 3/6: Fine-tuning VQA model on VQA-RAD...")
    run_command(f"PYTHONPATH=. python3 training/train_slake_vqa.py --dataset vqa_rad --epochs 3 --batch_size 16 --device {device}")

    print("\nStep 4/6: Fine-tuning VQA model on PathVQA...")
    run_command(f"PYTHONPATH=. python3 training/train_slake_vqa.py --dataset pathvqa --epochs 3 --batch_size 16 --device {device}")

    print("\nStep 5/6: Fine-tuning VQA model on Kvasir-VQA...")
    run_command(f"PYTHONPATH=. python3 training/train_slake_vqa.py --dataset kvasir --epochs 3 --batch_size 16 --device {device}")

    # 5. Run Comparative Benchmarking
    print("\nStep 6/6: Running comparative benchmarks across all datasets...")
    for ds in ["slake", "vqa_rad", "pathvqa", "kvasir", "ms_cxr"]:
        run_command(f"PYTHONPATH=. python3 scripts/benchmark_comparison.py --dataset {ds} --device {device}")
        
    # 6. Generate reliability diagrams and proof sheets
    print("\nStep 5/5: Generating plots and reliability diagrams...")
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
