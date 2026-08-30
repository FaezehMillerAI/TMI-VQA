# Kaggle Notebook Cells: Mature Multi-Dataset CI-GCI Pipeline

Copy and paste the following cells into your Kaggle Notebook. This version dynamically supports **SLAKE**, **VQA-RAD**, **MS-CXR**, and **HEAL-MedVQA**.

---

### **Cell 1: Clone Repository & Setup Environment**
Run this block to clone the codebase and install requirements.

```python
# 1. Clone the repository
!git clone https://github.com/FaezehMillerAI/TMI-VQA.git
%cd TMI-VQA

# 2. Install requirements
!pip install -r requirements.txt
```

---

### **Cell 2: Automatic Dataset Setup & Downloader**
This cell automatically checks `/kaggle/input/` for attached datasets, and if they are not attached, **auto-downloads them directly from Hugging Face** into `data/`:

```python
import os, shutil

os.makedirs("data", exist_ok=True)

# Install huggingface-cli
!pip install -q huggingface_hub[cli]

def setup_symlink(dataset_name, search_filename, target_link, hf_repo_id=None, post_download_cmd=None, path_filter=None):
    if os.path.exists(target_link) and len(os.listdir(target_link)) > 0:
        print(f"-> {dataset_name} is already prepared at: {target_link}")
        return True

    found_dir = None
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if path_filter and path_filter not in root.lower(): continue
            if search_filename in files:
                found_dir = root
                break
                
    if found_dir:
        print(f"-> Found {dataset_name} in /kaggle/input at: {found_dir}")
        if os.path.exists(target_link):
            if os.path.islink(target_link): os.unlink(target_link)
            else: shutil.rmtree(target_link)
        os.symlink(found_dir, target_link)
        print(f"-> Successfully linked to {target_link}")
        return True

    if hf_repo_id:
        print(f"-> Auto-downloading {dataset_name} from Hugging Face ({hf_repo_id})...")
        os.makedirs(target_link, exist_ok=True)
        !huggingface-cli download {hf_repo_id} --repo-type dataset --local-dir {target_link}
        if post_download_cmd:
            get_ipython().system(post_download_cmd)
        return True

    print(f"-> Warning: Could not prepare {dataset_name} dataset.")
    return False

# 1. Setup SLAKE
setup_symlink("SLAKE", "test.json", "data/slake", hf_repo_id="BoKelvin/SLAKE", post_download_cmd="if [ -f data/slake/imgs.zip ]; then unzip -q -o data/slake/imgs.zip -d data/slake/; fi", path_filter="slake")

# 2. Setup VQA-RAD
setup_symlink("VQA-RAD", "VQA_RAD Dataset Public.json", "data/VQA-RAD", hf_repo_id="flaviagiammarino/vqa-rad")

# 3. Setup PathVQA
setup_symlink("PathVQA", "train-00000-of-00007-f2d0e9ef9f022d38.parquet", "data/pathvqa", hf_repo_id="flaviagiammarino/path-vqa", path_filter="pathvqa")

# 4. Setup Kvasir-VQA
setup_symlink("Kvasir-VQA", "train-00000-of-00001.parquet", "data/kvasir", hf_repo_id="SimulaMet/Kvasir-VQA-x1", path_filter="kvasir")

# 5. Setup MS-CXR
setup_symlink("MS-CXR", "MS_CXR_Local_Alignment_v1.1.0.json", "data/ms-cxr")

print("\nAll datasets are configured and ready!")
```

---

### **Cell 3: Run Multi-Dataset Smoke Test**
Run this cell to immediately verify that VQA-RAD, MS-CXR, PathVQA, Kvasir-VQA, and SLAKE loaders and forward passes are working correctly.

```python
# Run the end-to-end smoke test on all datasets
!PYTHONPATH=. python3 scripts/smoke_test_all.py --device cuda
```

---

### **Cell 4: Train the Counterfactual Inpainter**
Trains the UNet generative network on SLAKE images (runs on GPU).

```python
# Train the Counterfactual Inpainter on GPU (saves to models/inpainter.pth)
!PYTHONPATH=. python3 training/train_inpainter.py --epochs 5 --batch_size 16 --device cuda
```

---

### **Cell 5: VQA Model Fine-Tuning**
Select which dataset you wish to train on by setting the `--dataset` argument:

```python
# Option A: Fine-tune on SLAKE
!PYTHONPATH=. python3 training/train_slake_vqa.py --dataset slake --epochs 3 --batch_size 16 --device cuda

# Option B: Fine-tune on VQA-RAD
# !PYTHONPATH=. python3 training/train_slake_vqa.py --dataset vqa_rad --epochs 3 --batch_size 16 --device cuda

# Option C: Fine-tune on PathVQA
# !PYTHONPATH=. python3 training/train_slake_vqa.py --dataset pathvqa --epochs 3 --batch_size 16 --device cuda

# Option D: Fine-tune on Kvasir-VQA
# !PYTHONPATH=. python3 training/train_slake_vqa.py --dataset kvasir --epochs 3 --batch_size 16 --device cuda
```

---

### **Cell 6: Run Comparative Benchmarks (SOTA Study)**
Evaluates and compares the uncalibrated baseline against our CI-GCI pipeline across all datasets.

```python
print("--- BENCHMARK RESULTS: SLAKE ---")
!PYTHONPATH=. python3 scripts/benchmark_comparison.py --dataset slake --device cuda

print("\n--- BENCHMARK RESULTS: VQA-RAD ---")
!PYTHONPATH=. python3 scripts/benchmark_comparison.py --dataset vqa_rad --device cuda

print("\n--- BENCHMARK RESULTS: PATHVQA ---")
!PYTHONPATH=. python3 scripts/benchmark_comparison.py --dataset pathvqa --device cuda

print("\n--- BENCHMARK RESULTS: KVASIR-VQA ---")
!PYTHONPATH=. python3 scripts/benchmark_comparison.py --dataset kvasir --device cuda

print("\n--- BENCHMARK RESULTS: MS-CXR ---")
!PYTHONPATH=. python3 scripts/benchmark_comparison.py --dataset ms_cxr --device cuda
```

---

### **Cell 7: Generate & Display Visual Proof Sheets**
Generates the reliability diagrams and side-by-side scan comparisons.

```python
# 1. Run plotting script for SLAKE
!PYTHONPATH=. python3 scripts/generate_plots_and_proofs.py --dataset slake --device cuda

# 2. Run plotting script for VQA-RAD
!PYTHONPATH=. python3 scripts/generate_plots_and_proofs.py --dataset vqa_rad --device cuda

# 3. Display the generated plots directly inside the Kaggle notebook
from IPython.display import Image, display

print("--- SLAKE Reliability Diagrams ---")
display(Image(filename="outputs/reliability_diagram_slake.png"))

print("\n--- VQA-RAD Reliability Diagrams ---")
display(Image(filename="outputs/reliability_diagram_vqa_rad.png"))

print("\n--- Visual Proof Sheet: SLAKE Patient 0 ---")
display(Image(filename="outputs/proofs/proof_slake_sample_0.png"))
```

---

### **Cell 8: Generate Publication Tables & Ablation Studies**
Triggers the results generator to compile and display the 7 publication-ready tables (including Main comparisons, Grounding quality, and Ablation breakdowns) for your paper.

```python
# 1. Generate tables
!python3 evaluation/result_table_generator.py

# 2. Display the main SOTA results table
with open("outputs/tables/table_1_main_comparison.md", "r") as f:
    print("=== MAIN SOTA COMPARISON TABLE ===")
    print(f.read())

print("\n" + "="*80 + "\n")

# 3. Display the module ablation table
with open("outputs/tables/ablation_1_modules.md", "r") as f:
    print("=== CORE MODULES ABLATION STUDY ===")
    print(f.read())
```
