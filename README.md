# CI-GCI: Causal-Interventional Grounding and Counterfactual Inpainting for Calibrated Medical Visual Question Answering (CQC-Net)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/)
[![IEEE TMI](https://img.shields.io/badge/Manuscript-IEEE%20TMI-00629B.svg)](https://www.embs.org/tmi/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **CI-GCI** (CQC-Net) submitted to **IEEE Transactions on Medical Imaging (IEEE TMI)**.

---

## 🔬 Overview & Architecture

Conventional Medical Vision-Language Models (VLMs) frequently exploit dataset linguistic shortcuts ($I \leftarrow C \rightarrow A$), resulting in severe visual hallucinations ($\sim 38.5\%$) and overconfident calibration errors ($\text{ECE} \approx 18.5\%$). 

**CI-GCI** resolves this by performing **physical generative counterfactual inpainting in the pixel domain ($do(I = I \setminus \text{ROI})$)** to sever backdoor confounding and quantify the true causal Individual Treatment Effect (ITE).

![CI-GCI Architecture](Manuscript%20TMI/fig_framework.jpg)

### Core Innovations:
1. **Gaze-Guided ROI Locator (GGRL)**: Cross-attention module projecting PubMedBERT queries onto ViT visual patch tokens to locate question-conditioned anatomical lesion masks $\mathbf{M}$.
2. **Generative Counterfactual Inpainter (CFI)**: Physical generative inpainting network simulating $do(I = I \setminus \text{ROI})$, seamlessly replacing pathological tissue with healthy background anatomy ($I_{\text{cf}}$).
3. **Causal Contrastive Decoder (CCD)**: Computes the treatment effect $\text{ITE} = \mathbf{L}_{\text{orig}} - \mathbf{L}_{\text{cf}}$ scaled by a dynamic learned question-dependent factor $\gamma(Q)$, yielding calibrated interventional probabilities $\mathbf{L}_{\text{calib}} = \mathbf{L}_{\text{orig}} + \gamma(Q) \odot \text{ITE}$.
4. **Selective Abstention Triage Gate**: Decision-theoretic thresholding ($\tau$) routing verified high-confidence answers to automated output and ambiguous cases to human specialists.

---

## 📊 Empirical SOTA Benchmark Results

Evaluated across **4 multi-center medical benchmarks** spanning Radiology (SLAKE, VQA-RAD), Histopathology (PathVQA), and Endoscopy (Kvasir-VQA-x1):

| Model | VQA-RAD Acc | SLAKE Acc | PathVQA Acc | BLEU-4 | BERTScore | Halluc. Rate ↓ | AUROC | ECE (Calib Error) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline-1 (ResNet + RNN) | 0.684 | 0.702 | 0.556 | 0.245 | 0.712 | 0.385 | 0.701 | 0.1850 |
| Baseline-2 (ViT + PubMedBERT) | 0.917 | 0.814 | 0.602 | 0.301 | 0.768 | 0.294 | 0.768 | 0.0412 |
| **Proposed CI-GCI (CQC-Net)** | **0.918** | **0.812** | **0.692** | **0.412** | **0.862** | **0.108** | **0.938** | **0.0221** |

> **Key Impact**: ECE dropped by **88% relative error** (down to $2.21\%$), visual hallucination rate cut by more than two-thirds ($38.5\% \to 10.8\%$), and clinical triage achieves a **2.4% error rate at 72.5% automated coverage**.

---

## 📁 Repository Structure

```text
├── configs/                     # YAML configuration files (baseline, ablation, joint)
├── data/                        # Datasets (SLAKE, VQA-RAD, PathVQA, Kvasir-VQA, MS-CXR)
├── models/                      # Neural network components
│   ├── cqc_net.py               # Main CQC-Net model architecture
│   ├── visual_encoder.py        # Dual-scale ViT / ResNet image encoders
│   ├── text_encoder.py          # PubMedBERT / BioClinicalBERT language backbones
│   ├── inpainter.py             # Generative Counterfactual Inpainter (CFI)
│   └── causal_decoder.py        # Causal Contrastive Decoder (CCD) with dynamic γ(Q)
├── training/                    # Model training pipelines
│   ├── train_inpainter.py       # CFI generative training script
│   └── train_slake_vqa.py       # End-to-end Med-VQA fine-tuning
├── evaluation/                  # Metrics & logging suites
│   ├── export_detailed_predictions.py # Per-sample prediction CSV/JSON logger
│   └── eval_calibration_grounding.py # ECE, MCE, and Brier metrics
├── scripts/                     # Executable tools & plotting
│   ├── benchmark_comparison.py  # Automated comparative benchmarking
│   ├── generate_all_manuscript_figures.py # Generates 300 DPI publication figures
│   └── plot_exact_curves_from_logs.py     # Plots exact ROC, PR, & Risk-Coverage curves
├── demo/                        # Interactive web application
│   └── web_demo.py              # Gradio-based clinical interface
├── Manuscript TMI/              # IEEE TMI LaTeX paper, figures & supplementary
├── run_kaggle_pipeline.py       # 1-Click automated execution runner for Kaggle/Colab
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/FaezehMillerAI/TMI-VQA.git
cd TMI-VQA
pip install -r requirements.txt
```

### 2. Run 1-Click Automated Pipeline on Kaggle GPU / Local
```bash
python3 run_kaggle_pipeline.py --epochs 15 --batch_size 16
```
This automatically downloads the official Parquet datasets from Hugging Face, trains the Inpainter, fine-tunes CQC-Net across all datasets, computes benchmark metrics, and exports full per-sample prediction logs.

### 3. Plot Exact Empirical Curves from Per-Sample Logs
```bash
# Plot exact ROC, Precision-Recall, Risk-Coverage, and Reliability Diagrams
python3 scripts/plot_exact_curves_from_logs.py --dataset slake
```

### 4. Launch Interactive Clinical Web Demo
```bash
python3 demo/web_demo.py
```
Open `http://localhost:7860` in your browser to test interactive scan uploads, gaze heatmaps, inpainting, and clinical triage decisions.

---

## 📖 Citation

If you find this work useful in your research, please cite our IEEE TMI paper:

```bibtex
@article{miller2026cigci,
  title={Causal-Interventional Grounding and Generative Counterfactual Inpainting for Calibrated and Hallucination-Resistant Medical Visual Question Answering},
  author={Miller, Faezeh},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  publisher={IEEE}
}
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
