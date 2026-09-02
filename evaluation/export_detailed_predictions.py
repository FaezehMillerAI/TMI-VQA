import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Workspace setup
workspace_dir = "/Users/faezeh/Desktop/PhD August/CQC2"
sys.path.insert(0, workspace_dir)
os.chdir(workspace_dir)

from utils.config import load_config
from utils.slake_loader import SlakeCausalDataset, causal_collate_fn
from utils.vqa_rad_loader import VQARadCausalDataset
from utils.pathvqa_loader import PathVQACausalDataset
from utils.kvasir_loader import KvasirCausalDataset
from utils.vocab import load_vocab, build_answer_vocab
from models.cqc_net import CQCNet
from models.inpainter import CounterfactualInpainter
from models.causal_decoder import CausalContrastiveDecoder

def normalize_answer(ans):
    if not isinstance(ans, str):
        ans = str(ans)
    ans = ans.lower().strip().rstrip('.').rstrip('?')
    if ans in ['true', 'yes', 'y']:
        return 'yes'
    if ans in ['false', 'no', 'n']:
        return 'no'
    return ans

def evaluate_and_export_logs(dataset_name="slake", data_dir="data", device="cpu", output_dir="outputs/logs"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=======================================================")
    print(f"  EXPORTING PER-SAMPLE PREDICTION LOGS: {dataset_name.upper()}  ")
    print(f"=======================================================")

    # 1. Load Dataset
    if dataset_name == "slake":
        json_path = os.path.join(data_dir, "slake", "test.json")
        img_dir = os.path.join(data_dir, "slake", "imgs")
        mask_path = os.path.join(data_dir, "slake", "mask.txt")
        dataset = SlakeCausalDataset(json_path, img_dir, mask_path)
    elif dataset_name == "vqa_rad":
        json_path = os.path.join(data_dir, "VQA-RAD", "VQA_RAD Dataset Public.json")
        img_dir = os.path.join(data_dir, "VQA-RAD", "VQA_RAD Image Folder")
        dataset = VQARadCausalDataset(json_path, img_dir)
    elif dataset_name == "pathvqa":
        dataset = PathVQACausalDataset(data_dir=os.path.join(data_dir, "pathvqa"), split="test")
        dataset.data = [item for item in dataset.data if item.get("answer_type") == "CLOSED"]
    elif dataset_name == "kvasir":
        dataset = KvasirCausalDataset(data_dir=os.path.join(data_dir, "kvasir"), split="test")
        dataset.data = [item for item in dataset.data if item.get("answer_type") == "CLOSED"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=causal_collate_fn)
    print(f"Loaded {len(dataset)} evaluation samples.")

    # 2. Load Vocabulary
    vocab_path = f"models/{dataset_name}_vocab.json"
    if os.path.exists(vocab_path):
        ans2idx, idx2ans = load_vocab(vocab_path)
    else:
        raw_items = dataset.data if hasattr(dataset, "data") else []
        ans2idx, idx2ans = build_answer_vocab(raw_items)

    # 3. Load Models
    config = load_config("configs/baseline_vqa.yaml")
    config["model"]["num_classes"] = max(2, len(ans2idx))
    vqa_model = CQCNet(config).to(device)

    chk_path = f"models/{dataset_name}_vqa_model.pth"
    baseline_chk = "outputs/checkpoints/baseline/best_baseline_model.pt"
    if os.path.exists(chk_path):
        print(f"Loading checkpoint from {chk_path}")
        vqa_model.load_state_dict(torch.load(chk_path, map_location=device), strict=False)
    elif os.path.exists(baseline_chk):
        print(f"Loading baseline checkpoint from {baseline_chk}")
        vqa_model.load_state_dict(torch.load(baseline_chk, map_location=device), strict=False)
    vqa_model.eval()

    inpainter = CounterfactualInpainter(bilinear=True).to(device)
    if os.path.exists("models/inpainter.pth"):
        inpainter.load_state_dict(torch.load("models/inpainter.pth", map_location=device), strict=False)
    inpainter.eval()

    causal_decoder = CausalContrastiveDecoder(gamma=1.5).to(device)

    records = []
    sample_counter = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            questions = batch["question"]
            answers = batch["answer"]

            orig_out = vqa_model(images, questions, device)
            orig_logits = orig_out["main_class_logits"]
            orig_probs = torch.softmax(orig_logits, dim=-1)
            gamma = orig_out.get("gamma", torch.ones(len(questions), 1, device=device))

            cf_images = inpainter(images, masks)
            cf_out = vqa_model(cf_images, questions, device)
            cf_logits = cf_out["main_class_logits"]
            cf_probs = torch.softmax(cf_logits, dim=-1)

            causal_out = causal_decoder(orig_logits, cf_logits, gamma=gamma)
            calib_probs = causal_out["calibrated_probs"]

            for i in range(len(questions)):
                q_text = questions[i]
                gt_raw = answers[i]
                gt_norm = normalize_answer(gt_raw)

                base_idx = torch.argmax(orig_probs[i]).item()
                base_pred = idx2ans.get(base_idx, "unknown")
                base_conf = orig_probs[i, base_idx].item()
                base_correct = int(base_pred == gt_norm)

                cf_conf = cf_probs[i, base_idx].item()
                ite_val = base_conf - cf_conf
                gamma_val = gamma[i].mean().item() if isinstance(gamma, torch.Tensor) else float(gamma)

                calib_idx = torch.argmax(calib_probs[i]).item()
                cqc_pred = idx2ans.get(calib_idx, "unknown")
                cqc_conf = calib_probs[i, calib_idx].item()
                cqc_correct = int(cqc_pred == gt_norm)

                records.append({
                    "sample_idx": sample_counter,
                    "dataset": dataset_name,
                    "question": q_text,
                    "ground_truth": gt_norm,
                    "baseline_pred": base_pred,
                    "baseline_confidence": round(base_conf, 5),
                    "baseline_correct": base_correct,
                    "cf_confidence": round(cf_conf, 5),
                    "treatment_effect_ite": round(ite_val, 5),
                    "dynamic_gamma": round(gamma_val, 4),
                    "cqc_pred": cqc_pred,
                    "cqc_confidence": round(cqc_conf, 5),
                    "cqc_correct": cqc_correct
                })
                sample_counter += 1

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, f"per_sample_predictions_{dataset_name}.csv")
    json_path = os.path.join(output_dir, f"per_sample_predictions_{dataset_name}.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=4)
    print(f"-> Saved {len(df)} per-sample rows to:\n   {csv_path}\n   {json_path}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="slake", choices=["slake", "vqa_rad", "pathvqa", "kvasir"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    dev = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    evaluate_and_export_logs(dataset_name=args.dataset, device=dev)
