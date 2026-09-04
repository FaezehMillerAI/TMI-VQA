"""
Inference Adapter: Layer 1 -> Layer 2 Bridge.
Runs PyTorch forward passes on real datasets, records full prediction outputs,
and writes conforming records.jsonl + manifest.json.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from cigci_eval.records import (
    PredictionRecord,
    RunManifest,
    config_sha256,
    git_sha,
    now_utc,
    write_run,
)
from utils.config import load_config
from utils.slake_loader import SlakeCausalDataset, causal_collate_fn
from utils.vqa_rad_loader import VQARadCausalDataset
from utils.pathvqa_loader import PathVQACausalDataset
from utils.kvasir_loader import KvasirCausalDataset
from utils.vocab import load_vocab, normalize_answer
from models.cqc_net import CQCNet
from models.inpainter import CounterfactualInpainter
from models.causal_decoder import CausalContrastiveDecoder


def run_dataset_inference(
    dataset_name: str,
    split: str = "test",
    model_name: str = "ci_gci",
    inpainter_mode: str = "generative",
    seed: int = 42,
    device: str = "cpu",
    config_path: str = "configs/baseline_vqa.yaml",
    data_dir: str = "data",
    output_dir: str = "outputs/runs",
    limit_samples: int | None = None,
) -> str:
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n=======================================================")
    print(f"  RUNNING INFERENCE: {model_name} on {dataset_name.upper()} ({split}, seed={seed})")
    print(f"=======================================================")

    # 1. Dataset loading
    if dataset_name == "slake":
        json_path = os.path.join(data_dir, "slake", f"{split}.json")
        img_dir = os.path.join(data_dir, "slake", "imgs")
        mask_path = os.path.join(data_dir, "slake", "mask.txt")
        dataset = SlakeCausalDataset(json_path, img_dir, mask_path)
    elif dataset_name == "vqa_rad":
        json_path = os.path.join(data_dir, "VQA-RAD", "VQA_RAD Dataset Public.json")
        img_dir = os.path.join(data_dir, "VQA-RAD", "VQA_RAD Image Folder")
        dataset = VQARadCausalDataset(json_path, img_dir)
        # Filter closed if specified
        dataset.data = [item for item in dataset.data if item.get("answer_type") == "CLOSED"]
    elif dataset_name == "pathvqa":
        pathvqa_dir = os.path.join(data_dir, "pathvqa")
        dataset = PathVQACausalDataset(data_dir=pathvqa_dir, split=split)
        dataset.data = [item for item in dataset.data if item.get("answer_type") == "CLOSED"]
    elif dataset_name == "kvasir_x1":
        kvasir_dir = os.path.join(data_dir, "kvasir")
        dataset = KvasirCausalDataset(data_dir=kvasir_dir, split="test" if split == "test" else "train")
        dataset.data = [item for item in dataset.data if item.get("answer_type") == "CLOSED"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if limit_samples is not None and limit_samples < len(dataset):
        dataset.data = dataset.data[:limit_samples]

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=causal_collate_fn)
    print(f"Loaded {len(dataset)} evaluation samples.")

    # 2. Vocabulary & Model Setup
    vocab_path = f"models/{dataset_name}_vocab.json"
    if not os.path.exists(vocab_path):
        vocab_path = f"models/slake_vocab.json"
    ans2idx, idx2ans = load_vocab(vocab_path)
    answer_space = [idx2ans[i] for i in range(len(idx2ans))]

    config = load_config(config_path)
    config["model"]["num_classes"] = len(ans2idx)

    vqa_model = CQCNet(config).to(device)
    chk_path = f"models/{dataset_name}_vqa_model.pth"
    if not os.path.exists(chk_path):
        chk_path = "models/slake_vqa_model.pth"

    if os.path.exists(chk_path):
        print(f"Loading checkpoint: {chk_path}")
        vqa_model.load_state_dict(torch.load(chk_path, map_location=device), strict=False)
    vqa_model.eval()

    inpainter = CounterfactualInpainter(bilinear=True).to(device)
    if os.path.exists("models/inpainter.pth"):
        inpainter.load_state_dict(torch.load("models/inpainter.pth", map_location=device), strict=False)
    inpainter.eval()

    causal_decoder = CausalContrastiveDecoder(gamma=1.5).to(device)

    records: list[PredictionRecord] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            questions = batch["question"]
            answers = batch["answer"]

            # Factual Pass
            orig_out = vqa_model(images, questions, device)
            orig_logits = orig_out["main_class_logits"]
            orig_probs = torch.softmax(orig_logits, dim=-1)
            gamma_tensor = orig_out.get("gamma", torch.ones(len(questions), 1, device=device))

            if model_name == "baseline_1" or model_name == "baseline_2":
                # Baseline uses raw uncalibrated observational logits
                calib_probs = orig_probs
                cf_logits = None
            else:
                # CI-GCI uses physical counterfactual intervention
                # Supports Decisive Ablation (Task 8): generative vs black_box vs blur vs nearest
                active_mode = inpainter_mode
                if "ablation_zero" in model_name or "black_box" in model_name:
                    active_mode = "black_box"
                elif "ablation_blur" in model_name or "gaussian" in model_name:
                    active_mode = "blur"
                elif "ablation_nearest" in model_name or "nearest" in model_name:
                    active_mode = "nearest"

                if active_mode == "black_box":
                    cf_images = images * (1.0 - masks)
                elif active_mode == "blur":
                    import torchvision.transforms.functional as TF
                    blurred = TF.gaussian_blur(images, kernel_size=15, sigma=[15.0, 15.0])
                    cf_images = images * (1.0 - masks) + blurred * masks
                elif active_mode == "nearest":
                    bg_mask = 1.0 - masks
                    bg_sum = (images * bg_mask).sum(dim=(-1, -2), keepdim=True)
                    denom = bg_mask.sum(dim=(-1, -2), keepdim=True).clamp(min=1.0)
                    bg_mean = bg_sum / denom
                    cf_images = images * bg_mask + bg_mean * masks
                else:
                    cf_images = inpainter(images, masks)

                cf_out = vqa_model(cf_images, questions, device)
                cf_logits = cf_out["main_class_logits"]

                causal_out = causal_decoder(orig_logits, cf_logits, gamma=gamma_tensor)
                calib_probs = causal_out["calibrated_probs"]

            for i in range(len(questions)):
                gold_norm = normalize_answer(answers[i])
                pred_idx = torch.argmax(calib_probs[i]).item()
                pred_ans = idx2ans.get(pred_idx, "unknown")
                conf = float(calib_probs[i, pred_idx].item())
                prob_vec = [float(p) for p in calib_probs[i].cpu().tolist()]

                # Normalize prob vector to sum strictly to 1.0
                vec_sum = sum(prob_vec)
                if vec_sum > 0:
                    prob_vec = [p / vec_sum for p in prob_vec]

                rec = PredictionRecord(
                    sample_id=f"{dataset_name}_{split}_{batch_idx*8 + i}",
                    dataset=dataset_name,
                    split=split,
                    question=questions[i],
                    gold=gold_norm,
                    pred=pred_ans,
                    confidence=conf,
                    prob_vector=prob_vec,
                    answer_space=answer_space,
                    logits_orig=[float(l) for l in orig_logits[i].cpu().tolist()],
                    logits_cf=[float(l) for l in cf_logits[i].cpu().tolist()] if cf_logits is not None else None,
                    gamma=float(gamma_tensor[i].mean().item()),
                )
                records.append(rec)

    duration = time.time() - start_time
    manifest = RunManifest(
        model=model_name,
        dataset=dataset_name,
        split=split,
        seed=seed,
        n_records=len(records),
        git_sha=git_sha(require_clean=False),
        config_sha256=config_sha256(config),
        started_utc=now_utc(),
        duration_s=round(duration, 2),
        torch_version=torch.__version__,
        device=str(device),
    )

    out_file = os.path.join(output_dir, model_name, dataset_name, f"seed_{seed}", "records.jsonl")
    write_run(out_file, records, manifest)
    print(f"-> Successfully written {len(records)} records to {out_file}")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="slake", choices=["slake", "vqa_rad", "pathvqa", "kvasir_x1"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model", type=str, default="ci_gci")
    parser.add_argument("--inpainter_mode", type=str, default="generative", choices=["generative", "black_box", "blur", "nearest"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dev = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dataset_inference(
        dataset_name=args.dataset,
        split=args.split,
        model_name=args.model,
        inpainter_mode=args.inpainter_mode,
        seed=args.seed,
        device=dev,
        limit_samples=args.limit,
    )
