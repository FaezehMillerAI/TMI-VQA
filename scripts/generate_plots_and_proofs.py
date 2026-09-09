import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure code modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.slake_loader import SlakeCausalDataset, causal_collate_fn
from utils.vqa_rad_loader import VQARadCausalDataset
from utils.ms_cxr_loader import MSCXRCausalDataset
from utils.heal_loader import HealMedVQADataset
from utils.vocab import load_vocab, build_answer_vocab
from models.cqc_net import CQCNet
from models.inpainter import CounterfactualInpainter
from models.causal_decoder import CausalContrastiveDecoder

def generate_reliability_diagram(original_confidences, calibrated_confidences, ground_truths, dataset_name, save_dir="outputs/"):
    """
    Plots Reliability Diagrams (confidence vs. accuracy) for original and calibrated models.
    """
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    orig_accs = []
    orig_confs = []
    cal_accs = []
    cal_confs = []
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Original Bin Stats
        in_bin_orig = (original_confidences >= bin_lower) & (original_confidences < bin_upper)
        if i == num_bins - 1:
            in_bin_orig = in_bin_orig | (original_confidences == bin_upper)
        if np.sum(in_bin_orig) > 0:
            orig_accs.append(np.mean(ground_truths[in_bin_orig]))
            orig_confs.append(np.mean(original_confidences[in_bin_orig]))
        else:
            orig_accs.append(0.0)
            orig_confs.append((bin_lower + bin_upper) / 2.0)
            
        # Calibrated Bin Stats
        in_bin_cal = (calibrated_confidences >= bin_lower) & (calibrated_confidences < bin_upper)
        if i == num_bins - 1:
            in_bin_cal = in_bin_cal | (calibrated_confidences == bin_upper)
        if np.sum(in_bin_cal) > 0:
            cal_accs.append(np.mean(ground_truths[in_bin_cal]))
            cal_confs.append(np.mean(calibrated_confidences[in_bin_cal]))
        else:
            cal_accs.append(0.0)
            cal_confs.append((bin_lower + bin_upper) / 2.0)
            
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Original
    axes[0].bar(bin_boundaries[:-1], orig_accs, width=1.0/num_bins, align='edge', color='red', alpha=0.6, edgecolor='red', label='Outputs')
    axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"Original VQA ({dataset_name.upper()})")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Right: Calibrated
    axes[1].bar(bin_boundaries[:-1], cal_accs, width=1.0/num_bins, align='edge', color='green', alpha=0.6, edgecolor='green', label='Outputs')
    axes[1].plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Calibrated VQA ({dataset_name.upper()})")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"reliability_diagram_{dataset_name}.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Reliability Diagram saved successfully to {save_path}")

def generate_proof_sheets(dataset, inpainter, device, dataset_name, save_dir="outputs/proofs/"):
    """
    Saves visual side-by-side columns: Original Image, ROI Mask, and Inpainted Image.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    sampled_indices = []
    for idx in range(len(dataset)):
        mask = dataset[idx]["mask"]
        if mask.sum() > 200: # Has a significant mask
            sampled_indices.append(idx)
        if len(sampled_indices) >= 3:
            break
            
    # Fallback to first three items if no matching masks
    if not sampled_indices:
        sampled_indices = list(range(min(3, len(dataset))))
        
    print(f"Generating visual proofs for {dataset_name} test indices: {sampled_indices}")
    
    for idx in sampled_indices:
        item = dataset[idx]
        image = item["image"].unsqueeze(0).to(device)
        mask = item["mask"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            cf_image = inpainter(image, mask)
            
        img_np = (image[0].cpu().numpy().transpose(1, 2, 0) * 0.229 + 0.485).clip(0, 1)
        cf_np = (cf_image[0].cpu().numpy().transpose(1, 2, 0) * 0.229 + 0.485).clip(0, 1)
        mask_np = mask[0, 0].cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original Scan ({dataset_name.upper()})")
        axes[0].axis('off')
        
        axes[1].imshow(mask_np, cmap='gray')
        location = item.get("location", "pathology")
        axes[1].set_title(f"Target Mask M\n(Location: {location})")
        axes[1].axis('off')
        
        axes[2].imshow(cf_np)
        axes[2].set_title("Inpainted Healthy Scan")
        axes[2].axis('off')
        
        save_path = os.path.join(save_dir, f"proof_{dataset_name}_sample_{idx}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Proof Sheet saved successfully to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="slake", choices=["slake", "vqa_rad", "ms_cxr", "heal"])
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load dataset
    if args.dataset == "slake":
        json_path = os.path.join(args.data_dir, "slake", "test.json")
        img_dir = os.path.join(args.data_dir, "slake", "imgs")
        mask_mapping = os.path.join(args.data_dir, "slake", "mask.txt")
        dataset = SlakeCausalDataset(json_path, img_dir, mask_mapping)
        dataset.data = [item for item in dataset.data if item.get("answer_type") == "CLOSED"]
        collate = causal_collate_fn
    elif args.dataset == "vqa_rad":
        json_path = os.path.join(args.data_dir, "VQA-RAD", "VQA_RAD Dataset Public.json")
        img_dir = os.path.join(args.data_dir, "VQA-RAD", "VQA_RAD Image Folder")
        dataset = VQARadCausalDataset(json_path, img_dir)
        dataset.data = [item for item in dataset.data if item.get("answer_type") == "CLOSED"]
        collate = causal_collate_fn
    elif args.dataset == "ms_cxr":
        json_path = os.path.join(args.data_dir, "ms-cxr", "MS_CXR_Local_Alignment_v1.1.0.json")
        img_dir = os.path.join(args.data_dir, "ms-cxr")
        dataset = MSCXRCausalDataset(json_path, img_dir)
        collate = causal_collate_fn
    elif args.dataset == "heal":
        dataset = HealMedVQADataset(split="test")
        collate = causal_collate_fn
        
    # Load answer vocabulary
    vocab_path = f"models/{args.dataset}_vocab.json"
    if not os.path.exists(vocab_path) and args.dataset in ["ms_cxr", "heal"]:
        vocab_path = "models/slake_vocab.json"
        
    if os.path.exists(vocab_path):
        ans2idx, idx2ans = load_vocab(vocab_path)
    else:
        raw_items = dataset.data if hasattr(dataset, "data") else []
        ans2idx, idx2ans = build_answer_vocab(raw_items)

    # Load models
    config = load_config("configs/baseline_vqa.yaml")
    config["model"]["num_aux_questions"] = 0
    config["model"]["num_classes"] = max(2, len(ans2idx))
    vqa_model = CQCNet(config).to(device)
    
    chk_path = f"models/{args.dataset}_vqa_model.pth"
    if not os.path.exists(chk_path) and args.dataset in ["ms_cxr", "heal"]:
        chk_path = "models/slake_vqa_model.pth"
        
    if os.path.exists(chk_path):
        try:
            vqa_model.load_state_dict(torch.load(chk_path, map_location=device), strict=False)
        except Exception as e:
            print(f"Warning: strict load failed, loading matching keys: {e}")
            vqa_model.load_state_dict(torch.load(chk_path, map_location=device), strict=False)
        
    inpainter = CounterfactualInpainter(bilinear=True).to(device)
    if os.path.exists("models/inpainter.pth"):
        inpainter.load_state_dict(torch.load("models/inpainter.pth", map_location=device))
        
    causal_decoder = CausalContrastiveDecoder(gamma=1.5).to(device)
    
    vqa_model.eval()
    inpainter.eval()
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate)
    
    original_confidences = []
    calibrated_confidences = []
    ground_truths = []
    ans_map = {"no": 0, "yes": 1}
    
    # Generate proofs
    generate_proof_sheets(dataset, inpainter, device, args.dataset)
    
    # Collect calibration statistics
    print("Collecting calibration statistics...")
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            questions = batch["question"]
            answers = batch["answer"]
            
            batch_gts = [ans_map.get(ans.strip().lower(), 0) for ans in answers]
            ground_truths.extend(batch_gts)
            
            original_outputs = vqa_model(images, questions, device)
            original_logits = original_outputs["main_class_logits"]
            gamma = original_outputs["gamma"]
            
            cf_images = inpainter(images, masks)
            cf_outputs = vqa_model(cf_images, questions, device)
            cf_logits = cf_outputs["main_class_logits"]
            
            causal_out = causal_decoder(original_logits, cf_logits, gamma=gamma)
            calibrated_probs = causal_out["calibrated_probs"]
            
            orig_probs = torch.softmax(original_logits, dim=-1)
            original_confidences.extend(orig_probs[:, 1].cpu().numpy())
            calibrated_confidences.extend(calibrated_probs[:, 1].cpu().numpy())
            
    # Reliability diagram
    generate_reliability_diagram(
        np.array(original_confidences),
        np.array(calibrated_confidences),
        np.array(ground_truths),
        args.dataset
    )

if __name__ == "__main__":
    main()
