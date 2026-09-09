import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Add root folder to pythonpath to resolve packages properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.seed import seed_everything
from models.cqc_net import CQCNet
from training.dataset import CQCMedicalVQADataset, collate_fn
from evaluation.eval_vqa_core import compute_vqa_core_metrics
from evaluation.eval_nlg import compute_nlg_metrics
from evaluation.eval_hallucination import compute_hallucination_metrics
from evaluation.eval_calibration_grounding import compute_calibration_grounding_metrics
from evaluation.eval_qcg import compute_qcg_metrics
from evaluation.result_table_generator import generate_all_tables

def evaluate_model(config_path: str = None):
    config = load_config(config_path)
    seed_everything(config["seed"])
    device = torch.device(config["device"])
    print(f"[Evaluation] Using device: {device}")
    
    curriculum_dir = config["data"]["curriculum_dir"]
    dataset_name = config["data"].get("dataset_name", "curriculum")
    dataset_cand = os.path.join(curriculum_dir, f"{dataset_name}_curriculum.json")
    dataset_path = dataset_cand if os.path.exists(dataset_cand) else os.path.join(curriculum_dir, "curriculum.json")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(curriculum_dir, "synthetic_curriculum.json")
    test_dataset = CQCMedicalVQADataset(dataset_path)
    test_dataset.samples = [s for s in test_dataset.samples if s["split"] == "test"]
    
    # If test split is empty (e.g. random seed split variability), fallback to all samples
    if len(test_dataset) == 0:
        print("[Evaluation] Test split was empty. Using full dataset for evaluation.")
        test_dataset = CQCMedicalVQADataset(dataset_path)
        
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )
    print(f"[Evaluation] Total samples: {len(test_dataset)}")
    
    # Initialize Model
    model = CQCNet(config).to(device)
    
    # Try to load best trained model weight file
    checkpoint_path = os.path.join(config["train"]["checkpoint_dir"], "joint", "best_joint_model.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(config["train"]["checkpoint_dir"], "baseline", "best_baseline_model.pt")
        
    if os.path.exists(checkpoint_path):
        print(f"[Evaluation] Loading model checkpoint from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    else:
        print("[Evaluation] No pre-trained checkpoint found. Evaluating on randomly initialized model.")
        
    model.eval()
    
    # Storage for evaluation calculations
    all_main_preds_str = []
    all_main_gts_str = []
    all_main_probs = []
    all_main_labels = []
    all_h_scores = []
    all_gts_hallucinated = []
    all_pred_boxes = []
    all_gt_boxes = []
    all_decisions = []
    
    # QCG level evaluation
    all_generated_qs = []
    all_main_qs = []
    all_true_levels = []
    all_pred_levels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["images"].to(device)
            questions = batch["questions"]
            answer_classes = batch["answer_class"]
            grounding_boxes = batch["grounding_box"]
            hallucinated_flags = batch["hallucinated"]
            
            # Model Forward Pass
            outputs = model(images, questions, device)
            
            # 1. Main Answer outputs
            probs = torch.softmax(outputs["main_class_logits"], dim=-1)[:, 1].cpu().numpy()
            preds_class = torch.argmax(outputs["main_class_logits"], dim=-1).cpu().numpy()
            
            all_main_probs.extend(probs.tolist())
            all_main_labels.extend(answer_classes.numpy().tolist())
            
            for p, gt_c in zip(preds_class, answer_classes):
                all_main_preds_str.append("yes" if p == 1 else "no")
                all_main_gts_str.append("yes" if gt_c.item() == 1 else "no")
                
            # 2. Hallucination outputs
            h_scores = outputs["h"].cpu().numpy()
            all_h_scores.extend(h_scores.tolist())
            all_gts_hallucinated.extend(hallucinated_flags.numpy().tolist())
            
            # 3. Grounding boxes
            pred_boxes = outputs["main_region_coords"].cpu().numpy() * 224.0
            gt_boxes = grounding_boxes.numpy() * 224.0
            all_pred_boxes.extend(pred_boxes.tolist())
            all_gt_boxes.extend(gt_boxes.tolist())
            
            # 4. Decisions
            all_decisions.extend(outputs["decisions"])
            
            # 5. QCG evaluation (if QCG was active)
            if "aux_level_logits" in outputs:
                # Generate auxiliary question texts matching clinical modality
                # batch_size
                mod = batch["modalities"][0] if batch["modalities"] else "Chest X-Ray"
                batch_gen_qs = model.qcg.generate_questions_text(images.size(0), modality=mod)
                all_generated_qs.extend(batch_gen_qs)
                all_main_qs.extend(questions)
                
                # Predict levels: [B, 5, 3] -> argmax -> [B, 5]
                pred_lvls = torch.argmax(outputs["aux_level_logits"], dim=-1).cpu().numpy() # [B, 5]
                all_pred_levels.extend(pred_lvls.flatten().tolist())
                # True levels repeat pattern [0, 0, 1, 1, 2] per sample
                true_lvls = [0, 0, 1, 1, 2] * images.size(0)
                all_true_levels.extend(true_lvls)
                
    # Calculate metric subsets
    print("\nCalculating metrics...")
    vqa_core = compute_vqa_core_metrics(all_main_preds_str, all_main_gts_str)
    nlg = compute_nlg_metrics(all_main_preds_str, all_main_gts_str)
    hallu = compute_hallucination_metrics(all_h_scores, all_gts_hallucinated)
    cal_ground = compute_calibration_grounding_metrics(
        all_main_probs, all_main_labels, all_pred_boxes, all_gt_boxes, all_decisions
    )
    
    qcg_metrics = {}
    if all_generated_qs:
        qcg_metrics = compute_qcg_metrics(all_generated_qs, all_main_qs, all_true_levels, all_pred_levels)
        
    # Compile
    all_metrics = {
        "vqa_core": vqa_core,
        "nlg": nlg,
        "hallucination": hallu,
        "calibration_grounding": cal_ground,
        "qcg": qcg_metrics
    }
    
    # Save outputs
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "evaluation_metrics.json"), 'w') as f:
        json.dump(all_metrics, f, indent=2)
        
    print("\n==========================================")
    print("EVALUATION METRICS SUMMARY")
    print("==========================================")
    print(f"VQA Accuracy:             {vqa_core['accuracy']:.4f}")
    print(f"VQA Exact Match:          {vqa_core['exact_match']:.4f}")
    print(f"BLEU-4 Score:             {nlg['bleu4']:.4f}")
    print(f"ROUGE-L Score:            {nlg['rougeL']:.4f}")
    print(f"BERTScore F1:             {nlg['bertscore_f1']:.4f}")
    print(f"Hallucination Rate:       {hallu['hallucination_rate']:.4f}")
    print(f"Hallucination Detection F1: {hallu['hallucination_f1']:.4f}")
    print(f"Hallucination AUROC:       {hallu['auroc']:.4f}")
    print(f"Expected Calibration Error: {cal_ground['ece']:.4f}")
    print(f"Mean Grounding BBox IoU:   {cal_ground['mean_iou']:.4f}")
    print(f"Abstention Coverage:       {cal_ground['coverage']:.4f}")
    print(f"Selective Risk:            {cal_ground['selective_risk']:.4f}")
    if qcg_metrics:
        print(f"QCG Level Accuracy:       {qcg_metrics['level_accuracy']:.4f}")
        print(f"QCG Diversity Score:      {qcg_metrics['diversity']:.4f}")
    print("==========================================\n")
    
    # Generate LaTeX/Markdown/CSV publication tables
    generate_all_tables()
    
    print("[Evaluation] Done. Saved metrics and publication tables.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    evaluate_model(args.config)
