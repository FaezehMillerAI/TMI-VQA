import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class CQCLoss(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        train_cfg = config["train"]
        self.lambda_qa = train_cfg.get("lambda_qa", 1.0)
        self.lambda_ground = train_cfg.get("lambda_ground", 0.5)
        self.lambda_cons = train_cfg.get("lambda_cons", 0.5)
        self.lambda_hallu = train_cfg.get("lambda_hallu", 1.0)
        self.lambda_cal = train_cfg.get("lambda_cal", 0.1)
        self.delta = train_cfg.get("consistency_delta", 0.1)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        
        # 1. QA Loss (Classification cross entropy on yes/no answer)
        # target_answers: [B], class_logits: [B, num_classes]
        # target_aux_answers: [B, N], aux_class_logits: [B, N, num_classes]
        if "main_class_logits" in outputs and "answer_class" in targets:
            loss_qa_main = F.cross_entropy(outputs["main_class_logits"], targets["answer_class"])
            loss_qa = loss_qa_main
            
            # If QCG outputs are available, add auxiliary QA loss
            if "aux_class_logits" in outputs and "aux_answers_class" in targets:
                # Shape targets and logits for multi-question calculation
                aux_logits_flat = outputs["aux_class_logits"].view(-1, outputs["aux_class_logits"].size(-1))
                aux_targets_flat = targets["aux_answers_class"].view(-1)
                loss_qa_aux = F.cross_entropy(aux_logits_flat, aux_targets_flat)
                loss_qa = 0.5 * loss_qa_main + 0.5 * loss_qa_aux
                
            loss_dict["loss_qa"] = loss_qa
        else:
            loss_dict["loss_qa"] = torch.tensor(0.0, device=outputs["h"].device)
            
        # 2. Grounding & Region Attribution Loss
        # Computes MSE loss on predicted grounding bounding box coordinate targets
        if "main_region_coords" in outputs and "grounding_box" in targets:
            # grounding_box shape: [B, 4]
            loss_ground = F.mse_loss(outputs["main_region_coords"], targets["grounding_box"])
            
            if "aux_region_coords" in outputs and "aux_grounding_boxes" in targets:
                # Shape: [B, N, 4]
                loss_ground_aux = F.mse_loss(outputs["aux_region_coords"], targets["aux_grounding_boxes"])
                loss_ground = 0.5 * loss_ground + 0.5 * loss_ground_aux
                
            loss_dict["loss_ground"] = loss_ground
        else:
            loss_dict["loss_ground"] = torch.tensor(0.0, device=outputs["h"].device)
            
        # 3. Consistency Loss
        # Enforces hierarchy consistency: L1 grounding score >= L2 grounding score >= L3 grounding score
        # L_cons = sum_{l=1}^2 max(0, bar_s^{(l+1)} - bar_s^{(l)} + delta)
        # That is, grounding of higher inference level (L_l+1) should not exceed ground-level visual details (L_l) plus margin
        if "c" in outputs and outputs["c"].shape[-1] >= 3:
            c = outputs["c"] # [B, 4] containing [s_l1, s_l2, s_l3, s_0]
            s_l1 = c[:, 0]
            s_l2 = c[:, 1]
            s_l3 = c[:, 2]
            
            # Penalize when s_l2 > s_l1 + delta or s_l3 > s_l2 + delta
            cons_1 = F.relu(s_l2 - s_l1 + self.delta)
            cons_2 = F.relu(s_l3 - s_l2 + self.delta)
            loss_cons = (cons_1 + cons_2).mean()
            loss_dict["loss_cons"] = loss_cons
        else:
            loss_dict["loss_cons"] = torch.tensor(0.0, device=outputs["h"].device)
            
        # 4. Hallucination Classification Loss
        # Binary Cross Entropy on hallucination indicator h
        if "h" in outputs and "hallucinated" in targets:
            target_hallu = targets["hallucinated"].float()
            loss_hallu = F.binary_cross_entropy(outputs["h"], target_hallu)
            loss_dict["loss_hallu"] = loss_hallu
        else:
            loss_dict["loss_hallu"] = torch.tensor(0.0, device=outputs["h"].device)
            
        # 5. Calibration Loss
        # Brier score loss: mean square error between prediction probability and ground-truth classification
        if "main_class_logits" in outputs and "answer_class" in targets:
            probs = F.softmax(outputs["main_class_logits"], dim=-1)[:, 1] # Class 1 probability
            targets_brier = targets["answer_class"].float()
            loss_cal = F.mse_loss(probs, targets_brier)
            loss_dict["loss_cal"] = loss_cal
        else:
            loss_dict["loss_cal"] = torch.tensor(0.0, device=outputs["h"].device)
            
        # Combined Total Loss
        total_loss = (
            self.lambda_qa * loss_dict["loss_qa"] +
            self.lambda_ground * loss_dict["loss_ground"] +
            self.lambda_cons * loss_dict["loss_cons"] +
            self.lambda_hallu * loss_dict["loss_hallu"] +
            self.lambda_cal * loss_dict["loss_cal"]
        )
        loss_dict["total_loss"] = total_loss
        
        return loss_dict

if __name__ == "__main__":
    from utils.config import load_config
    cfg = load_config()
    criterion = CQCLoss(cfg)
    
    device = torch.device("cpu")
    outputs = {
        "h": torch.tensor([0.2, 0.8], device=device),
        "c": torch.tensor([[0.8, 0.7, 0.5, 0.9], [0.9, 0.4, 0.3, 0.2]], device=device),
        "main_class_logits": torch.tensor([[0.1, 0.9], [0.8, 0.2]], device=device),
        "aux_class_logits": torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.1, 0.9], [0.8, 0.2], [0.1, 0.9]],
                                          [[0.8, 0.2], [0.1, 0.9], [0.8, 0.2], [0.1, 0.9], [0.8, 0.2]]], device=device),
        "main_region_coords": torch.tensor([[50.0, 50.0, 100.0, 100.0], [30.0, 40.0, 80.0, 90.0]], device=device),
        "aux_region_coords": torch.rand(2, 5, 4, device=device) * 224.0
    }
    targets = {
        "answer_class": torch.tensor([1, 0], device=device),
        "aux_answers_class": torch.tensor([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], device=device),
        "grounding_box": torch.tensor([[50.0, 50.0, 100.0, 100.0], [30.0, 40.0, 80.0, 90.0]], device=device),
        "aux_grounding_boxes": torch.rand(2, 5, 4, device=device) * 224.0,
        "hallucinated": torch.tensor([0, 1], device=device)
    }
    
    losses = criterion(outputs, targets)
    print("Computed losses:")
    for k, v in losses.items():
        print(f" - {k}: {v.item():.4f}")
