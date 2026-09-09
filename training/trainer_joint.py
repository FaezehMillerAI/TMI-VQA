import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import torch.nn.functional as F

from utils.config import load_config
from utils.seed import seed_everything
from models.cqc_net import CQCNet
from training.dataset import CQCMedicalVQADataset, collate_fn
from training.loss_functions import CQCLoss

def train_joint(config_path: str = None):
    config = load_config(config_path)
    seed_everything(config["seed"])
    device = torch.device(config["device"])
    print(f"[Joint Training] Using device: {device}")
    
    # Load dataset
    curriculum_dir = config["data"]["curriculum_dir"]
    dataset_name = config["data"].get("dataset_name", "curriculum")
    dataset_cand = os.path.join(curriculum_dir, f"{dataset_name}_curriculum.json")
    dataset_path = dataset_cand if os.path.exists(dataset_cand) else os.path.join(curriculum_dir, "curriculum.json")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(curriculum_dir, "synthetic_curriculum.json")
    train_dataset = CQCMedicalVQADataset(dataset_path)
    train_dataset.samples = [s for s in train_dataset.samples if s["split"] == "train"]
    val_dataset = CQCMedicalVQADataset(dataset_path)
    val_dataset.samples = [s for s in val_dataset.samples if s["split"] == "val"]
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize Model
    model = CQCNet(config).to(device)
    
    # Load baseline checkpoints and QCG checkpoints if available
    baseline_path = os.path.join(config["train"]["checkpoint_dir"], "baseline", "best_baseline_model.pt")
    if os.path.exists(baseline_path):
        print(f"[Joint Training] Loading baseline model state from {baseline_path}")
        model.load_state_dict(torch.load(baseline_path, map_location=device), strict=False)
        
    qcg_path = os.path.join(config["train"]["checkpoint_dir"], "qcg", "best_qcg_model.pt")
    if os.path.exists(qcg_path):
        print(f"[Joint Training] Loading QCG model state from {qcg_path}")
        model.load_state_dict(torch.load(qcg_path, map_location=device), strict=False)
        
    # Freezing QCG during joint training of Stage 4 (per spec instructions)
    # Stage 5 can unfreeze all for end-to-end tuning.
    if model.qcg is not None:
        print("[Joint Training] Freezing QCG module for joint training stage")
        for param in model.qcg.parameters():
            param.requires_grad = False
            
    # Optimizer over the trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"]
    )
    
    criterion = CQCLoss(config)
    
    checkpoint_dir = os.path.join(config["train"]["checkpoint_dir"], "joint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float("inf")
    
    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        # Ensure frozen modules stay frozen
        if model.qcg is not None:
            model.qcg.eval()
            
        epoch_losses = {"total_loss": 0.0, "loss_qa": 0.0, "loss_ground": 0.0, "loss_cons": 0.0, "loss_hallu": 0.0, "loss_cal": 0.0}
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config['train']['epochs']}")
        for batch in loop:
            images = batch["images"].to(device)
            questions = batch["questions"]
            answer_classes = batch["answer_class"].to(device)
            grounding_box = batch["grounding_box"].to(device)
            hallucinated = batch["hallucinated"].to(device)
            aux_answers_class = batch["aux_answers_class"].to(device)
            aux_grounding_boxes = batch["aux_grounding_boxes"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images, questions, device)
            
            targets = {
                "answer_class": answer_classes,
                "grounding_box": grounding_box,
                "hallucinated": hallucinated,
                "aux_answers_class": aux_answers_class,
                "aux_grounding_boxes": aux_grounding_boxes
            }
            
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["total_loss"]
            
            loss.backward()
            optimizer.step()
            
            # Record losses
            for k in epoch_losses.keys():
                if k in loss_dict:
                    epoch_losses[k] += loss_dict[k].item()
                    
            loop.set_postfix(total_loss=loss.item(), qa=loss_dict["loss_qa"].item(), hallu=loss_dict["loss_hallu"].item())
            
        # Validation
        model.eval()
        val_losses = {k: 0.0 for k in epoch_losses.keys()}
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                questions = batch["questions"]
                answer_classes = batch["answer_class"].to(device)
                grounding_box = batch["grounding_box"].to(device)
                hallucinated = batch["hallucinated"].to(device)
                aux_answers_class = batch["aux_answers_class"].to(device)
                aux_grounding_boxes = batch["aux_grounding_boxes"].to(device)
                
                outputs = model(images, questions, device)
                targets = {
                    "answer_class": answer_classes,
                    "grounding_box": grounding_box,
                    "hallucinated": hallucinated,
                    "aux_answers_class": aux_answers_class,
                    "aux_grounding_boxes": aux_grounding_boxes
                }
                
                loss_dict = criterion(outputs, targets)
                for k in val_losses.keys():
                    if k in loss_dict:
                        val_losses[k] += loss_dict[k].item()
                        
                # Count correct predictions
                preds = torch.argmax(outputs["final_class_logits"], dim=-1)
                val_correct += (preds == answer_classes).sum().item()
                val_total += answer_classes.size(0)
                
        # Log val stats
        val_loss_avg = val_losses["total_loss"] / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        print(f"[Epoch {epoch}] Val Loss: {val_loss_avg:.4f} | Val QA: {val_losses['loss_qa']/len(val_loader):.4f} | Val Hallu: {val_losses['loss_hallu']/len(val_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            checkpoint_path = os.path.join(checkpoint_dir, "best_joint_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f" Saved new best joint model checkpoint to {checkpoint_path}")
            
    print(f"[Joint Training] Complete. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_joint("configs/joint_training.yaml")
