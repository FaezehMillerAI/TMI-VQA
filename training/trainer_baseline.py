import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from utils.config import load_config
from utils.seed import seed_everything
from models.cqc_net import CQCNet
from training.dataset import CQCMedicalVQADataset, collate_fn
from training.loss_functions import CQCLoss

def train_baseline(config_path: str = None):
    # Load configuration
    config = load_config(config_path)
    seed_everything(config["seed"])
    device = torch.device(config["device"])
    print(f"[Baseline] Using device: {device}")
    
    # Paths
    curriculum_dir = config["data"]["curriculum_dir"]
    dataset_name = config["data"].get("dataset_name", "curriculum")
    dataset_cand = os.path.join(curriculum_dir, f"{dataset_name}_curriculum.json")
    dataset_path = dataset_cand if os.path.exists(dataset_cand) else os.path.join(curriculum_dir, "curriculum.json")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(curriculum_dir, "synthetic_curriculum.json")
    
    # Initialize datasets and loaders
    train_dataset = CQCMedicalVQADataset(dataset_path) # Uses internal split filtering inside train loop or dataset
    # Filter by split manually to keep dataset flexible
    train_dataset.samples = [s for s in train_dataset.samples if s["split"] == "train"]
    val_dataset = CQCMedicalVQADataset(dataset_path)
    val_dataset.samples = [s for s in val_dataset.samples if s["split"] == "val"]
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn
    )
    
    print(f"[Baseline] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize Model (disable QCG for baseline by setting num_aux_questions to 0)
    config["model"]["num_aux_questions"] = 0
    model = CQCNet(config).to(device)
    
    # Loss criterion
    # Force auxiliary weights to 0.0 for baseline training
    config["train"]["lambda_ground"] = 0.0
    config["train"]["lambda_cons"] = 0.0
    config["train"]["lambda_hallu"] = 0.0
    config["train"]["lambda_cal"] = 0.0
    criterion = CQCLoss(config)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"]
    )
    
    checkpoint_dir = config["train"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_acc = -1.0
    
    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config['train']['epochs']}")
        for batch in loop:
            images = batch["images"].to(device)
            questions = batch["questions"]
            answer_classes = batch["answer_class"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, questions, device)
            
            targets = {
                "answer_class": answer_classes
            }
            
            # Compute loss
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["total_loss"]
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Metrics
            preds = torch.argmax(outputs["main_class_logits"], dim=-1)
            correct_predictions += (preds == answer_classes).sum().item()
            total_predictions += answer_classes.size(0)
            
            loop.set_postfix(loss=loss.item(), acc=correct_predictions/total_predictions)
            
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                questions = batch["questions"]
                answer_classes = batch["answer_class"].to(device)
                
                outputs = model(images, questions, device)
                targets = {"answer_class": answer_classes}
                
                loss_dict = criterion(outputs, targets)
                val_loss += loss_dict["total_loss"].item()
                
                preds = torch.argmax(outputs["main_class_logits"], dim=-1)
                val_correct += (preds == answer_classes).sum().item()
                val_total += answer_classes.size(0)
                
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"[Epoch {epoch}] Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        # Save check point
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(checkpoint_dir, "best_baseline_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f" Saved new best model checkpoint to {checkpoint_path}")
            
    print(f"[Baseline] Training complete. Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    train_baseline("configs/baseline_vqa.yaml")
