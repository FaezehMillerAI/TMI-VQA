import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import load_config
from utils.seed import seed_everything
from models.cqc_net import CQCNet
from training.dataset import CQCMedicalVQADataset, collate_fn

def train_qcg(config_path: str = None):
    config = load_config(config_path)
    seed_everything(config["seed"])
    device = torch.device(config["device"])
    print(f"[QCG Training] Using device: {device}")
    
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
    
    # Initialize the CQC-Net model
    # We set num_aux_questions to the config value
    model = CQCNet(config).to(device)
    
    # Load baseline model weights if available to get trained encoders
    baseline_path = os.path.join(config["train"]["checkpoint_dir"], "baseline", "best_baseline_model.pt")
    if os.path.exists(baseline_path):
        print(f"[QCG Training] Loading visual and text encoders from baseline checkpoint: {baseline_path}")
        state_dict = torch.load(baseline_path, map_location=device)
        # Load only visual and text encoders to keep things clean, or load state dict with strict=False
        model.load_state_dict(state_dict, strict=False)
        
    # Freeze everything except the QCG module
    for name, param in model.named_parameters():
        if "qcg" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    optimizer = optim.AdamW(
        model.qcg.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"]
    )
    
    checkpoint_dir = os.path.join(config["train"]["checkpoint_dir"], "qcg")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float("inf")
    level_criterion = nn.CrossEntropyLoss()
    
    # Target level classification targets for 5 auxiliary questions:
    # Level 1: index 0, 1 (Class 0)
    # Level 2: index 2, 3 (Class 1)
    # Level 3: index 4 (Class 2)
    level_targets = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long, device=device)
    
    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_level_correct = 0
        total_level_tokens = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config['train']['epochs']}")
        for batch in loop:
            images = batch["images"].to(device)
            questions = batch["questions"]
            aux_questions_batch = batch["aux_questions"] # [B, 5] list of strings
            
            optimizer.zero_grad()
            
            # Step 1: Forward to extract encoders features
            with torch.no_grad():
                visual_global, _ = model.visual_encoder(images)
                q_encoding = model.text_encoder(questions, device)
                q_feat = q_encoding["pooler_output"]
                
                # Get target text embeddings of auxiliary questions
                # Flatten the list of questions
                flat_aux_qs = []
                for sample_qs in aux_questions_batch:
                    flat_aux_qs.extend(sample_qs)
                aux_encoding = model.text_encoder(flat_aux_qs, device)
                target_aux_embeds = aux_encoding["pooler_output"].view(images.size(0), 5, -1) # [B, 5, T]
                
            # Step 2: Forward pass through QCG
            aux_q_embeds, level_logits = model.qcg(visual_global, q_feat) # [B, 5, T], [B, 5, 3]
            
            # Step 3: Compute Loss
            # A. Question generation embedding loss (MSE)
            loss_gen = F.mse_loss(aux_q_embeds, target_aux_embeds)
            
            # B. Level classification loss
            # Repeat level targets for batch size
            batch_level_targets = level_targets.unsqueeze(0).repeat(images.size(0), 1) # [B, 5]
            loss_level = level_criterion(level_logits.view(-1, 3), batch_level_targets.view(-1))
            
            # Combine
            loss = loss_gen + 1.0 * loss_level
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Level Accuracy calculation
            preds = torch.argmax(level_logits, dim=-1)
            total_level_correct += (preds == batch_level_targets).sum().item()
            total_level_tokens += batch_level_targets.numel()
            
            loop.set_postfix(loss=loss.item(), lvl_acc=total_level_correct/total_level_tokens)
            
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_level_correct = 0
        val_level_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                questions = batch["questions"]
                aux_questions_batch = batch["aux_questions"]
                
                visual_global, _ = model.visual_encoder(images)
                q_encoding = model.text_encoder(questions, device)
                q_feat = q_encoding["pooler_output"]
                
                flat_aux_qs = []
                for sample_qs in aux_questions_batch:
                    flat_aux_qs.extend(sample_qs)
                aux_encoding = model.text_encoder(flat_aux_qs, device)
                target_aux_embeds = aux_encoding["pooler_output"].view(images.size(0), 5, -1)
                
                aux_q_embeds, level_logits = model.qcg(visual_global, q_feat)
                
                loss_gen = F.mse_loss(aux_q_embeds, target_aux_embeds)
                batch_level_targets = level_targets.unsqueeze(0).repeat(images.size(0), 1)
                loss_level = level_criterion(level_logits.view(-1, 3), batch_level_targets.view(-1))
                
                val_loss += (loss_gen + 1.0 * loss_level).item()
                
                preds = torch.argmax(level_logits, dim=-1)
                val_level_correct += (preds == batch_level_targets).sum().item()
                val_level_total += batch_level_targets.numel()
                
        val_loss_avg = val_loss / len(val_loader)
        val_acc = val_level_correct / val_level_total if val_level_total > 0 else 0.0
        print(f"[Epoch {epoch}] Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss_avg:.4f} | Val Level Acc: {val_acc:.4f}")
        
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            checkpoint_path = os.path.join(checkpoint_dir, "best_qcg_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f" Saved new best QCG model checkpoint to {checkpoint_path}")
            
    print(f"[QCG Training] Complete. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_qcg("configs/default_config.yaml")
