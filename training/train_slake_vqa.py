import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.slake_loader import SlakeCausalDataset, causal_collate_fn
from utils.vqa_rad_loader import VQARadCausalDataset
from utils.vocab import build_answer_vocab, save_vocab, normalize_answer
from models.cqc_net import CQCNet

def train_vqa(dataset_name="slake", data_dir="data/", config_path="configs/baseline_vqa.yaml", epochs=3, batch_size=16, lr=1e-4, device="cpu"):
    print(f"Starting {dataset_name.upper()} VQA fine-tuning on device: {device}")
    
    # Load configuration
    config = load_config(config_path)
    config["model"]["num_aux_questions"] = 0 # Disable auxiliary tasks for VQA baseline
    
    # Dataset & Loader resolution
    if dataset_name == "slake":
        slake_dir = os.path.join(data_dir, "slake")
        train_json = os.path.join(slake_dir, "train.json")
        if not os.path.exists(train_json):
            print("-> SLAKE train.json not found. Creating sample dataset...")
            try:
                from scripts.prepare_synthetic_slake_data import setup_sample_slake_data
                setup_sample_slake_data()
            except Exception as err:
                print(f"-> Warning setting up sample dataset: {err}")

        val_json = None
        for candidate in ["validate.json", "val.json", "test.json", "train.json"]:
            cand_path = os.path.join(slake_dir, candidate)
            if os.path.exists(cand_path):
                val_json = cand_path
                break
                
        if val_json is None:
            val_json = train_json
            
        img_dir = os.path.join(slake_dir, "imgs")
        mask_mapping_path = os.path.join(slake_dir, "mask.txt")
        
        train_dataset = SlakeCausalDataset(train_json, img_dir, mask_mapping_path)
        train_dataset.data = [item for item in train_dataset.data if item.get("answer_type") == "CLOSED"]
        
        val_dataset = SlakeCausalDataset(val_json, img_dir, mask_mapping_path)
        val_dataset.data = [item for item in val_dataset.data if item.get("answer_type") == "CLOSED"]
        collate = causal_collate_fn
        raw_train_items = train_dataset.data
        
    elif dataset_name == "vqa_rad":
        rad_dir = os.path.join(data_dir, "VQA-RAD")
        json_candidates = [
            "VQA_RAD Dataset Public.json",
            "vqa_rad.json",
            "train.json",
            "VQA_RAD_Dataset_Public.json"
        ]
        
        json_path = None
        for candidate in json_candidates:
            cand_path = os.path.join(rad_dir, candidate)
            if os.path.exists(cand_path):
                json_path = cand_path
                break
                
        if json_path is None:
            print("-> VQA-RAD dataset JSON not found. Creating sample dataset...")
            try:
                from scripts.prepare_synthetic_vqa_rad_data import setup_sample_vqa_rad_data
                setup_sample_vqa_rad_data()
                json_path = os.path.join(rad_dir, "VQA_RAD Dataset Public.json")
            except Exception as err:
                print(f"-> Warning setting up VQA-RAD sample dataset: {err}")
                json_path = os.path.join(rad_dir, "VQA_RAD Dataset Public.json")
                
        img_dir = os.path.join(rad_dir, "VQA_RAD Image Folder")
        
        # Load entire closed split and split 80/20 randomly
        full_dataset = VQARadCausalDataset(json_path, img_dir)
        full_dataset.data = [item for item in full_dataset.data if item.get("answer_type") == "CLOSED"]
        
        # Deterministic random split
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(full_dataset), generator=g).tolist()
        split_idx = int(len(indices) * 0.8)
        
        train_dataset = torch.utils.data.Subset(full_dataset, indices[:split_idx])
        val_dataset = torch.utils.data.Subset(full_dataset, indices[split_idx:])
        collate = causal_collate_fn
        raw_train_items = [full_dataset.data[i] for i in indices[:split_idx]]

    elif dataset_name == "pathvqa":
        pathvqa_dir = os.path.join(data_dir, "pathvqa")
        from utils.pathvqa_loader import PathVQACausalDataset
        train_dataset = PathVQACausalDataset(data_dir=pathvqa_dir, split="train")
        train_dataset.data = [item for item in train_dataset.data if item.get("answer_type") == "CLOSED"]

        val_dataset = PathVQACausalDataset(data_dir=pathvqa_dir, split="val")
        val_dataset.data = [item for item in val_dataset.data if item.get("answer_type") == "CLOSED"]
        collate = causal_collate_fn
        raw_train_items = train_dataset.data

    elif dataset_name == "kvasir":
        kvasir_dir = os.path.join(data_dir, "kvasir")
        from utils.kvasir_loader import KvasirCausalDataset
        train_dataset = KvasirCausalDataset(data_dir=kvasir_dir, split="train")
        train_dataset.data = [item for item in train_dataset.data if item.get("answer_type") == "CLOSED"]

        val_dataset = KvasirCausalDataset(data_dir=kvasir_dir, split="test")
        val_dataset.data = [item for item in val_dataset.data if item.get("answer_type") == "CLOSED"]
        collate = causal_collate_fn
        raw_train_items = train_dataset.data

    # Build vocabulary for exact candidate classification
    ans2idx, idx2ans = build_answer_vocab(raw_train_items)
    save_vocab(ans2idx, idx2ans, f"models/{dataset_name}_vocab.json")
    config["model"]["num_classes"] = max(2, len(ans2idx))
    print(f"Loaded {len(train_dataset)} train and {len(val_dataset)} val samples with {len(ans2idx)} answer classes.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    
    # Initialize Model
    model = CQCNet(config).to(device)
    
    # We load pre-trained weights if available to speed up convergence
    baseline_chk = "outputs/checkpoints/baseline/best_baseline_model.pt"
    if os.path.exists(baseline_chk):
        try:
            print(f"Initializing with pre-trained visual/text encoders from {baseline_chk}")
            model.load_state_dict(torch.load(baseline_chk, map_location=device), strict=False)
        except Exception as e:
            print(f"Skipping legacy baseline checkpoint due to dimension mismatch: {e}")
        
    # Set up dual-rate optimization: low LR for pre-trained backbones, higher LR for fusion & classification heads
    model.train()
    
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if "visual_encoder" in name or "text_encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.05},
        {"params": head_params, "lr": lr}
    ], weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            images = batch["image"].to(device)
            questions = batch["question"]
            answers = batch["answer"]
            
            # Convert labels to tensor using dynamic vocabulary
            labels = torch.tensor([ans2idx.get(normalize_answer(ans), 0) for ans in answers], dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            outputs = model(images, questions, device)
            logits = outputs["main_class_logits"]
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix(loss=loss.item(), acc=correct/total)
            
        epoch_loss = epoch_loss / total
        train_acc = correct / total
        
        # Validation loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                questions = batch["question"]
                answers = batch["answer"]
                
                labels = torch.tensor([ans2idx.get(normalize_answer(ans), 0) for ans in answers], dtype=torch.long, device=device)
                outputs = model(images, questions, device)
                logits = outputs["main_class_logits"]
                
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            # Save specific dataset checkpoint
            checkpoint_path = f"models/{dataset_name}_vqa_model.pth"
            # Fallback copy for backward compatibility
            if dataset_name == "slake":
                torch.save(model.state_dict(), "models/slake_vqa_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path} with Val Acc: {val_acc:.4f}")
            
        model.train()
        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="slake", choices=["slake", "vqa_rad", "pathvqa", "kvasir"])
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--config_path", type=str, default="configs/baseline_vqa.yaml")
    parser.add_argument("--epochs", type=str, default="3")
    parser.add_argument("--batch_size", type=str, default="16")
    parser.add_argument("--lr", type=str, default="1e-4")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    
    train_vqa(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        config_path=args.config_path,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=args.device
    )
