import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.slake_loader import SlakeCausalDataset, causal_collate_fn
from models.inpainter import CounterfactualInpainter
from tqdm import tqdm

def train_inpainter(data_dir, epochs=5, batch_size=8, lr=1e-4, device="cpu"):
    print(f"Starting inpainter training on device: {device}", flush=True)
    
    # Paths
    json_path = os.path.join(data_dir, "train.json")
    img_dir = os.path.join(data_dir, "imgs")
    mask_mapping_path = os.path.join(data_dir, "mask.txt")
    
    # Dataset & Loader
    dataset = SlakeCausalDataset(json_path, img_dir, mask_mapping_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=causal_collate_fn)
    
    # Model
    model = CounterfactualInpainter(bilinear=True).to(device)
    model.train()
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss() # Standard L1 reconstruction loss
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        loop = tqdm(dataloader, desc=f"[Inpainter] Epoch {epoch+1}/{epochs}", file=sys.stdout, leave=True, ncols=100)
        for i, batch in enumerate(loop):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            # Zero grad
            optimizer.zero_grad()
            
            # Predict the reconstructed image
            # The model is trained to reconstruct the original image when given the masked input
            outputs = model(images, masks)
            
            # Compute loss
            loss = criterion(outputs, images)
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
            if (i + 1) % 25 == 0 or (i + 1) == len(dataloader):
                print(f"  [Inpainter] Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(dataloader)} | Loss: {loss.item():.4f}", flush=True)
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"[Inpainter] Epoch {epoch+1} Complete. Average Loss: {avg_loss:.6f}", flush=True)
        
    # Save checkpoint
    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/inpainter.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/slake/")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    
    train_inpainter(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
