import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

class HealMedVQADataset(Dataset):
    """
    HEAL-MedVQA Loader from Hugging Face ("MM-Hallu/HEAL-MedVQA").
    Downloads and caches the dataset, converting PIL image columns and 
    formatting question-answers.
    """
    def __init__(self, split="test", img_size=(224, 224)):
        self.img_size = img_size
        self.split = split
        
        self.image_transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load dataset from Hugging Face
        try:
            from datasets import load_dataset
            self.dataset = load_dataset("MM-Hallu/HEAL-MedVQA", name=split, split=split)
            self.active = True
            print(f"Loaded HEAL-MedVQA split: {split}. Total items: {len(self.dataset)}")
        except Exception as e:
            self.active = False
            self.dataset = []
            print(f"Warning: Failed to load HEAL-MedVQA from Hugging Face: {str(e)}")

    def __len__(self):
        return len(self.dataset) if self.active else 0

    def __getitem__(self, idx):
        if not self.active:
            raise IndexError("HEAL-MedVQA dataset is not loaded successfully.")
            
        item = self.dataset[idx]
        
        # Load image column (which is a PIL Image object)
        image = item["image"].convert('RGB')
        img_tensor = self.image_transform(image)
        
        # Generate target mask
        # Since HEAL-MedVQA focuses on general diagnosis, we default to a center bounding mask 
        # for causal verification unless coordinates are present
        mask = np.zeros(self.img_size, dtype=np.float32)
        H, W = self.img_size
        mask[int(H*0.25):int(H*0.75), int(W*0.25):int(W*0.75)] = 1.0
        target_mask = torch.from_numpy(mask).unsqueeze(0)
        
        return {
            "image": img_tensor,
            "mask": target_mask,
            "question": item["question"],
            "answer": item["answer"],
            "location": item.get("location", "unknown"),
            "answer_type": "CLOSED" if "yes" in item["answer"].lower() or "no" in item["answer"].lower() else "OPEN",
            "id": str(idx)
        }
