import os
import io
import json
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T
import pandas as pd

class VQARadCausalDataset(Dataset):
    def __init__(self, json_path=None, img_dir=None, data_dir=None, split="train", img_size=(224, 224)):
        self.img_dir = img_dir
        self.img_size = img_size
        self.data = []

        # 1. Check if data_dir or json_path has parquet files
        target_dir = data_dir or (os.path.dirname(json_path) if json_path and os.path.exists(json_path) else None) or "data/VQA-RAD"
        parquet_files = glob.glob(os.path.join(target_dir, f"*{split}*.parquet")) + glob.glob(os.path.join(target_dir, "**", f"*{split}*.parquet"), recursive=True)
        
        if parquet_files:
            for p_file in parquet_files:
                df = pd.read_parquet(p_file)
                for _, row in df.iterrows():
                    self.data.append({
                        "image": row.get("image"),
                        "question": str(row.get("question", "")),
                        "answer": str(row.get("answer", "")),
                        "image_organ": str(row.get("image_organ", "chest")),
                        "answer_type": str(row.get("answer_type", "CLOSED")).upper(),
                        "qid": str(row.get("qid", len(self.data)))
                    })
        elif json_path and os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

        self.image_transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def _generate_organ_mask(self, organ):
        """
        Generates a physics-informed anatomical template mask when pixel segmentations 
        are not present (e.g. VQA-RAD).
        """
        mask = np.zeros(self.img_size, dtype=np.float32)
        H, W = self.img_size
        organ = str(organ).upper()
        
        if "HEAD" in organ or "BRAIN" in organ:
            y, x = np.ogrid[:H, :W]
            center_y, center_x = H // 2, W // 2
            rx, ry = W // 3, H // 3
            mask[((x - center_x)/rx)**2 + ((y - center_y)/ry)**2 <= 1] = 1.0
        elif "CHEST" in organ or "LUNG" in organ:
            mask[int(H*0.2):int(H*0.8), int(W*0.15):int(W*0.45)] = 1.0
            mask[int(H*0.2):int(H*0.8), int(W*0.55):int(W*0.85)] = 1.0
        elif "ABDOMEN" in organ:
            mask[int(H*0.3):int(H*0.8), int(W*0.25):int(W*0.75)] = 1.0
        else:
            mask[int(H*0.25):int(H*0.75), int(W*0.25):int(W*0.75)] = 1.0
            
        return torch.from_numpy(mask).unsqueeze(0) # (1, H, W)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_val = item.get("image")
        image = None

        if isinstance(image_val, dict) and "bytes" in image_val and image_val["bytes"] is not None:
            image = Image.open(io.BytesIO(image_val["bytes"])).convert("RGB")
        elif isinstance(image_val, bytes):
            image = Image.open(io.BytesIO(image_val)).convert("RGB")
        elif isinstance(image_val, Image.Image):
            image = image_val.convert("RGB")
        elif "image_name" in item and self.img_dir:
            img_path = os.path.join(self.img_dir, item["image_name"])
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')

        if image is None:
            image = Image.new('RGB', self.img_size, color='black')

        img_tensor = self.image_transform(image)
        organ = item.get("image_organ", "chest")
        target_mask = self._generate_organ_mask(organ)

        return {
            "image": img_tensor,
            "mask": target_mask,
            "question": item["question"],
            "answer": item["answer"],
            "location": organ,
            "answer_type": item.get("answer_type", "CLOSED"),
            "id": item.get("qid", str(idx))
        }
