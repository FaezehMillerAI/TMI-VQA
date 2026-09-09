import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T

class MSCXRCausalDataset(Dataset):
    def __init__(self, json_path, img_dir, img_size=(224, 224)):
        self.img_dir = img_dir
        self.img_size = img_size
        
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
            
        # Parse COCO format
        self.images = {img["id"]: img for img in coco_data["images"]}
        self.categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        
        # Group annotations by image ID
        self.annotations = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
            
        # Create a list of VQA-like samples
        self.samples = []
        for img_id, anns in self.annotations.items():
            img_meta = self.images.get(img_id)
            if not img_meta:
                continue
                
            for ann in anns:
                # Each annotation has a specific diagnostic label text
                label_text = ann.get("label_text", "")
                cat_name = self.categories.get(ann["category_id"], "pathology")
                
                # Formulate clinical question: "Does the chest X-ray indicate {pathology}?"
                question = f"Does the chest X-ray show signs of {cat_name.lower()}?"
                answer = "Yes" # Bounding boxes describe positive findings
                
                self.samples.append({
                    "img_id": img_id,
                    "file_name": img_meta["file_name"],
                    "width": img_meta["width"],
                    "height": img_meta["height"],
                    "bbox": ann["bbox"], # [x, y, w, h]
                    "question": question,
                    "answer": answer,
                    "category": cat_name,
                    "label_text": label_text
                })
                
        self.image_transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        file_name = item["file_name"]
        
        # Load image
        img_path = os.path.join(self.img_dir, file_name)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.new('RGB', self.img_size, color='black')
            
        img_tensor = self.image_transform(image)
        
        # Build binary mask from bbox coordinates
        # Bbox in COCO: [x_min, y_min, width, height]
        mask = np.zeros(self.img_size, dtype=np.float32)
        H_target, W_target = self.img_size
        
        # Original dimensions
        orig_w = item["width"]
        orig_h = item["height"]
        
        if orig_w > 0 and orig_h > 0:
            x, y, w, h = item["bbox"]
            # Scale coordinates to target image size
            x1 = int((x / orig_w) * W_target)
            y1 = int((y / orig_h) * H_target)
            x2 = int(((x + w) / orig_w) * W_target)
            y2 = int(((y + h) / orig_h) * H_target)
            
            # Constrain to dimensions
            x1 = max(0, min(W_target - 1, x1))
            y1 = max(0, min(H_target - 1, y1))
            x2 = max(0, min(W_target - 1, x2))
            y2 = max(0, min(H_target - 1, y2))
            
            mask[y1:y2, x1:x2] = 1.0
            
        target_mask = torch.from_numpy(mask).unsqueeze(0) # (1, H, W)
        
        return {
            "image": img_tensor,
            "mask": target_mask,
            "question": item["question"],
            "answer": item["answer"],
            "location": "chest",
            "category": item["category"],
            "label_text": item["label_text"],
            "answer_type": "CLOSED",
            "id": str(item["img_id"])
        }
