import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T

class SlakeCausalDataset(Dataset):
    def __init__(self, json_path, img_dir, mask_mapping_path, img_size=(224, 224), split="train"):
        self.img_dir = img_dir
        self.img_size = img_size
        self.split = split
        
        # Load VQA annotations
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # Filter for English questions to simplify the prototype
        self.data = [item for item in self.data if item.get("q_lang") == "en"]
        
        # Parse mask index mapping
        self.mask_val_to_class = {}
        self.class_to_mask_val = {}
        if os.path.exists(mask_mapping_path):
            with open(mask_mapping_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and ":" in line:
                        val_str, name = line.split(":", 1)
                        val = int(val_str)
                        self.mask_val_to_class[val] = name.strip()
                        self.class_to_mask_val[name.strip().lower()] = val

        # Define basic transformations
        self.image_transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = T.Compose([
            T.Resize(self.img_size, interpolation=T.InterpolationMode.NEAREST)
        ])

    def _get_target_classes(self, question):
        """
        Maps words in the VQA question to anatomical or pathology classes.
        """
        q_lower = question.lower()
        matched_classes = []
        
        # Keywords mapping to mask labels
        mapping_rules = {
            "lung": ["left lung", "right lung", "lung cancer"],
            "liver": ["liver", "liver cancer"],
            "kidney": ["left kidney", "right kidney", "kidney cancer"],
            "heart": ["heart"],
            "spleen": ["spleen"],
            "stomach": ["stomach"],
            "bladder": ["bladder"],
            "spinal": ["spinal cord"],
            "mandible": ["left mandible", "right mandible"],
            "temporal": ["left temporal lobe", "right temporal lobe"],
            "tumor": ["brain enhancing tumor", "brain non-enhancing tumor", "liver cancer", "lung cancer", "kidney cancer"],
            "cancer": ["brain enhancing tumor", "brain non-enhancing tumor", "liver cancer", "lung cancer", "kidney cancer"],
            "edema": ["brain edema"]
        }
        
        for key, classes in mapping_rules.items():
            if key in q_lower:
                matched_classes.extend(classes)
                
        # Fallback to direct matching if any class label appears in the question
        for class_name in self.class_to_mask_val.keys():
            if class_name in q_lower and class_name not in matched_classes:
                matched_classes.append(class_name)
                
        return list(set(matched_classes))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item["img_name"]
        
        # Paths
        img_path = os.path.join(self.img_dir, img_name)
        # Slices in SLAKE store the mask in the same folder as source.jpg
        mask_path = os.path.join(self.img_dir, os.path.dirname(img_name), "mask.png")
        
        # Load image
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            # Fallback placeholder image
            image = Image.new('RGB', self.img_size, color='black')
            
        # Transform image
        img_tensor = self.image_transform(image)
        
        # Load and build target mask
        target_mask = torch.zeros(self.img_size, dtype=torch.float32)
        matched_labels = self._get_target_classes(item["question"])
        
        if os.path.exists(mask_path) and matched_labels:
            mask_img = Image.open(mask_path).convert('L')
            mask_resized = self.mask_transform(mask_img)
            mask_np = np.array(mask_resized)
            
            # Aggregate binary mask for all matched classes
            binary_mask_np = np.zeros_like(mask_np, dtype=np.float32)
            for cls_name in matched_labels:
                val = self.class_to_mask_val.get(cls_name)
                if val is not None:
                    binary_mask_np[mask_np == val] = 1.0
            
            target_mask = torch.from_numpy(binary_mask_np).unsqueeze(0) # (1, H, W)
        else:
            # If no specific organ matched, return a 0-filled mask
            target_mask = target_mask.unsqueeze(0)
            
        return {
            "image": img_tensor,
            "mask": target_mask,
            "question": item["question"],
            "answer": item["answer"],
            "location": item.get("location", "unknown"),
            "modality": item.get("modality", "unknown"),
            "answer_type": item.get("answer_type", "OPEN"),
            "content_type": item.get("content_type", "unknown"),
            "matched_classes": matched_labels
        }

def causal_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    locations = [item.get("location", "unknown") for item in batch]
    modalities = [item.get("modality", "unknown") for item in batch]
    answer_types = [item.get("answer_type", "OPEN") for item in batch]
    content_types = [item.get("content_type", "unknown") for item in batch]
    matched_classes = [item.get("matched_classes", []) for item in batch]
    
    return {
        "image": images,
        "mask": masks,
        "question": questions,
        "answer": answers,
        "location": locations,
        "modality": modalities,
        "answer_type": answer_types,
        "content_type": content_types,
        "matched_classes": matched_classes
    }

