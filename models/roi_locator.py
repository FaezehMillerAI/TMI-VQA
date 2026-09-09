import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T

class GazeGuidedROILocator(nn.Module):
    """
    Gaze-Guided ROI Locator (GGRL) maps a query text to anatomical/pathological
    regions and extracts the binary region mask from the segmented index mask.
    """
    def __init__(self, mask_mapping_path, img_size=(224, 224)):
        super().__init__()
        self.img_size = img_size
        self.class_to_mask_val = {}
        
        # Load mask mapping
        if os.path.exists(mask_mapping_path):
            with open(mask_mapping_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and ":" in line:
                        val_str, name = line.split(":", 1)
                        self.class_to_mask_val[name.strip().lower()] = int(val_str)
                        
        self.mask_transform = T.Compose([
            T.Resize(self.img_size, interpolation=T.InterpolationMode.NEAREST)
        ])

    def get_matched_classes(self, question):
        q_lower = question.lower()
        matched_classes = []
        
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
                
        for class_name in self.class_to_mask_val.keys():
            if class_name in q_lower and class_name not in matched_classes:
                matched_classes.append(class_name)
                
        return list(set(matched_classes))

    def forward(self, mask_path, question):
        """
        Extracts binary mask for matched labels.
        Args:
            mask_path (str): path to the scan's mask.png
            question (str): question query
        Returns:
            torch.Tensor: (1, H, W) float32 binary mask
        """
        target_mask = torch.zeros(self.img_size, dtype=torch.float32)
        matched_labels = self.get_matched_classes(question)
        
        if os.path.exists(mask_path) and matched_labels:
            mask_img = Image.open(mask_path).convert('L')
            mask_resized = self.mask_transform(mask_img)
            mask_np = np.array(mask_resized)
            
            binary_mask_np = np.zeros_like(mask_np, dtype=np.float32)
            for cls_name in matched_labels:
                val = self.class_to_mask_val.get(cls_name)
                if val is not None:
                    binary_mask_np[mask_np == val] = 1.0
                    
            target_mask = torch.from_numpy(binary_mask_np)
            
        return target_mask.unsqueeze(0) # (1, H, W)
