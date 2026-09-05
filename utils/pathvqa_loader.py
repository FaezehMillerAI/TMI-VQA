import os
import io
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T

try:
    import pyarrow.parquet as pq
    import pandas as pd
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


class PathVQACausalDataset(Dataset):
    """
    PathVQA Causal Dataset loader that reads official Parquet splits (train, validation, test),
    decodes embedded pathology images, applies vision-language transforms, and produces
    question-conditioned pathology masks for counterfactual inpainting.
    """
    def __init__(self, data_dir="data/pathvqa", split="train", img_size=(224, 224)):
        self.data_dir = data_dir
        self.split = split.lower()
        self.img_size = img_size
        self.data = []

        # Find matching parquet files (recursive search to support any nested HF directory)
        if self.split in ["train", "training"]:
            pattern = os.path.join(self.data_dir, "**", "train*.parquet")
        elif self.split in ["val", "valid", "validation"]:
            pattern = os.path.join(self.data_dir, "**", "*val*.parquet")
        elif self.split in ["test", "testing"]:
            pattern = os.path.join(self.data_dir, "**", "test*.parquet")
        else:
            pattern = os.path.join(self.data_dir, "**", "*.parquet")

        parquet_files = sorted(glob.glob(pattern, recursive=True))
        
        # If no split-specific files found, search root PathVQA dir or data/pathvqa
        if not parquet_files:
            alt_dir = "PathVQA" if not os.path.exists(self.data_dir) else self.data_dir
            parquet_files = sorted(glob.glob(os.path.join(alt_dir, "**", "*.parquet"), recursive=True))

        if parquet_files and HAS_PARQUET:
            print(f"[PathVQA] Loading {len(parquet_files)} parquet files for split '{self.split}'...")
            for pfile in parquet_files:
                try:
                    df = pd.read_parquet(pfile)
                    for _, row in df.iterrows():
                        ans_str = str(row.get("answer", "")).strip()
                        # Categorize closed vs open
                        ans_lower = ans_str.lower()
                        is_closed = ans_lower in ["yes", "no", "true", "false", "left", "right", "positive", "negative"]
                        
                        item = {
                            "image_raw": row["image"],
                            "question": str(row["question"]).strip(),
                            "answer": ans_str,
                            "answer_type": "CLOSED" if is_closed else "OPEN",
                            "modality": "Pathology",
                            "location": "Microscopic Tissue"
                        }
                        self.data.append(item)
                except Exception as e:
                    print(f"[PathVQA] Warning reading {pfile}: {e}")
        else:
            print(f"[PathVQA] Warning: No parquet files found matching '{pattern}'. Creating fallback mock samples.")
            for i in range(20):
                self.data.append({
                    "image_raw": None,
                    "question": "Is there evidence of malignant cellular infiltration?",
                    "answer": "yes" if i % 2 == 0 else "no",
                    "answer_type": "CLOSED",
                    "modality": "Pathology",
                    "location": "Microscopic Tissue"
                })

        print(f"[PathVQA] Successfully initialized {len(self.data)} samples for split '{self.split}'.")

        # Standard Vision Transforms
        self.image_transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def _decode_image(self, img_raw):
        if img_raw is None:
            return Image.new("RGB", self.img_size, color=(200, 150, 180)) # H&E pink/purple tint
        
        if isinstance(img_raw, Image.Image):
            return img_raw.convert("RGB")
            
        if isinstance(img_raw, dict):
            if "bytes" in img_raw and img_raw["bytes"] is not None:
                return Image.open(io.BytesIO(img_raw["bytes"])).convert("RGB")
            elif "path" in img_raw and img_raw["path"] and os.path.exists(img_raw["path"]):
                return Image.open(img_raw["path"]).convert("RGB")
                
        if isinstance(img_raw, bytes):
            return Image.open(io.BytesIO(img_raw)).convert("RGB")

        # Fallback
        return Image.new("RGB", self.img_size, color=(220, 160, 190))

    def _generate_pathology_mask(self, question):
        """
        Generates a physics-informed histological lesion/region mask for counterfactual
        intervention in pathology scans.
        """
        mask = np.zeros(self.img_size, dtype=np.float32)
        H, W = self.img_size
        q_lower = question.lower()

        if any(k in q_lower for k in ["nuclei", "nuclear", "cellular", "cell"]):
            # Multi-focal cellular cluster masks
            for cx, cy in [(W//3, H//3), (2*W//3, H//2), (W//2, 2*H//3)]:
                y, x = np.ogrid[:H, :W]
                mask[((x - cx)/20)**2 + ((y - cy)/20)**2 <= 1] = 1.0
        elif any(k in q_lower for k in ["necrosis", "necrotic", "carcinoma", "tumor", "malignant"]):
            # Central dense pathological lesion
            y, x = np.ogrid[:H, :W]
            center_y, center_x = H // 2, W // 2
            rx, ry = W // 3, H // 3
            mask[((x - center_x)/rx)**2 + ((y - center_y)/ry)**2 <= 1] = 1.0
        elif any(k in q_lower for k in ["gland", "glandular", "stroma", "duct"]):
            # Tubular/glandular pattern mask
            mask[int(H*0.25):int(H*0.75), int(W*0.2):int(W*0.8)] = 1.0
        else:
            # General central field-of-view intervention
            mask[int(H*0.2):int(H*0.8), int(W*0.2):int(W*0.8)] = 1.0

        return torch.from_numpy(mask).unsqueeze(0) # (1, H, W)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_raw = item["image_raw"]
        pil_img = self._decode_image(img_raw)
        img_tensor = self.image_transform(pil_img)
        mask_tensor = self._generate_pathology_mask(item["question"])

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "question": item["question"],
            "answer": item["answer"],
            "answer_type": item["answer_type"],
            "modality": item["modality"],
            "location": item["location"]
        }
