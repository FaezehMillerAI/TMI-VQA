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


class KvasirCausalDataset(Dataset):
    """
    Kvasir-VQA Causal Dataset loader that reads official Parquet splits (train, test),
    decodes gastrointestinal endoscopic images, applies vision-language transforms, and produces
    question-conditioned endoscopic intervention masks for counterfactual inpainting.
    """
    def __init__(self, data_dir="data/kvasir", split="train", img_size=(224, 224)):
        self.data_dir = data_dir
        self.split = split.lower()
        self.img_size = img_size
        self.data = []

        # Find matching parquet files
        if self.split in ["train", "training"]:
            pattern = os.path.join(self.data_dir, "train-*.parquet")
        elif self.split in ["test", "testing", "val", "validation"]:
            pattern = os.path.join(self.data_dir, "test-*.parquet")
        else:
            pattern = os.path.join(self.data_dir, "*.parquet")

        parquet_files = sorted(glob.glob(pattern))
        
        # Fallback to Kvasir root folder if not in data/kvasir
        if not parquet_files:
            alt_dir = "Kvasir" if not os.path.exists(self.data_dir) else self.data_dir
            if self.split in ["train", "training"]:
                pattern = os.path.join(alt_dir, "train-*.parquet")
            elif self.split in ["test", "testing", "val", "validation"]:
                pattern = os.path.join(alt_dir, "test-*.parquet")
            else:
                pattern = os.path.join(alt_dir, "*.parquet")
            parquet_files = sorted(glob.glob(pattern))

        if parquet_files and HAS_PARQUET:
            print(f"[Kvasir-VQA] Loading {len(parquet_files)} parquet files for split '{self.split}'...")
            for pfile in parquet_files:
                try:
                    df = pd.read_parquet(pfile)
                    for _, row in df.iterrows():
                        ans_str = str(row.get("answer", "")).strip()
                        ans_lower = ans_str.lower()
                        is_closed = ans_lower in ["yes", "no", "true", "false", "polyp", "normal", "adenoma", "cecum", "colon", "esophagus", "stomach"]
                        
                        item = {
                            "image_raw": row["image"],
                            "question": str(row["question"]).strip(),
                            "answer": ans_str,
                            "answer_type": "CLOSED" if is_closed else "OPEN",
                            "modality": "Endoscopy",
                            "location": "Gastrointestinal Tract"
                        }
                        self.data.append(item)
                except Exception as e:
                    print(f"[Kvasir-VQA] Warning reading {pfile}: {e}")
        else:
            print(f"[Kvasir-VQA] Warning: No parquet files found matching '{pattern}'. Creating fallback mock samples.")
            for i in range(20):
                self.data.append({
                    "image_raw": None,
                    "question": "Is there a polyp or mucosal abnormality in the field of view?",
                    "answer": "yes" if i % 2 == 0 else "no",
                    "answer_type": "CLOSED",
                    "modality": "Endoscopy",
                    "location": "Gastrointestinal Tract"
                })

        print(f"[Kvasir-VQA] Successfully initialized {len(self.data)} samples for split '{self.split}'.")

        # Vision Transforms
        self.image_transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def _decode_image(self, img_raw):
        if img_raw is None:
            return Image.new("RGB", self.img_size, color=(180, 50, 50)) # Endoscopic red/mucosal hue
        
        if isinstance(img_raw, Image.Image):
            return img_raw.convert("RGB")
            
        if isinstance(img_raw, dict):
            if "bytes" in img_raw and img_raw["bytes"] is not None:
                return Image.open(io.BytesIO(img_raw["bytes"])).convert("RGB")
            elif "path" in img_raw and img_raw["path"] and os.path.exists(img_raw["path"]):
                return Image.open(img_raw["path"]).convert("RGB")
                
        if isinstance(img_raw, bytes):
            return Image.open(io.BytesIO(img_raw)).convert("RGB")

        return Image.new("RGB", self.img_size, color=(180, 60, 60))

    def _generate_endoscopy_mask(self, question):
        """
        Generates a physics-informed gastrointestinal polyp/mucosal lesion mask for counterfactual
        intervention in endoscopic images.
        """
        mask = np.zeros(self.img_size, dtype=np.float32)
        H, W = self.img_size
        q_lower = question.lower()

        if any(k in q_lower for k in ["polyp", "adenoma", "sessile", "growth", "tumor"]):
            # Focal circular/elliptical polyp candidate mask in central lumen
            y, x = np.ogrid[:H, :W]
            center_y, center_x = int(H * 0.48), int(W * 0.52)
            rx, ry = W // 4, H // 4
            mask[((x - center_x)/rx)**2 + ((y - center_y)/ry)**2 <= 1] = 1.0
        elif any(k in q_lower for k in ["ulcer", "bleeding", "erythema", "inflammation", "angiodysplasia"]):
            # Mucosal wall lesion band
            mask[int(H*0.25):int(H*0.75), int(W*0.15):int(W*0.85)] = 1.0
        elif any(k in q_lower for k in ["instrument", "forcep", "snare", "catheter", "tool"]):
            # Endoscope instrument mask coming from lower edge
            mask[int(H*0.6):int(H), int(W*0.4):int(W*0.6)] = 1.0
        else:
            # Central luminal circular mask
            y, x = np.ogrid[:H, :W]
            center_y, center_x = H // 2, W // 2
            rx, ry = W // 3, H // 3
            mask[((x - center_x)/rx)**2 + ((y - center_y)/ry)**2 <= 1] = 1.0

        return torch.from_numpy(mask).unsqueeze(0) # (1, H, W)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_raw = item["image_raw"]
        pil_img = self._decode_image(img_raw)
        img_tensor = self.image_transform(pil_img)
        mask_tensor = self._generate_endoscopy_mask(item["question"])

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "question": item["question"],
            "answer": item["answer"],
            "answer_type": item["answer_type"],
            "modality": item["modality"],
            "location": item["location"]
        }
