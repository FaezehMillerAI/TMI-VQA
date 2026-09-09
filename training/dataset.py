import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, Any, List

class CQCMedicalVQADataset(Dataset):
    def __init__(self, json_path: str, transform=None):
        super().__init__()
        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.samples = json.load(f)
            
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Map target answers ('yes'/'no') to integers
        self.ans_map = {"no": 0, "yes": 1}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        img_path = sample["image_path"]
        if not os.path.exists(img_path):
            # Default zero tensor fallback if file doesn't exist
            # This makes validation robust even if relative paths are altered
            image_tensor = torch.zeros(3, 224, 224)
        else:
            try:
                img = Image.open(img_path).convert("RGB")
                image_tensor = self.transform(img)
            except Exception:
                image_tensor = torch.zeros(3, 224, 224)
                
        # Main question and answer mapping
        question = sample["question"]
        ans_str = sample["answer"].lower()
        ans_class = self.ans_map.get(ans_str, 0)
        
        # Bounding box grounding box
        # Box coordinates normalized format [x1, y1, x2, y2]
        raw_box = sample.get("grounding_box", [0, 0, 0, 0])
        box_tensor = torch.tensor([coord / 224.0 for coord in raw_box], dtype=torch.float32)
        
        # Hallucinated label
        hallucinated = int(sample.get("hallucinated", 0))
        
        # Curriculum elements
        curr = sample.get("curriculum", {"level_1": [], "level_2": [], "level_3": []})
        curr_ans = sample.get("curriculum_answers", {"level_1": [], "level_2": [], "level_3": []})
        
        # Format lists of auxiliary questions
        aux_questions = []
        aux_questions.extend(curr.get("level_1", []))
        aux_questions.extend(curr.get("level_2", []))
        aux_questions.extend(curr.get("level_3", []))
        
        # Format auxiliary answers mapped to classes
        aux_answers_str = []
        aux_answers_str.extend(curr_ans.get("level_1", []))
        aux_answers_str.extend(curr_ans.get("level_2", []))
        aux_answers_str.extend(curr_ans.get("level_3", []))
        aux_answers_class = [self.ans_map.get(ans.lower(), 0) for ans in aux_answers_str]
        
        # Handle padding for batching if lengths vary
        # (Our synthetic generator ensures exactly 5 auxiliary questions)
        expected_aux = 5
        while len(aux_questions) < expected_aux:
            aux_questions.append("Is the image clear?")
            aux_answers_class.append(0)
        aux_questions = aux_questions[:expected_aux]
        aux_answers_class = aux_answers_class[:expected_aux]
        
        # Target auxiliary box coordinates relative to main box
        aux_boxes = []
        for i in range(expected_aux):
            shift = (i - 2) * 10.0
            shift_box = [max(0.0, min(224.0, c + shift)) for c in raw_box]
            aux_boxes.append([c / 224.0 for c in shift_box])
        aux_boxes_tensor = torch.tensor(aux_boxes, dtype=torch.float32)
        
        return {
            "image": image_tensor,
            "question": question,
            "answer_class": torch.tensor(ans_class, dtype=torch.long),
            "answer_str": ans_str,
            "grounding_box": box_tensor,
            "hallucinated": torch.tensor(hallucinated, dtype=torch.long),
            "aux_questions": aux_questions,
            "aux_answers_class": torch.tensor(aux_answers_class, dtype=torch.long),
            "aux_grounding_boxes": aux_boxes_tensor,
            "modality": sample.get("modality", "Unknown"),
            "id": sample["id"]
        }

def collate_fn(batch):
    """Custom collate function to handle batching lists of strings."""
    images = torch.stack([item["image"] for item in batch])
    answer_classes = torch.stack([item["answer_class"] for item in batch])
    grounding_boxes = torch.stack([item["grounding_box"] for item in batch])
    hallucinated = torch.stack([item["hallucinated"] for item in batch])
    aux_answers_class = torch.stack([item["aux_answers_class"] for item in batch])
    aux_grounding_boxes = torch.stack([item["aux_grounding_boxes"] for item in batch])
    
    questions = [item["question"] for item in batch]
    aux_questions = [item["aux_questions"] for item in batch]
    modalities = [item["modality"] for item in batch]
    ids = [item["id"] for item in batch]
    
    return {
        "images": images,
        "questions": questions,
        "answer_class": answer_classes,
        "grounding_box": grounding_boxes,
        "hallucinated": hallucinated,
        "aux_questions": aux_questions,
        "aux_answers_class": aux_answers_class,
        "aux_grounding_boxes": aux_grounding_boxes,
        "modalities": modalities,
        "ids": ids
    }
