import os
import json
import argparse
from typing import Dict, Any, List

# Basic templates for clinical curriculum generation
MODALITY_TEMPLATES = {
    "Chest X-Ray": {
        "level_1": ["Is the lung field fully visible?", "Are the pleural spaces clear of fluid?"],
        "level_2": ["Is there any focal opacity or consolidation?", "Is the cardiothoracic ratio normal?"],
        "level_3": ["Does the radiographic appearance suggest pneumonia or effusion?"]
    },
    "Brain MRI": {
        "level_1": ["Is there abnormal signal intensity in the parenchyma?", "Are the ventricles symmetrical?"],
        "level_2": ["Is the lesion causing midline shift?", "Is the signal hyperintense on T2/FLAIR?"],
        "level_3": ["Are the findings suggestive of ischemia or demyelinating disease?"]
    },
    "CT Abdomen": {
        "level_1": ["Are the solid abdominal organs (liver, spleen, kidneys) visible?", "Is there free fluid in the peritoneum?"],
        "level_2": ["Is there wall thickening of the bowel loops?", "Is there abnormal enhancement of the pancreas?"],
        "level_3": ["Are the findings consistent with acute inflammatory process?"]
    },
    "default": {
        "level_1": ["Is the primary anatomical region clearly visualized?", "Are there obvious structural abnormalities?"],
        "level_2": ["Is the size or shape of the detected finding abnormal?", "Is there any surrounding edema or reaction?"],
        "level_3": ["Does the finding suggest a specific clinical pathology?"]
    }
}

def generate_curriculum_for_sample(sample: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Generates auxiliary questions based on modality templates."""
    mod = sample.get("modality", "default")
    templates = MODALITY_TEMPLATES.get(mod, MODALITY_TEMPLATES["default"])
    
    # In template or hybrid mode, we extract templates
    l1_qs = templates["level_1"]
    l2_qs = templates["level_2"]
    l3_qs = templates["level_3"]
    
    if mode in ["hybrid", "slm"]:
        # Adapt slightly using question keywords
        q_text = sample["question"].lower()
        if "effusion" in q_text:
            l1_qs = [q.replace("lung field", "pleural cavity") for q in l1_qs]
            l2_qs = [q.replace("focal opacity", "fluid level") for q in l2_qs]
            l3_qs = [q.replace("pneumonia", "pleural effusion") for q in l3_qs]
        elif "pneumonia" in q_text or "lung" in q_text or "clear" in q_text:
            l3_qs = [q.replace("effusion", "pneumonia") for q in l3_qs]
        elif "lesion" in q_text or "mass" in q_text or "tumor" in q_text:
            l1_qs = [q.replace("abnormal signal intensity", "mass effect or lesion") for q in l1_qs]
            l2_qs = [q.replace("midline shift", "surrounding edema") for q in l2_qs]
            l3_qs = [q.replace("ischemia", "neoplastic process") for q in l3_qs]
            
    # Simple answers mapping (defaulting to random or rule-based)
    l1_ans = ["yes" if "yes" in sample.get("answer", "") else "no" for _ in l1_qs]
    l2_ans = ["yes" if "yes" in sample.get("answer", "") else "no" for _ in l2_qs]
    l3_ans = [sample.get("answer", "no") for _ in l3_qs]
    
    return {
        "level_1": l1_qs,
        "level_2": l2_qs,
        "level_3": l3_qs,
        "answers": {
            "level_1": l1_ans,
            "level_2": l2_ans,
            "level_3": l3_ans
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    parser.add_argument("--mode", type=str, default="hybrid", choices=["template", "slm", "hybrid"])
    args = parser.parse_args()
    
    with open(args.input_data, 'r') as f:
        data = json.load(f)
        
    curriculum_data = []
    for sample in data:
        if "curriculum" not in sample or not sample["curriculum"]:
            curr = generate_curriculum_for_sample(sample, args.mode)
            sample["curriculum"] = {
                "level_1": curr["level_1"],
                "level_2": curr["level_2"],
                "level_3": curr["level_3"]
            }
            sample["curriculum_answers"] = curr["answers"]
            
        curriculum_data.append(sample)
        
    os.makedirs(os.path.dirname(args.output_data), exist_ok=True)
    with open(args.output_data, 'w') as f:
        json.dump(curriculum_data, f, indent=2)
        
    print(f"[Curriculum] Built curriculum questions for {len(curriculum_data)} samples to {args.output_data}")

if __name__ == "__main__":
    main()
