import os
import sys
import torch
import argparse
from PIL import Image, ImageDraw
import torchvision.transforms as transforms

# Add root folder to pythonpath to resolve packages properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from models.cqc_net import CQCNet

def run_inference(image_path: str, question: str, config_path: str = None, output_viz: str = "outputs/inference_grounding.png"):
    config = load_config(config_path)
    device = torch.device(config["device"])
    
    # Initialize model
    model = CQCNet(config).to(device)
    
    # Try loading trained joint checkpoint
    checkpoint_path = os.path.join(config["train"]["checkpoint_dir"], "joint", "best_joint_model.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(config["train"]["checkpoint_dir"], "baseline", "best_baseline_model.pt")
        
    if os.path.exists(checkpoint_path):
        print(f"[Inference] Loading model weights from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    else:
        print("[Inference] No checkpoint found. Running on randomly initialized model weights.")
        
    model.eval()
    
    # Load and transform image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    original_img = Image.open(image_path).convert("RGB")
    width, height = original_img.size
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(original_img).unsqueeze(0).to(device) # [1, 3, 224, 224]
    
    # Run forward pass
    with torch.no_grad():
        outputs = model(image_tensor, [question], device)
        
    # Process outputs
    # Main prediction
    main_logits = outputs["main_class_logits"][0]
    main_pred_class = torch.argmax(main_logits).item()
    main_answer = "yes" if main_pred_class == 1 else "no"
    main_grounding_score = outputs["main_grounding_score"][0].item()
    
    # Hallucination score & Decision routing
    h_score = outputs["h"][0].item()
    decision = outputs["decisions"][0]
    
    # Revised answer if any
    revised_logits = outputs["revised_class_logits"][0]
    revised_pred_class = torch.argmax(revised_logits).item()
    revised_answer = "yes" if revised_pred_class == 1 else "no"
    
    # Grounding box coordinates
    main_box = outputs["main_region_coords"][0].cpu().numpy() # [4] coordinates scaled to 224
    # Map back to original image size
    orig_x1 = int(main_box[0] * width / 224.0)
    orig_y1 = int(main_box[1] * height / 224.0)
    orig_x2 = int(main_box[2] * width / 224.0)
    orig_y2 = int(main_box[3] * height / 224.0)
    
    print("\n==========================================")
    print("CQC-NET INFERENCE RESULTS")
    print("==========================================")
    print(f"Main Question:           '{question}'")
    print(f"Original Answer Predict:  '{main_answer}' (grounding: {main_grounding_score:.4f})")
    print(f"Hallucination Risk h:     {h_score:.4f}")
    print(f"Clinical Decision Policy: {decision.upper()}")
    
    if decision == "accept":
        print(f"Final Retained Answer:   '{main_answer}'")
    elif decision == "revise":
        print(f"Final Retained Answer:   '{revised_answer}' (REVISED due to moderate inconsistency)")
    elif decision == "abstain":
        print(f"Final Retained Answer:   [ABSTAINED/FLAGGED] (High hallucination hazard!)")
    print("==========================================\n")
    
    # Modality and curriculum questions check
    if model.qcg is not None and "aux_q_embeds" in outputs:
        # Generate actual clinical text questions
        aux_qs = model.qcg.generate_questions_text(1, modality="Chest X-Ray")[0]
        aux_classes = torch.argmax(outputs["aux_class_logits"][0], dim=-1).cpu().numpy()
        aux_grounding = outputs["aux_grounding_scores"][0].cpu().numpy()
        
        print("Curriculum Auxiliary Questions:")
        print("------------------------------------------")
        levels = ["L1 - Exist/Local", "L1 - Exist/Local", "L2 - Attr/Relat", "L2 - Attr/Relat", "L3 - Clinical Inf"]
        for idx, (aq, ac, ag) in enumerate(zip(aux_qs, aux_classes, aux_grounding)):
            a_ans = "yes" if ac == 1 else "no"
            print(f" - [{levels[idx]}] Q: '{aq}' -> Answer: '{a_ans}' (grounding: {ag:.4f})")
        print("------------------------------------------\n")
        
    # Draw grounding bounding box on image
    draw = ImageDraw.Draw(original_img)
    # Check if coords are logical, draw rectangle
    draw.rectangle([orig_x1, orig_y1, orig_x2, orig_y2], outline="red", width=3)
    
    # Save the output image
    os.makedirs(os.path.dirname(output_viz), exist_ok=True)
    original_img.save(output_viz)
    print(f"[Inference] Grounding visualization saved to {output_viz}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to medical image file")
    parser.add_argument("--question", type=str, required=True, help="Main VQA query")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--output_viz", type=str, default="outputs/inference_grounding.png")
    args = parser.parse_args()
    
    run_inference(args.image, args.question, args.config, args.output_viz)
