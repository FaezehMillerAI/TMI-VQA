import os
import sys
import torch

# Ensure code modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.slake_loader import SlakeCausalDataset
from utils.vqa_rad_loader import VQARadCausalDataset
from utils.ms_cxr_loader import MSCXRCausalDataset
from utils.heal_loader import HealMedVQADataset
from utils.pathvqa_loader import PathVQACausalDataset
from utils.kvasir_loader import KvasirCausalDataset
from models.cqc_net import CQCNet
from models.inpainter import CounterfactualInpainter
from models.causal_decoder import CausalContrastiveDecoder

def smoke_test_dataset(name, dataset, vqa_model, inpainter, causal_decoder, device):
    print(f"\n--- Smoke Testing: {name.upper()} ---")
    print(f"Dataset Size: {len(dataset)}")
    
    if len(dataset) == 0:
        print(f"Skipping {name} (empty dataset or offline load fallback).")
        return True
        
    try:
        # Fetch one sample
        sample = dataset[0]
        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        question = [sample["question"]]
        
        # Test original forward pass
        with torch.no_grad():
            original_outputs = vqa_model(image, question, device)
            original_logits = original_outputs["main_class_logits"]
            gamma = original_outputs["gamma"]
            
            # Test inpainting
            cf_image = inpainter(image, mask)
            
            # Test counterfactual VQA pass
            cf_outputs = vqa_model(cf_image, question, device)
            cf_logits = cf_outputs["main_class_logits"]
            
            # Test CCD calibration
            causal_out = causal_decoder(original_logits, cf_logits, gamma=gamma)
            calibrated_probs = causal_out["calibrated_probs"]
            
            print(f"-> Sample Question: '{sample['question']}'")
            print(f"-> Sample Answer: '{sample['answer']}'")
            print(f"-> Dynamic Gamma: {gamma.item():.4f}")
            print(f"-> Original logits: {original_logits.cpu().numpy().tolist()}")
            print(f"-> Calibrated probs: {calibrated_probs.cpu().numpy().tolist()}")
        print(f"-> {name.upper()} Smoke Test: PASS")
        return True
    except Exception as e:
        print(f"-> {name.upper()} Smoke Test: FAIL (Error: {str(e)})")
        return False

def main():
    print("==================================================")
    print("        CI-GCI MULTI-DATASET SMOKE TEST           ")
    print("==================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Load model configurations
    config = load_config("configs/baseline_vqa.yaml")
    config["model"]["num_aux_questions"] = 0
    vqa_model = CQCNet(config).to(device)
    vqa_model.eval()
    
    inpainter = CounterfactualInpainter(bilinear=True).to(device)
    inpainter.eval()
    
    causal_decoder = CausalContrastiveDecoder(gamma=1.5).to(device)
    
    data_dir = "data/"
    results = {}
    
    # 1. SLAKE
    try:
        slake_json = os.path.join(data_dir, "slake", "test.json")
        slake_imgs = os.path.join(data_dir, "slake", "imgs")
        slake_mask = os.path.join(data_dir, "slake", "mask.txt")
        slake_dataset = SlakeCausalDataset(slake_json, slake_imgs, slake_mask)
        results["slake"] = smoke_test_dataset("slake", slake_dataset, vqa_model, inpainter, causal_decoder, device)
    except Exception as e:
        print(f"Failed to initialize SLAKE: {str(e)}")
        results["slake"] = False
        
    # 2. VQA-RAD
    try:
        rad_json = os.path.join(data_dir, "VQA-RAD", "VQA_RAD Dataset Public.json")
        rad_imgs = os.path.join(data_dir, "VQA-RAD", "VQA_RAD Image Folder")
        rad_dataset = VQARadCausalDataset(rad_json, rad_imgs)
        results["vqa_rad"] = smoke_test_dataset("vqa_rad", rad_dataset, vqa_model, inpainter, causal_decoder, device)
    except Exception as e:
        print(f"Failed to initialize VQA-RAD: {str(e)}")
        results["vqa_rad"] = False
        
    # 3. MS-CXR
    try:
        ms_json = os.path.join(data_dir, "ms-cxr", "MS_CXR_Local_Alignment_v1.1.0.json")
        ms_imgs = os.path.join(data_dir, "ms-cxr")
        ms_dataset = MSCXRCausalDataset(ms_json, ms_imgs)
        results["ms_cxr"] = smoke_test_dataset("ms_cxr", ms_dataset, vqa_model, inpainter, causal_decoder, device)
    except Exception as e:
        print(f"Failed to initialize MS-CXR: {str(e)}")
        results["ms_cxr"] = False
        
    # 4. HEAL-MedVQA (HuggingFace load check)
    try:
        heal_dataset = HealMedVQADataset(split="test")
        results["heal"] = smoke_test_dataset("heal", heal_dataset, vqa_model, inpainter, causal_decoder, device)
    except Exception as e:
        print(f"Failed to initialize HEAL-MedVQA: {str(e)}")
        results["heal"] = False

    # 5. PathVQA
    try:
        pathvqa_dataset = PathVQACausalDataset(data_dir=os.path.join(data_dir, "pathvqa"), split="test")
        results["pathvqa"] = smoke_test_dataset("pathvqa", pathvqa_dataset, vqa_model, inpainter, causal_decoder, device)
    except Exception as e:
        print(f"Failed to initialize PathVQA: {str(e)}")
        results["pathvqa"] = False

    # 6. Kvasir-VQA
    try:
        kvasir_dataset = KvasirCausalDataset(data_dir=os.path.join(data_dir, "kvasir"), split="test")
        results["kvasir"] = smoke_test_dataset("kvasir", kvasir_dataset, vqa_model, inpainter, causal_decoder, device)
    except Exception as e:
        print(f"Failed to initialize Kvasir-VQA: {str(e)}")
        results["kvasir"] = False

    print("\n==================================================")
    print("               SMOKE TEST SUMMARY                 ")
    print("==================================================")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name.upper()}: {status}")
        if not passed:
            all_pass = False
            
    if all_pass:
        print("All active smoke tests passed successfully!")
    else:
        print("Warning: Some smoke tests failed.")
    print("==================================================")

if __name__ == "__main__":
    main()
