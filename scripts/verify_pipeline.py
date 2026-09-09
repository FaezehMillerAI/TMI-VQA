import os
import sys
import torch

# Ensure code modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.slake_loader import SlakeCausalDataset
from models.roi_locator import GazeGuidedROILocator
from models.inpainter import CounterfactualInpainter
from models.causal_decoder import CausalContrastiveDecoder

def run_pipeline_check():
    print("==================================================")
    print("       CI-GCI AUTOMATED PIPELINE VERIFICATION     ")
    print("==================================================")
    
    data_dir = "data/slake/"
    json_path = os.path.join(data_dir, "train.json")
    img_dir = os.path.join(data_dir, "imgs")
    mask_mapping_path = os.path.join(data_dir, "mask.txt")
    
    # --------------------------------------------------
    # Step 1: Verification of Data Loader & Parsing
    # --------------------------------------------------
    print("\n[Step 1/5] Verifying Data Loader...")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"SLAKE train.json not found at '{json_path}'.\n"
            "Please download SLAKE or run 'python run_kaggle_pipeline.py' to prepare datasets."
        )

    try:
        dataset = SlakeCausalDataset(json_path, img_dir, mask_mapping_path)
        print(f"-> Successfully loaded dataset. Total samples: {len(dataset)}")
        
        # Pull a sample
        sample = dataset[0]
        assert "image" in sample, "Missing 'image' key in dataset item"
        assert "mask" in sample, "Missing 'mask' key in dataset item"
        assert "question" in sample, "Missing 'question' key in dataset item"
        assert "answer" in sample, "Missing 'answer' key in dataset item"
        
        # Verify shapes
        assert sample["image"].shape == (3, 224, 224), f"Incorrect image shape: {sample['image'].shape}"
        assert sample["mask"].shape == (1, 224, 224), f"Incorrect mask shape: {sample['mask'].shape}"
        print("-> Data Loader Verification: PASS")
    except Exception as e:
        print(f"-> Data Loader Verification: FAIL ({str(e)})")
        sys.exit(1)
        
    # --------------------------------------------------
    # Step 2: Verification of Gaze-Guided ROI Locator
    # --------------------------------------------------
    print("\n[Step 2/5] Verifying Gaze-Guided ROI Locator (GGRL)...")
    try:
        locator = GazeGuidedROILocator(mask_mapping_path)
        test_q = "Does the picture contain liver?"
        test_mask_path = os.path.join(img_dir, "xmlab1", "mask.png")
        
        matched_classes = locator.get_matched_classes(test_q)
        print(f"-> Question: '{test_q}' mapped to: {matched_classes}")
        assert "liver" in matched_classes, "Failed to map 'liver' keyword"
        
        extracted_mask = locator(test_mask_path, test_q)
        assert extracted_mask.shape == (1, 224, 224), f"Incorrect locator mask shape: {extracted_mask.shape}"
        print("-> ROI Locator Verification: PASS")
    except Exception as e:
        print(f"-> ROI Locator Verification: FAIL ({str(e)})")
        sys.exit(1)
        
    # --------------------------------------------------
    # Step 3: Verification of Counterfactual Inpainter
    # --------------------------------------------------
    print("\n[Step 3/5] Verifying Counterfactual Inpainter (CFI)...")
    try:
        inpainter = CounterfactualInpainter(bilinear=True)
        img_batch = sample["image"].unsqueeze(0) # (1, 3, 224, 224)
        mask_batch = sample["mask"].unsqueeze(0)   # (1, 1, 224, 224)
        
        output_batch = inpainter(img_batch, mask_batch)
        assert output_batch.shape == (1, 3, 224, 224), f"Incorrect output shape: {output_batch.shape}"
        
        # Verify background preservation check:
        # The background outside the mask MUST be identical to the original image
        diff = torch.max(torch.abs((output_batch - img_batch) * (1.0 - mask_batch)))
        print(f"-> Background deviation outside mask: {diff.item():.6f}")
        assert diff.item() < 1e-5, f"Background was modified outside mask! Deviation: {diff.item()}"
        print("-> Counterfactual Inpainter Verification: PASS")
    except Exception as e:
        print(f"-> Counterfactual Inpainter Verification: FAIL ({str(e)})")
        sys.exit(1)
        
    # --------------------------------------------------
    # Step 4: Verification of Causal Contrastive Decoder
    # --------------------------------------------------
    print("\n[Step 4/5] Verifying Causal Contrastive Decoder (CCD)...")
    try:
        decoder = CausalContrastiveDecoder(gamma=1.5)
        # Test logits: confident original prediction vs. uncertain counterfactual prediction
        test_orig = torch.tensor([[2.0, -1.0]])       # confident "Yes" (index 0)
        test_cf = torch.tensor([[0.0, 0.0]])          # uncertain / neutral
        
        results = decoder(test_orig, test_cf)
        print("-> Test Input: Original Logits [2.0, -1.0], Counterfactual Logits [0.0, 0.0]")
        print(f"-> Calculated ICE: {results['ice'].numpy()}")
        print(f"-> Hallucination Risk Score: {results['hallucination_score'].numpy()}")
        print(f"-> Calibrated Probs: {results['calibrated_probs'].numpy()}")
        
        # Assertion check: ICE should be original - counterfactual
        assert torch.allclose(results["ice"], test_orig - test_cf), "ICE calculation error"
        print("-> Causal Contrastive Decoder Verification: PASS")
    except Exception as e:
        print(f"-> Causal Contrastive Decoder Verification: FAIL ({str(e)})")
        sys.exit(1)

    # --------------------------------------------------
    # Step 5: End-to-End Execution & Integration Test
    # --------------------------------------------------
    print("\n[Step 5/5] Running End-to-End Integration Check...")
    try:
        # Select VQA item 3 (Does the picture contain liver?)
        item = dataset[3]
        img = item["image"].unsqueeze(0)
        mask = item["mask"].unsqueeze(0)
        
        # Inpaint
        inpainted_img = inpainter(img, mask)
        
        # Assertions
        assert inpainted_img.shape == (1, 3, 224, 224)
        print("-> End-to-End Integration Verification: PASS")
    except Exception as e:
        print(f"-> End-to-End Integration Verification: FAIL ({str(e)})")
        sys.exit(1)
        
    print("\n==================================================")
    print("   ALL PIPELINE CHECKS PASSED SUCCESSFULLY!       ")
    print("==================================================")
    sys.exit(0)

if __name__ == "__main__":
    run_pipeline_check()
