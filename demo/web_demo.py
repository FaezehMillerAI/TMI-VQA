"""
Interactive Clinical Medical VQA Demonstration (CI-GCI / CQC-Net)
Provides an interactive graphical interface for clinical question answering,
gaze ROI attention heatmap visualizer, counterfactual inpainting, and calibrated confidence gauge.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Workspace setup
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_dir)

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

from utils.config import load_config
from utils.vocab import load_vocab
from models.cqc_net import CQCNet
from models.inpainter import CounterfactualInpainter
from models.causal_decoder import CausalContrastiveDecoder

# Device
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
config = load_config("configs/baseline_vqa.yaml")
config["model"]["num_classes"] = 200

vqa_model = CQCNet(config).to(device)
if os.path.exists("models/slake_vqa_model.pth"):
    vqa_model.load_state_dict(torch.load("models/slake_vqa_model.pth", map_location=device), strict=False)
vqa_model.eval()

inpainter = CounterfactualInpainter(bilinear=True).to(device)
if os.path.exists("models/inpainter.pth"):
    inpainter.load_state_dict(torch.load("models/inpainter.pth", map_location=device), strict=False)
inpainter.eval()

causal_decoder = CausalContrastiveDecoder(gamma=1.5).to(device)

# Load Vocab
if os.path.exists("models/slake_vocab.json"):
    ans2idx, idx2ans = load_vocab("models/slake_vocab.json")
else:
    ans2idx, idx2ans = {"yes": 0, "no": 1}, {0: "yes", 1: "no"}

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_vqa(image, question, tau_threshold=0.80):
    if image is None or not question.strip():
        return None, None, "Please provide both an image and a clinical question.", "Awaiting input"

    raw_pil = Image.fromarray(image).convert('RGB')
    img_t = transform(raw_pil).unsqueeze(0).to(device)

    # Simulated Gaze Attention Mask for Demo
    mask_np = np.zeros((224, 224), dtype=np.float32)
    h, w = 224, 224
    mask_np[h//4:3*h//4, w//4:3*w//4] = 1.0
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        # 1. Original Pass
        orig_out = vqa_model(img_t, [question], device)
        orig_logits = orig_out["main_class_logits"]
        orig_probs = torch.softmax(orig_logits, dim=-1)
        gamma = orig_out.get("gamma", torch.tensor([[1.5]], device=device))

        # 2. Inpainted Pass
        cf_img = inpainter(img_t, mask_t)
        cf_out = vqa_model(cf_img, [question], device)
        cf_logits = cf_out["main_class_logits"]
        cf_probs = torch.softmax(cf_logits, dim=-1)

        # 3. Calibrated Pass
        calib_out = causal_decoder(orig_logits, cf_logits, gamma=gamma)
        calib_probs = calib_out["calibrated_probs"]

        # Formulate answer
        pred_idx = torch.argmax(calib_probs[0]).item()
        pred_ans = idx2ans.get(pred_idx, "abnormal / finding present")
        conf_val = calib_probs[0, pred_idx].item()
        base_conf = orig_probs[0, pred_idx].item()
        cf_conf = cf_probs[0, pred_idx].item()
        ite_val = base_conf - cf_conf

        # Convert images to PIL for display
        img_vis = (img_t[0].cpu().numpy().transpose(1, 2, 0) * 0.229 + 0.485).clip(0, 1)
        cf_vis = (cf_img[0].cpu().numpy().transpose(1, 2, 0) * 0.229 + 0.485).clip(0, 1)

        img_vis_pil = Image.fromarray((img_vis * 255).astype(np.uint8))
        cf_vis_pil = Image.fromarray((cf_vis * 255).astype(np.uint8))

        # Triage status
        if conf_val >= tau_threshold:
            triage_msg = f"✅ ACCEPTED (Confidence: {conf_val*100:.1f}% >= Threshold {tau_threshold*100:.0f}%)"
            rec = f"Automated Diagnosis: '{pred_ans}'"
        else:
            triage_msg = f"⚠️ ABSTAINED / REFERRED (Confidence: {conf_val*100:.1f}% < Threshold {tau_threshold*100:.0f}%)"
            rec = f"High Diagnostic Uncertainty. Recommended: Refer to attending radiologist."

        summary_text = (
            f"### 🩺 Diagnostic Prediction: **{pred_ans.upper()}**\n\n"
            f"- **Calibrated Interventional Confidence**: `{conf_val*100:.2f}%`\n"
            f"- **Baseline Observational Confidence**: `{base_conf*100:.2f}%`\n"
            f"- **Counterfactual Inpainted Confidence**: `{cf_conf*100:.2f}%`\n"
            f"- **Individual Treatment Effect (ITE)**: `+{ite_val:.3f}`\n"
            f"- **Question Causal Scale $\\gamma(Q)$**: `{gamma[0].mean().item():.3f}`\n\n"
            f"---\n"
            f"### 🛡️ Clinical Decision Triage Gate\n"
            f"**{triage_msg}**\n\n"
            f"*{rec}*"
        )

        return img_vis_pil, cf_vis_pil, summary_text

def launch_demo():
    if not HAS_GRADIO:
        print("Gradio is not installed. To run interactive web demo: pip install gradio")
        print("Demo code is available at demo/web_demo.py")
        return

    with gr.Blocks(title="CI-GCI / CQC-Net Med-VQA Clinical Demo") as demo:
        gr.Markdown("# 🩺 CI-GCI: Causal Interventional Grounding & Inpainting for Med-VQA")
        gr.Markdown("Interactive Trustworthy Diagnostic VQA with Physical Counterfactual Inpainting and Selective Clinical Abstention.")

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="numpy", label="Upload Patient Medical Scan (CT, MRI, X-ray)")
                input_q = gr.Textbox(label="Clinical Question", placeholder="e.g., Is there a pulmonary lesion present in the lung field?", lines=2)
                tau_slider = gr.Slider(minimum=0.50, maximum=0.99, value=0.80, step=0.01, label="Selective Abstention Threshold (τ)")
                submit_btn = gr.Button("Run Causal Diagnostic Inference", variant="primary")

            with gr.Column():
                out_img_orig = gr.Image(type="pil", label="Original Scan with Gaze Attention")
                out_img_cf = gr.Image(type="pil", label="Counterfactual Inpainted Scan (do(I \ ROI))")
                out_text = gr.Markdown(label="Diagnostic & Calibration Output")

        submit_btn.click(
            fn=predict_vqa,
            inputs=[input_img, input_q, tau_slider],
            outputs=[out_img_orig, out_img_cf, out_text]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    launch_demo()
