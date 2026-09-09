#!/usr/bin/env python3
"""
Generate Publication-Grade Figure 1: CI-GCI Framework Architecture.
Four clear, non-duplicated stages:
  Stage 1: Question-Conditioned ROI Locator (QCRL)
  Stage 2: Generative Counterfactual Inpainter (CFI) [do(I = I_cf)]
  Stage 3: Causal Contrastive Decoder (CCD) [Causal Contrast Delta L & Dynamic gamma(Q)]
  Stage 4: Selective Abstention Triage Gate [Clinical Deferral / Specialist Referral]
Outputs:
  - Manuscript TMI/fig_framework.png & .pdf
  - Manuscript TMI/fig_framework.jpg (for backwards compatibility)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
os.makedirs("Manuscript TMI", exist_ok=True)

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

def create_framework_figure():
    fig, ax = plt.subplots(figsize=(19.0, 9.2), dpi=300)
    ax.set_xlim(0, 190)
    ax.set_ylim(0, 92)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    def draw_box(x, y, w, h, bg_col, edge_col, lw=1.5, r=1.8, zorder=2):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad={r},rounding_size=2.0",
            facecolor=bg_col, edgecolor=edge_col, linewidth=lw, zorder=zorder
        )
        ax.add_patch(box)
        return box

    def draw_arrow(x1, y1, x2, y2, color="#64748B", lw=2.0, style="-|>"):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=13,
            linewidth=lw, color=color, zorder=4
        )
        ax.add_patch(arrow)
        return arrow

    # =========================================================================
    # STAGE HEADER BANNERS (Across Top)
    # =========================================================================
    # Stage 1 Banner
    draw_box(4.0, 81.0, 42.0, 8.0, "#EFF6FF", "#2563EB", lw=1.8)
    ax.text(25.0, 86.2, "Stage 1: Question-Conditioned ROI Locator", fontsize=11.2, fontweight="bold", color="#1E40AF", ha="center", va="center")
    ax.text(25.0, 82.8, "Cross-Modal Patch-Word Attention $\\mathbf{S} \\to$ ROI Mask $\\mathbf{M}$", fontsize=8.8, color="#3B82F6", ha="center", va="center")

    # Stage 2 Banner
    draw_box(49.0, 81.0, 42.0, 8.0, "#F5F3FF", "#7C3AED", lw=1.8)
    ax.text(70.0, 86.2, "Stage 2: Generative Counterfactual Inpainter", fontsize=11.2, fontweight="bold", color="#5B21B6", ha="center", va="center")
    ax.text(70.0, 82.8, "Physical Intervention $do(I = I_{\\mathrm{cf}})$ with Anatomy Preservation", fontsize=8.8, color="#7C3AED", ha="center", va="center")

    # Stage 3 Banner
    draw_box(94.0, 81.0, 45.0, 8.0, "#ECFDF5", "#059669", lw=1.8)
    ax.text(116.5, 86.2, "Stage 3: Causal Contrastive Decoder", fontsize=11.2, fontweight="bold", color="#065F46", ha="center", va="center")
    ax.text(116.5, 82.8, "Contrast $\\Delta\\mathbf{L} = \\mathbf{L}_{\\mathrm{orig}} - \\mathbf{L}_{\\mathrm{cf}}$ with Question Scale $\\gamma(Q)$", fontsize=8.8, color="#059669", ha="center", va="center")

    # Stage 4 Banner
    draw_box(142.0, 81.0, 44.0, 8.0, "#FFFBEB", "#D97706", lw=1.8)
    ax.text(164.0, 86.2, "Stage 4: Selective Abstention Triage Gate", fontsize=11.2, fontweight="bold", color="#92400E", ha="center", va="center")
    ax.text(164.0, 82.8, "Clinical Decision Support: Automated vs Specialist Referral", fontsize=8.8, color="#B45309", ha="center", va="center")

    # =========================================================================
    # STAGE 1 CONTENT (x: 4 to 46)
    # =========================================================================
    # Dual Inputs Box
    draw_box(5.0, 45.0, 16.0, 32.0, "#F8FAFC", "#64748B", lw=1.4)
    ax.text(13.0, 74.0, "Patient Inputs", fontsize=10.0, fontweight="bold", color="#334155", ha="center")
    # Image icon
    rect_img = patches.Rectangle((7.0, 56.5), 12.0, 14.5, facecolor="#0F172A", edgecolor="#475569", lw=1.2, zorder=3)
    ax.add_patch(rect_img)
    lung_l = patches.Ellipse((10.5, 64.0), 3.0, 9.5, facecolor="#1E293B", edgecolor="#64748B", lw=0.8, zorder=4)
    lung_r = patches.Ellipse((15.5, 64.0), 3.0, 9.5, facecolor="#1E293B", edgecolor="#64748B", lw=0.8, zorder=4)
    ax.add_patch(lung_l)
    ax.add_patch(lung_r)
    lesion1 = patches.Circle((15.5, 62.0), 1.8, facecolor="#EF4444", alpha=0.9, zorder=5)
    ax.add_patch(lesion1)
    ax.text(13.0, 53.0, "Radiograph $I$", fontsize=8.5, fontweight="bold", color="#1E293B", ha="center")
    ax.text(13.0, 48.0, "Clinical Query $Q$:\n\"Consolidation?\"", fontsize=7.5, style="italic", color="#475569", ha="center", va="center")

    # Encoders Box
    draw_box(24.0, 45.0, 10.0, 32.0, "#FFFFFF", "#2563EB", lw=1.5)
    ax.text(29.0, 74.0, "Encoders", fontsize=10.0, fontweight="bold", color="#1E40AF", ha="center")
    draw_box(25.0, 60.5, 8.0, 10.0, "#DBEAFE", "#3B82F6", lw=1.0, r=0.5)
    ax.text(29.0, 66.5, "ViT-B/16", fontsize=8.0, fontweight="bold", color="#1E40AF", ha="center")
    ax.text(29.0, 62.5, "Tokens $\\mathbf{V}$", fontsize=7.2, color="#2563EB", ha="center")
    draw_box(25.0, 47.5, 8.0, 10.0, "#FEF3C7", "#F59E0B", lw=1.0, r=0.5)
    ax.text(29.0, 53.5, "PubMedBERT", fontsize=7.5, fontweight="bold", color="#92400E", ha="center")
    ax.text(29.0, 49.5, "Tokens $\\mathbf{Q}$", fontsize=7.2, color="#B45309", ha="center")

    draw_arrow(21.0, 65.0, 24.0, 65.0, color="#2563EB", lw=1.6)
    draw_arrow(21.0, 52.0, 24.0, 52.0, color="#D97706", lw=1.6)

    # QCRL Core Box
    draw_box(37.0, 45.0, 10.0, 32.0, "#EFF6FF", "#1D4ED8", lw=1.6)
    ax.text(42.0, 74.0, "QCRL Module", fontsize=9.5, fontweight="bold", color="#1E40AF", ha="center")
    ax.text(42.0, 66.0, "Cross-Attn\n$\\mathbf{S} = \\mathrm{Softmax}(\\frac{\\mathbf{Q}\\mathbf{V}^T}{\\sqrt{D}})$", fontsize=7.5, color="#1E3A8A", ha="center", va="center")
    # Mask thumbnail
    rect_m = patches.Rectangle((39.0, 50.0), 6.0, 8.0, facecolor="#000000", edgecolor="#10B981", lw=1.0, zorder=3)
    ax.add_patch(rect_m)
    spot_m = patches.Circle((43.0, 54.0), 1.5, facecolor="#FFFFFF", zorder=4)
    ax.add_patch(spot_m)
    ax.text(42.0, 47.0, "ROI Mask $\\mathbf{M}$", fontsize=8.0, fontweight="bold", color="#047857", ha="center")

    draw_arrow(34.0, 65.0, 37.0, 65.0, color="#2563EB", lw=1.6)
    draw_arrow(34.0, 52.0, 37.0, 52.0, color="#D97706", lw=1.6)

    # Mathematical note Stage 1
    draw_box(5.0, 8.0, 42.0, 32.0, "#F8FAFC", "#94A3B8", lw=1.2)
    ax.text(26.0, 36.5, "Stage 1: Formulation & Grounding", fontsize=9.5, fontweight="bold", color="#334155", ha="center")
    s1_eq = (
        r"$\mathbf{S}_{i,j} = \frac{\exp((\mathbf{Q}_i \mathbf{W}_q)(\mathbf{V}_j \mathbf{W}_v)^T / \sqrt{D})}{\sum_k \exp((\mathbf{Q}_i \mathbf{W}_q)(\mathbf{V}_k \mathbf{W}_v)^T / \sqrt{D})}$" + "\n\n"
        r"$\mathbf{M} = \mathrm{BilinearInterp}\left(\frac{1}{N}\sum_{i=1}^N \mathbf{S}_{i, :}\right) \in [0, 1]^{H \times W}$" + "\n\n"
        "• Pinpoints candidate lesion causally queried by question\n"
        "• Evaluated via Pointing Game, IoU, and Dice score\n"
        "• Differentiable spatial attention without manual annotations"
    )
    ax.text(6.5, 22.0, s1_eq, fontsize=7.6, color="#1E293B", va="center")

    # =========================================================================
    # STAGE 2 CONTENT (x: 50 to 90)
    # =========================================================================
    draw_box(50.0, 45.0, 39.0, 32.0, "#FAF5FF", "#7C3AED", lw=1.6)
    ax.text(69.5, 74.0, "Physical Counterfactual Synthesis $do(I = I_{\\mathrm{cf}})$", fontsize=10.0, fontweight="bold", color="#5B21B6", ha="center")

    # Inpainter Network
    draw_box(52.0, 52.0, 11.0, 18.0, "#FFFFFF", "#8B5CF6", lw=1.2, r=0.8)
    ax.text(57.5, 65.5, "Generative\nInpainter", fontsize=8.2, fontweight="bold", color="#6D28D9", ha="center", va="center")
    ax.text(57.5, 57.0, "$G_\\phi(I, \\mathbf{M})$", fontsize=8.5, color="#5B21B6", ha="center")

    # Blend block
    draw_box(66.5, 52.0, 9.5, 18.0, "#FFFFFF", "#059669", lw=1.2, r=0.8)
    ax.text(71.25, 66.0, "Strict Background\nPreservation", fontsize=7.5, fontweight="bold", color="#065F46", ha="center", va="center")
    ax.text(71.25, 56.5, "$(1 - \\mathbf{M}) \\odot I$\n$+\\mathbf{M} \\odot G_\\phi$", fontsize=7.8, color="#047857", ha="center", va="center")

    # Counterfactual Scan thumbnail
    rect_cf = patches.Rectangle((80.0, 54.0), 7.5, 14.0, facecolor="#0F172A", edgecolor="#059669", lw=1.2, zorder=3)
    ax.add_patch(rect_cf)
    lung_cf_l = patches.Ellipse((82.5, 61.0), 2.0, 8.5, facecolor="#1E293B", edgecolor="#475569", lw=0.6, zorder=4)
    lung_cf_r = patches.Ellipse((85.5, 61.0), 2.0, 8.5, facecolor="#1E293B", edgecolor="#475569", lw=0.6, zorder=4)
    ax.add_patch(lung_cf_l)
    ax.add_patch(lung_cf_r)
    healthy_patch = patches.Circle((85.5, 59.5), 1.3, facecolor="#10B981", alpha=0.9, zorder=5)
    ax.add_patch(healthy_patch)
    ax.text(83.75, 50.0, "Scan $I_{\\mathrm{cf}}$", fontsize=8.0, fontweight="bold", color="#065F46", ha="center")

    draw_arrow(47.0, 61.0, 52.0, 61.0, color="#7C3AED", lw=1.6)
    draw_arrow(63.0, 61.0, 66.5, 61.0, color="#7C3AED", lw=1.4)
    draw_arrow(76.0, 61.0, 80.0, 61.0, color="#059669", lw=1.6)

    # Mathematical note Stage 2
    draw_box(50.0, 8.0, 39.0, 32.0, "#F8FAFC", "#94A3B8", lw=1.2)
    ax.text(69.5, 36.5, "Stage 2: Intervention & Verification", fontsize=9.5, fontweight="bold", color="#334155", ha="center")
    s2_eq = (
        r"$I_{\mathrm{cf}} = (1 - \mathbf{M}) \odot I + \mathbf{M} \odot G_\phi(I, \mathbf{M})$" + "\n\n"
        "• Replaces pathological ROI with photorealistic healthy texture\n"
        "• Guarantees identical background preservation outside mask\n"
        "• Verified by Two One-Sided Tests (TOST, non-target findings)\n"
        "• Quantitative Fidelity: PSNR = 38.64 dB, SSIM = 0.982\n"
        "• Realism: Discriminator ROC-AUC = 0.542 (indistinguishable)"
    )
    ax.text(51.5, 22.0, s2_eq, fontsize=7.6, color="#1E293B", va="center")

    # =========================================================================
    # STAGE 3 CONTENT (x: 94 to 138)
    # =========================================================================
    draw_box(94.0, 45.0, 43.0, 32.0, "#ECFDF5", "#059669", lw=1.6)
    ax.text(115.5, 74.0, "Causal Contrast & Calibrated Decoding", fontsize=10.0, fontweight="bold", color="#065F46", ha="center")

    # Dual forward passes
    draw_box(96.0, 60.0, 11.0, 10.5, "#EFF6FF", "#2563EB", lw=1.0, r=0.5)
    ax.text(101.5, 66.5, "Factual Pass", fontsize=7.8, fontweight="bold", color="#1E40AF", ha="center")
    ax.text(101.5, 62.5, "Logits $\\mathbf{L}_{\\mathrm{orig}}$", fontsize=7.4, color="#2563EB", ha="center")

    draw_box(96.0, 47.5, 11.0, 10.5, "#F5F3FF", "#7C3AED", lw=1.0, r=0.5)
    ax.text(101.5, 54.0, "Counterfactual", fontsize=7.8, fontweight="bold", color="#5B21B6", ha="center")
    ax.text(101.5, 50.0, "Logits $\\mathbf{L}_{\\mathrm{cf}}$", fontsize=7.4, color="#7C3AED", ha="center")

    # Causal Contrast Block
    draw_box(111.0, 53.0, 11.5, 15.0, "#FFFFFF", "#059669", lw=1.5, r=0.8)
    ax.text(116.75, 64.0, "Causal Contrast", fontsize=8.2, fontweight="bold", color="#065F46", ha="center")
    ax.text(116.75, 58.5, "$\\Delta\\mathbf{L} = \\mathbf{L}_{\\mathrm{orig}} - \\mathbf{L}_{\\mathrm{cf}}$", fontsize=8.0, fontweight="bold", color="#047857", ha="center")
    ax.text(116.75, 54.5, "(Causal Effect)", fontsize=7.0, color="#64748B", ha="center")

    # Dynamic Gamma Block
    draw_box(126.0, 53.0, 9.5, 15.0, "#FEF3C7", "#D97706", lw=1.2, r=0.8)
    ax.text(130.75, 64.0, "Dynamic", fontsize=8.0, fontweight="bold", color="#92400E", ha="center")
    ax.text(130.75, 59.5, "Scale $\\gamma(Q)$", fontsize=8.0, fontweight="bold", color="#B45309", ha="center")
    ax.text(130.75, 55.0, "Softplus Head", fontsize=6.8, color="#78350F", ha="center")

    draw_arrow(87.5, 61.0, 96.0, 64.0, color="#2563EB", lw=1.5)
    draw_arrow(87.5, 57.0, 96.0, 53.0, color="#7C3AED", lw=1.5)
    draw_arrow(107.0, 62.0, 111.0, 62.0, color="#059669", lw=1.4)
    draw_arrow(107.0, 53.0, 111.0, 58.0, color="#059669", lw=1.4)
    draw_arrow(122.5, 60.5, 126.0, 60.5, color="#D97706", lw=1.4)

    # Mathematical note Stage 3
    draw_box(94.0, 8.0, 43.0, 32.0, "#F8FAFC", "#94A3B8", lw=1.2)
    ax.text(115.5, 36.5, "Stage 3: Calibration Mechanism", fontsize=9.5, fontweight="bold", color="#334155", ha="center")
    s3_eq = (
        r"$\gamma(Q) = \mathrm{Softplus}\left(\mathbf{W}_\gamma \cdot \mathrm{MeanPool}(\mathbf{Q}) + b_\gamma\right) \in \mathbb{R}^{+}$" + "\n\n"
        r"$P(A \mid do(I), Q) = \mathrm{Softmax}\left(\mathbf{L}_{\mathrm{orig}} + \gamma(Q) \cdot \Delta\mathbf{L}\right)$" + "\n\n"
        "• Cancels non-causal language prior shortcuts ($Q \\to A$)\n"
        "• Heteroscedastic scaling prevents overconfident miscalibration\n"
        "• Cuts Macro Expected Calibration Error (ECE) by 64.8%\n"
        "• Reduces Brier score from 0.2394 to 0.1415 (-40.9% error)"
    )
    ax.text(95.5, 22.0, s3_eq, fontsize=7.6, color="#1E293B", va="center")

    # =========================================================================
    # STAGE 4 CONTENT (x: 142 to 186)
    # =========================================================================
    draw_box(142.0, 45.0, 43.0, 32.0, "#FFFBEB", "#D97706", lw=1.6)
    ax.text(163.5, 74.0, "Selective Triage Gate $\\mathcal{R} \\in \\{0, 1\\}$", fontsize=10.0, fontweight="bold", color="#92400E", ha="center")

    # Threshold test box
    draw_box(144.0, 51.5, 12.0, 18.0, "#FFFFFF", "#D97706", lw=1.2, r=0.8)
    ax.text(150.0, 66.0, "Confidence\nAudit", fontsize=8.0, fontweight="bold", color="#92400E", ha="center", va="center")
    ax.text(150.0, 56.5, r"$\max_a P(a) \geq \tau$" + "\n" + r"& $|\Delta\mathbf{L}| > \epsilon$", fontsize=7.5, color="#B45309", ha="center", va="center")

    # Triage Branches
    draw_box(160.0, 62.0, 23.0, 9.5, "#ECFDF5", "#059669", lw=1.4, r=0.5)
    ax.text(171.5, 68.0, "Automated Diagnostic Output $\\hat{A}$ ($\\mathcal{R} = 0$)", fontsize=7.5, fontweight="bold", color="#065F46", ha="center")
    ax.text(171.5, 64.0, "High confidence, grounded: 72.5% coverage @ 2.4% risk", fontsize=6.8, color="#047857", ha="center")

    draw_box(160.0, 48.0, 23.0, 9.5, "#FEF2F2", "#DC2626", lw=1.4, r=0.5)
    ax.text(171.5, 54.0, "Defer to Specialist Radiologist ($\\mathcal{R} = 1$)", fontsize=7.5, fontweight="bold", color="#991B1B", ha="center")
    ax.text(171.5, 50.0, "Ambiguous contrast ($\\Delta\\mathbf{L} \\approx 0$) / uncertain case", fontsize=6.8, color="#B91C1C", ha="center")

    draw_arrow(137.0, 60.5, 144.0, 60.5, color="#059669", lw=1.6)
    draw_arrow(156.0, 63.0, 160.0, 66.5, color="#059669", lw=1.4)
    draw_arrow(156.0, 57.0, 160.0, 53.0, color="#DC2626", lw=1.4)

    # Mathematical note Stage 4
    draw_box(142.0, 8.0, 43.0, 32.0, "#F8FAFC", "#94A3B8", lw=1.2)
    ax.text(163.5, 36.5, "Stage 4: Decision-Theoretic Policy", fontsize=9.5, fontweight="bold", color="#334155", ha="center")
    s4_eq = (
        r"$\mathrm{Decision} = \hat{A}\; (\mathcal{R}=0) \quad \mathrm{if}\; \max_a P(a \mid do(I), Q) \geq \tau$" + "\n"
        r"$\mathrm{Decision} = \mathrm{Referral}\; (\mathcal{R}=1) \quad \mathrm{otherwise}$" + "\n\n"
        "• Empirically selects threshold $\\tau_2 = 0.85$ on validation\n"
        "• Achieves 2.4% clinical risk at 72.5% patient coverage\n"
        "• Seamlessly defers ambiguous edge cases to attending clinicians\n"
        "• Transforms Med-VQA from black-box into dependable triage partner"
    )
    ax.text(143.5, 22.0, s4_eq, fontsize=7.6, color="#1E293B", va="center")

    plt.tight_layout()
    out_png = "Manuscript TMI/fig_framework.png"
    out_jpg = "Manuscript TMI/fig_framework.jpg"
    out_pdf = "Manuscript TMI/fig_framework.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_jpg, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated clean, publication-grade Figure 1 at {out_png}, {out_jpg}, {out_pdf}!")

if __name__ == "__main__":
    create_framework_figure()
