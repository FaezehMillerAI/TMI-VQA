#!/usr/bin/env python3
"""
Publication-Quality Scientific Architectural Diagram for IEEE TMI:
Candidate 1: The Causal Contrastive Decoder (CCD) & Selective Triage Gate
Generates:
1. Manuscript TMI/fig_ccd_triage.png (300 DPI)
2. Manuscript TMI/fig_ccd_triage.pdf (Vector format)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
os.makedirs("Manuscript TMI", exist_ok=True)

# Set high-quality styling
plt.rcParams["font.sans-serif"] = "Helvetica, Arial, DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "dejavusans"

fig, ax = plt.subplots(figsize=(15.5, 8.2), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Background styling
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FFFFFF")

# -------------------------------------------------------------
# COLOR PALETTE (IEEE / Nature Biomedical Palette)
# -------------------------------------------------------------
C_BG_BLUE   = "#F0F7FF"
C_EDGE_BLUE = "#2563EB"
C_TXT_BLUE  = "#1E40AF"

C_BG_PURPLE = "#F5F3FF"
C_EDGE_PURP = "#7C3AED"
C_TXT_PURP  = "#5B21B6"

C_BG_AMBER  = "#FFFBEB"
C_EDGE_AMB  = "#D97706"
C_TXT_AMB   = "#92400E"

C_BG_TEAL   = "#ECFDF5"
C_EDGE_TEAL = "#059669"
C_TXT_TEAL  = "#065F46"

C_BG_GREEN  = "#F0FDF4"
C_EDGE_GRN  = "#16A34A"
C_TXT_GRN   = "#166534"

C_BG_RED    = "#FEF2F2"
C_EDGE_RED  = "#DC2626"
C_TXT_RED   = "#991B1B"

C_SLATE     = "#334155"
C_MUTED     = "#64748B"

# Helper for rounded boxes
def draw_box(x, y, w, h, bg_col, edge_col, lw=1.5, r=1.5):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={r},rounding_size=2.0",
        facecolor=bg_col, edgecolor=edge_col, linewidth=lw, zorder=2
    )
    ax.add_patch(box)
    return box

# Helper for arrows
def draw_arrow(x1, y1, x2, y2, color=C_MUTED, lw=2.0, style="->"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, zorder=3
    )
    ax.add_patch(arrow)
    return arrow

# -------------------------------------------------------------
# TITLE & HEADER
# -------------------------------------------------------------
ax.text(2.5, 96.5, "Causal Contrastive Decoding (CCD) & Selective Clinical Triage Architecture",
        fontsize=16, fontweight="bold", color="#0F172A", va="top")
ax.text(2.5, 93.0, "Dynamic Question Scaling $\\gamma(Q)$, Spurious Bias Cancellation, and Dual-Threshold Safety Referral",
        fontsize=11, color=C_MUTED, va="top")

# =============================================================
# COLUMN 1: DUAL INPUT PAIR & CLINICAL QUERY (x: 3 to 18)
# =============================================================
# Factual Scan Card
draw_box(3.5, 68, 14, 21, C_BG_BLUE, C_EDGE_BLUE, lw=1.8)
ax.text(10.5, 87.5, "Factual Radiograph $I$", fontsize=11, fontweight="bold", color=C_TXT_BLUE, ha="center")
# Simulated radiograph icon / lesion patch
rect_img = patches.Rectangle((5.5, 71.5), 10, 13, facecolor="#0F172A", edgecolor="#334155", lw=1.2, zorder=3)
ax.add_patch(rect_img)
# Left and right lung silhouettes
lung_l = patches.Ellipse((8.5, 78), 2.8, 7.5, facecolor="#1E293B", edgecolor="#475569", lw=0.8, zorder=4)
lung_r = patches.Ellipse((12.5, 78), 2.8, 7.5, facecolor="#1E293B", edgecolor="#475569", lw=0.8, zorder=4)
ax.add_patch(lung_l)
ax.add_patch(lung_r)
# Pathological lesion
lesion = patches.Circle((12.5, 76), 1.8, facecolor="#EF4444", alpha=0.9, edgecolor="#FCA5A5", lw=1.2, zorder=5)
ax.add_patch(lesion)
ax.text(12.5, 76, "Lesion", fontsize=6.5, color="white", fontweight="bold", ha="center", va="center", zorder=6)
ax.text(10.5, 69.5, "Original Patient Scan", fontsize=8.5, color=C_MUTED, ha="center")

# Question Card (Middle)
draw_box(3.5, 45, 14, 18, C_BG_AMBER, C_EDGE_AMB, lw=1.8)
ax.text(10.5, 61.2, "Clinical Query $Q$", fontsize=11, fontweight="bold", color=C_TXT_AMB, ha="center")
ax.text(10.5, 54.5, '"Is right lower lobe\nconsolidation present?"', fontsize=9, style="italic", color="#78350F", ha="center", va="center")
ax.text(10.5, 47.2, "PubMedBERT Tokens $\\mathbf{Q}$", fontsize=8.5, color=C_MUTED, ha="center")

# Counterfactual Scan Card
draw_box(3.5, 18, 14, 21, C_BG_PURPLE, C_EDGE_PURP, lw=1.8)
ax.text(10.5, 37.5, "Counterfactual $I_{\\mathrm{cf}}$", fontsize=11, fontweight="bold", color=C_TXT_PURP, ha="center")
rect_cf = patches.Rectangle((5.5, 21.5), 10, 13, facecolor="#0F172A", edgecolor="#334155", lw=1.2, zorder=3)
ax.add_patch(rect_cf)
lung_cf_l = patches.Ellipse((8.5, 28), 2.8, 7.5, facecolor="#1E293B", edgecolor="#475569", lw=0.8, zorder=4)
lung_cf_r = patches.Ellipse((12.5, 28), 2.8, 7.5, facecolor="#1E293B", edgecolor="#475569", lw=0.8, zorder=4)
ax.add_patch(lung_cf_l)
ax.add_patch(lung_cf_r)
# Inpainted healthy parenchyma patch
healthy_patch = patches.Circle((12.5, 26), 1.8, facecolor="#10B981", alpha=0.9, edgecolor="#6EE7B7", lw=1.2, zorder=5)
ax.add_patch(healthy_patch)
ax.text(12.5, 26, "Healthy", fontsize=6.5, color="white", fontweight="bold", ha="center", va="center", zorder=6)
ax.text(10.5, 19.5, "Physical $do(I = I_{\\mathrm{cf}})$", fontsize=8.5, color=C_MUTED, ha="center")

# =============================================================
# COLUMN 2: MULTIMODAL ENCODERS & FORWARD PASS (x: 23 to 37)
# =============================================================
# Factual ViT Encoder
draw_box(23, 70, 14, 17, "#FFFFFF", C_EDGE_BLUE, lw=1.6)
ax.text(30, 84.5, "ViT + Cross-Attn", fontsize=10.5, fontweight="bold", color=C_TXT_BLUE, ha="center")
ax.text(30, 78.5, "Visual Tokens $\\mathbf{V}_{\\mathrm{orig}}$\nFused with $\\mathbf{Q}$", fontsize=8.5, color=C_SLATE, ha="center")
ax.text(30, 72.5, "Forward Pass 1", fontsize=8, fontweight="bold", color=C_EDGE_BLUE, ha="center")

# Question Dynamic Scale Scorer (Middle)
draw_box(23, 45.5, 14, 17, "#FFFFFF", C_EDGE_AMB, lw=1.6)
ax.text(30, 60, "Dynamic Scaling $\\gamma(Q)$", fontsize=10.5, fontweight="bold", color=C_TXT_AMB, ha="center")
ax.text(30, 54.0, "$\\gamma(Q) = \\mathrm{Softplus}(\\mathbf{W}_\\gamma \\bar{\\mathbf{q}} + b_\\gamma)$", fontsize=8.5, color="#78350F", ha="center")
ax.text(30, 48.0, "Semantic Visual Reliance", fontsize=8, color=C_MUTED, ha="center")

# Counterfactual ViT Encoder
draw_box(23, 20, 14, 17, "#FFFFFF", C_EDGE_PURP, lw=1.6)
ax.text(30, 34.5, "ViT + Cross-Attn", fontsize=10.5, fontweight="bold", color=C_TXT_PURP, ha="center")
ax.text(30, 28.5, "Visual Tokens $\\mathbf{V}_{\\mathrm{cf}}$\nFused with $\\mathbf{Q}$", fontsize=8.5, color=C_SLATE, ha="center")
ax.text(30, 22.5, "Forward Pass 2", fontsize=8, fontweight="bold", color=C_EDGE_PURP, ha="center")

# Arrows from Col 1 to Col 2
draw_arrow(18.5, 78.5, 22.5, 78.5, color=C_EDGE_BLUE)
draw_arrow(18.5, 54.0, 22.5, 54.0, color=C_EDGE_AMB)
draw_arrow(18.5, 54.0, 22.5, 76.0, color=C_EDGE_BLUE)
draw_arrow(18.5, 54.0, 22.5, 30.0, color=C_EDGE_PURP)
draw_arrow(18.5, 28.5, 22.5, 28.5, color=C_EDGE_PURP)

# =============================================================
# COLUMN 3: LOGIT VECTORS & CONTRAST (x: 43 to 60)
# =============================================================
# Observational Logits Box
draw_box(43, 70, 16.5, 17, C_BG_BLUE, C_EDGE_BLUE, lw=1.6)
ax.text(51.25, 84.5, "Observational Logits $\\mathbf{L}_{\\mathrm{orig}}$", fontsize=10.5, fontweight="bold", color=C_TXT_BLUE, ha="center")
# Mini bar chart for L_orig
ax.barh([78.5, 74.5], [7, 4], left=46.5, height=2.2, color=[C_EDGE_BLUE, "#93C5FD"], zorder=4)
ax.text(45.5, 78.5, "Yes", fontsize=8, ha="right", va="center", color=C_SLATE)
ax.text(45.5, 74.5, "No", fontsize=8, ha="right", va="center", color=C_SLATE)
ax.text(51.25, 71.5, "Contains True Pathology + Bias", fontsize=7.5, color=C_TXT_BLUE, ha="center")

# Counterfactual Logits Box
draw_box(43, 20, 16.5, 17, C_BG_PURPLE, C_EDGE_PURP, lw=1.6)
ax.text(51.25, 34.5, "Counterfactual Logits $\\mathbf{L}_{\\mathrm{cf}}$", fontsize=10.5, fontweight="bold", color=C_TXT_PURP, ha="center")
# Mini bar chart for L_cf
ax.barh([28.5, 24.5], [4.5, 4.0], left=46.5, height=2.2, color=[C_EDGE_PURP, "#C4B5FD"], zorder=4)
ax.text(45.5, 28.5, "Yes", fontsize=8, ha="right", va="center", color=C_SLATE)
ax.text(45.5, 24.5, "No", fontsize=8, ha="right", va="center", color=C_SLATE)
ax.text(51.25, 21.5, "Contains Prior Language Bias", fontsize=7.5, color=C_TXT_PURP, ha="center")

# Arrows from Encoders to Logits
draw_arrow(37.5, 78.5, 42.5, 78.5, color=C_EDGE_BLUE)
draw_arrow(37.5, 28.5, 42.5, 28.5, color=C_EDGE_PURP)

# Subtraction / Contrast Box in Center
draw_box(43.5, 45.5, 15.5, 17, C_BG_TEAL, C_EDGE_TEAL, lw=1.8)
ax.text(51.25, 60.0, "Causal Contrast $\\Delta \\mathbf{L}$", fontsize=10.5, fontweight="bold", color=C_TXT_TEAL, ha="center")
ax.text(51.25, 54.5, "$\\Delta \\mathbf{L} = \\mathbf{L}_{\\mathrm{orig}} - \\mathbf{L}_{\\mathrm{cf}}$", fontsize=10, fontweight="bold", color=C_TXT_TEAL, ha="center")
ax.text(51.25, 48.0, "Language Bias Cancels Out!", fontsize=8, fontweight="bold", color="#047857", ha="center")

# Convergence arrows to Contrast Box
draw_arrow(51.25, 69.5, 51.25, 63.5, color=C_EDGE_BLUE)
draw_arrow(51.25, 37.5, 51.25, 44.5, color=C_EDGE_PURP)
draw_arrow(37.5, 54.0, 42.8, 54.0, color=C_EDGE_AMB)

# =============================================================
# COLUMN 4: INTERVENTIONAL PROBABILITY (x: 65 to 78)
# =============================================================
draw_box(65, 36, 14, 36, "#FFFFFF", C_EDGE_TEAL, lw=2.0)
ax.text(72, 69.0, "Interventional\nSoftmax", fontsize=11, fontweight="bold", color=C_TXT_TEAL, ha="center")

ax.text(72, 59.0, "$\\mathbf{L}_{\\mathrm{interv}} = \\mathbf{L}_{\\mathrm{orig}}$\n$+ \\gamma(Q) \\cdot \\Delta \\mathbf{L}$",
        fontsize=9.5, fontweight="bold", color=C_SLATE, ha="center")

ax.text(72, 47.0, "$P(A \\mid do(I), Q)$", fontsize=10.5, fontweight="bold", color=C_EDGE_TEAL, ha="center")
# Calibrated bar chart
ax.barh([43.5, 40.2], [8.5, 1.5], left=67.5, height=2.0, color=[C_EDGE_TEAL, "#A7F3D0"], zorder=4)
ax.text(66.5, 43.5, "Yes", fontsize=7.5, ha="right", va="center", color=C_SLATE)
ax.text(66.5, 40.2, "No", fontsize=7.5, ha="right", va="center", color=C_SLATE)
ax.text(72, 36.8, "Calibrated & Grounded", fontsize=7.5, color=C_TXT_TEAL, ha="center")

# Connect Contrast to Interventional Box
draw_arrow(59.5, 54.0, 64.5, 54.0, color=C_EDGE_TEAL, lw=2.2)

# =============================================================
# COLUMN 5: SELECTIVE CLINICAL TRIAGE GATE (x: 83 to 98)
# =============================================================
# Decision Diamond / Gate Box
draw_box(83, 46, 14, 16, "#F8FAFC", "#475569", lw=2.0)
ax.text(90, 58.0, "Selective Triage Gate", fontsize=10, fontweight="bold", color="#1E293B", ha="center")
ax.text(90, 52.5, "$\\max_a P(a) \\geq \\tau$ ?", fontsize=11, fontweight="bold", color="#0F172A", ha="center")
ax.text(90, 48.0, "Threshold $\\tau = 0.85$", fontsize=8, color=C_MUTED, ha="center")

# Connect Interventional Box to Gate
draw_arrow(79.5, 54.0, 82.5, 54.0, color="#475569", lw=2.2)

# PATHWAY A: AUTOMATED DIAGNOSIS (Top Right)
draw_box(83, 73, 14, 18, C_BG_GREEN, C_EDGE_GRN, lw=2.0)
ax.text(90, 87.5, "Automated Output", fontsize=10.5, fontweight="bold", color=C_TXT_GRN, ha="center")
ax.text(90, 81.5, "Prediction: $\\hat{A} = \\mathrm{Yes}$\nConfidence: 94.2%", fontsize=8.5, fontweight="bold", color="#14532D", ha="center")
ax.text(90, 75.0, "Coverage: 72.5%\nRisk: 2.4% (Ultra-Safe)", fontsize=7.5, color=C_TXT_GRN, ha="center")

# PATHWAY B: SPECIALIST REFERRAL (Bottom Right)
draw_box(83, 14, 14, 18, C_BG_RED, C_EDGE_RED, lw=2.0)
ax.text(90, 28.5, "Specialist Referral", fontsize=10.5, fontweight="bold", color=C_TXT_RED, ha="center")
ax.text(90, 22.5, "Ambiguous Contrast\n$\\Delta \\mathbf{L} \\approx 0$ or $\\mathrm{Conf} < \\tau$", fontsize=8.5, fontweight="bold", color="#7F1D1D", ha="center")
ax.text(90, 16.0, "Deferred to Radiologist\nZero False Certainty", fontsize=7.5, color=C_TXT_RED, ha="center")

# Decision Arrows
draw_arrow(90, 62.5, 90, 72.5, color=C_EDGE_GRN, lw=2.2)
ax.text(92.5, 67.5, "YES", fontsize=9, fontweight="bold", color=C_EDGE_GRN)

draw_arrow(90, 45.5, 90, 32.5, color=C_EDGE_RED, lw=2.2)
ax.text(92.5, 39.0, "NO", fontsize=9, fontweight="bold", color=C_EDGE_RED)

# -------------------------------------------------------------
# SAVE OUTPUTS
# -------------------------------------------------------------
out_png = "Manuscript TMI/fig_ccd_triage.png"
out_pdf = "Manuscript TMI/fig_ccd_triage.pdf"

plt.tight_layout()
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.close()

print(f"-> Successfully rendered Figure to:")
print(f"   1. {out_png}")
print(f"   2. {out_pdf}")
