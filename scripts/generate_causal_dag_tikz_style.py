#!/usr/bin/env python3
"""
Generate Causal DAGs in the exact TikZ style of the user's reference image:
- (a) Observational Med-VQA (Confounded SCM with Backdoor path)
- (b) Proposed CI-GCI: Interventional SCM with Causal Contrastive Decoding (CCD) & Selective Triage Gate
Outputs:
- Manuscript TMI/fig_dag_comparative.png (300 DPI)
- Manuscript TMI/fig_dag_comparative.pdf (Vector format)
- Manuscript TMI/fig_dag_proposal_only.png (300 DPI)
- Manuscript TMI/fig_dag_proposal_only.pdf (Vector format)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
os.makedirs("Manuscript TMI", exist_ok=True)

# Use LaTeX style font rendering if possible, fallback to clean serif/DejaVu Serif
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern math font like LaTeX

# Colors matching TikZ colors:
# latent: dashed, fill=orange!10 (#FEF5ED)
# observed: fill=gray!15 (#EBECEE)
# outcome: fill=blue!15 (#DDE3FD)
# counterfactual: fill=teal!15 (#E0F2F1)
# dynamic_scale: fill=yellow!15 (#FFF9C4)
# gate: fill=purple!15 (#F3E8FF)
# referral: fill=red!15 (#FFEBEE)

COL_LATENT = "#FDF3E7"
COL_OBSERVED = "#EAECEE"
COL_OUTCOME = "#DDE3FD"
COL_CF = "#E0F4EE"
COL_SCALE = "#FEF9E7"
COL_CONTRAST = "#D1EDE8"
COL_GATE = "#F5EEF8"
COL_REFERRAL = "#FCE4E4"

def draw_node(ax, xy, radius, label, fill_color, dashed=False, lw=1.2, font_size=15, font_weight="normal"):
    x, y = xy
    linestyle = "--" if dashed else "-"
    circle = patches.Circle(
        (x, y), radius,
        facecolor=fill_color, edgecolor="black",
        linewidth=lw, linestyle=linestyle, zorder=4
    )
    ax.add_patch(circle)
    ax.text(x, y, label, fontsize=font_size, ha="center", va="center", zorder=5, fontweight=font_weight)
    return circle

def draw_dag_arrow(ax, p1, p2, r1, r2, color="black", lw=1.4, linestyle="-", shrink=0.0):
    """Draw straight arrow between node boundaries"""
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    v = p2 - p1
    dist = np.linalg.norm(v)
    if dist == 0:
        return
    u = v / dist
    start = p1 + u * (r1 + shrink)
    end = p2 - u * (r2 + shrink)
    
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=14,
        color=color, linewidth=lw, linestyle=linestyle, zorder=3
    )
    ax.add_patch(arrow)
    return arrow

def draw_curved_dag_arrow(ax, p1, p2, r1, r2, rad=0.3, color="red", lw=1.4, linestyle="--", label=None, label_offset=(0,0)):
    """Draw curved arrow between node boundaries with optional label"""
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    v = p2 - p1
    dist = np.linalg.norm(v)
    u = v / dist
    start = p1 + u * r1
    end = p2 - u * r2
    
    arrow = FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=14,
        color=color, linewidth=lw, linestyle=linestyle, zorder=3
    )
    ax.add_patch(arrow)
    
    if label:
        # Approximate midpoint of arc
        mid = (start + end) / 2.0
        norm = np.array([-u[1], u[0]])
        label_pos = mid + norm * (dist * rad * 0.5) + np.array(label_offset)
        ax.text(label_pos[0], label_pos[1], label, color=color, fontsize=10, ha="center", va="center", zorder=6)
    return arrow

# ====================================================================
# FIGURE 1: SIDE-BY-SIDE COMPARATIVE CAUSAL DAG (IEEE TMI Format)
# ====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 7.8), dpi=300)
for ax in (ax1, ax2):
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-4.2, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")

R = 0.52  # Standard node radius

# -------------------------------------------------------------
# PANEL (A): OBSERVATIONAL MED-VQA (EXACT REPLICA OF USER ATTACHMENT)
# -------------------------------------------------------------
ax1.text(0, 3.8, "(a) Observational Med-VQA (Confounded)", fontsize=13, fontweight="bold", ha="center")

# Node positions
C_pos = (0.0, 2.7)
I_pos = (-1.8, 1.1)
Q_pos = (1.8, 1.1)
V_pos = (-1.8, -1.0)
A_pos = (0.0, -2.8)

# Draw nodes
draw_node(ax1, C_pos, R, "$C$", COL_LATENT, dashed=True)
draw_node(ax1, I_pos, R, "$I$", COL_OBSERVED)
draw_node(ax1, Q_pos, R, "$Q$", COL_OBSERVED)
draw_node(ax1, V_pos, R, "$V$", COL_OBSERVED)
draw_node(ax1, A_pos, R, "$A$", COL_OUTCOME)

# Draw edges
draw_dag_arrow(ax1, C_pos, I_pos, R, R)
draw_dag_arrow(ax1, C_pos, Q_pos, R, R)
draw_dag_arrow(ax1, I_pos, V_pos, R, R)
draw_dag_arrow(ax1, Q_pos, V_pos, R, R)
draw_dag_arrow(ax1, V_pos, A_pos, R, R)
draw_dag_arrow(ax1, Q_pos, A_pos, R, R)

# Draw red dashed Backdoor edge
draw_curved_dag_arrow(ax1, C_pos, A_pos, R, R, rad=0.28, color="red", label="Backdoor", label_offset=(-0.35, 0.0))

# Sub-annotation
ax1.text(0, -3.7, "Spurious Correlation via $I \\leftarrow C \\rightarrow Q \\rightarrow A$\nUnchecked Reporting Bias",
         fontsize=9.5, color="#475569", ha="center", va="top")

# -------------------------------------------------------------
# PANEL (B): PROPOSED CI-GCI (INTERVENTIONAL SCM, CCD & TRIAGE)
# -------------------------------------------------------------
ax2.text(0, 3.8, "(b) Proposed CI-GCI (Intervention, CCD & Selective Triage)", fontsize=13, fontweight="bold", ha="center")

# Node positions in Panel B
C_b    = (0.0, 2.7)
I_b    = (-2.2, 1.3)
Icf_b  = (-0.8, 1.3)
Q_b    = (2.0, 1.3)

V_b    = (-2.2, -0.4)
Vcf_b  = (-0.8, -0.4)
Gam_b  = (1.8, -0.4)

DL_b   = (-1.0, -1.8)
Gate_b = (0.5, -1.8)

A_b    = (-0.6, -3.2)
R_b    = (1.6, -3.2)

# Draw Nodes
draw_node(ax2, C_b, R, "$C$", COL_LATENT, dashed=True)

# Factual and Counterfactual Image Nodes
draw_node(ax2, I_b, R, "$I$", COL_OBSERVED)
draw_node(ax2, Icf_b, R, "$I_{\\mathrm{cf}}$", COL_CF)

# Clinical Question Node
draw_node(ax2, Q_b, R, "$Q$", COL_OBSERVED)

# Multimodal Visual Representations
draw_node(ax2, V_b, R, "$\\mathbf{V}$", COL_OBSERVED)
draw_node(ax2, Vcf_b, R, "$\\mathbf{V}_{\\mathrm{cf}}$", COL_CF)

# Dynamic Scale Factor
draw_node(ax2, Gam_b, R, "$\\gamma(Q)$", COL_SCALE, font_size=11)

# Causal Contrast / CCD Node
draw_node(ax2, DL_b, R, "$\\Delta \\mathbf{L}$", COL_CONTRAST, font_size=12)

# Selective Triage Gate Node
draw_node(ax2, Gate_b, R, "$\\tau$", COL_GATE, font_size=13, font_weight="bold")

# Clinical Outcomes
draw_node(ax2, A_b, R, "$\\hat{A}$", COL_OUTCOME, font_size=13)
draw_node(ax2, R_b, R, "$\\mathcal{R}$", COL_REFERRAL, font_size=13)

# Draw Edges for Panel B:
# 1. Severed edge C -> I (Pearl's graph surgery via physical inpainting)
draw_dag_arrow(ax2, C_b, I_b, R, R, color="#94A3B8", linestyle=":")
# Cross mark on C -> I
mid_ci = (np.array(C_b) + np.array(I_b)) / 2.0
ax2.text(mid_ci[0], mid_ci[1], "$\\times$", fontsize=18, color="red", fontweight="bold", ha="center", va="center")
ax2.text(mid_ci[0] - 0.45, mid_ci[1] + 0.25, "$do(I)$", fontsize=9.5, color="red", fontweight="bold", ha="center")

# 2. C -> Q remains
draw_dag_arrow(ax2, C_b, Q_b, R, R)

# 3. I -> V and Icf -> Vcf
draw_dag_arrow(ax2, I_b, V_b, R, R)
draw_dag_arrow(ax2, Icf_b, Vcf_b, R, R, color="#059669")

# 4. Q -> V, Q -> Vcf, Q -> gamma
draw_dag_arrow(ax2, Q_b, V_b, R, R)
draw_dag_arrow(ax2, Q_b, Vcf_b, R, R)
draw_dag_arrow(ax2, Q_b, Gam_b, R, R, color="#D97706")

# 5. Contrast computation: V -> Delta L, Vcf -> Delta L, gamma -> Delta L
draw_dag_arrow(ax2, V_b, DL_b, R, R)
draw_dag_arrow(ax2, Vcf_b, DL_b, R, R, color="#059669")
draw_dag_arrow(ax2, Gam_b, DL_b, R, R, color="#D97706")

# 6. Delta L and V feed into Selective Triage Gate tau
draw_dag_arrow(ax2, DL_b, Gate_b, R, R, color="#0F766E", lw=1.6)

# 7. Selective Triage Outcomes: tau -> A (Automated) vs tau -> R (Referral)
draw_dag_arrow(ax2, Gate_b, A_b, R, R, color="#2563EB", lw=1.6)
ax2.text((Gate_b[0] + A_b[0])/2.0 - 0.28, (Gate_b[1] + A_b[1])/2.0, "$\\geq \\tau$", fontsize=9, color="#2563EB", fontweight="bold")

draw_dag_arrow(ax2, Gate_b, R_b, R, R, color="#DC2626", lw=1.6)
ax2.text((Gate_b[0] + R_b[0])/2.0 + 0.28, (Gate_b[1] + R_b[1])/2.0, "$< \\tau$", fontsize=9, color="#DC2626", fontweight="bold")

# Sub-annotation
ax2.text(0, -3.7, "Spurious Shortcut Canceled via $\\Delta \\mathbf{L} = \\mathbf{L}_{\\mathrm{orig}} - \\mathbf{L}_{\\mathrm{cf}}$\nSafe Abstention: Automated ($\\hat{A}$) vs Referral ($\\mathcal{R}$)",
         fontsize=9.5, color="#475569", ha="center", va="top")

# Save outputs
plt.tight_layout()
out_comp_png = "Manuscript TMI/fig_dag_comparative.png"
out_comp_pdf = "Manuscript TMI/fig_dag_comparative.pdf"
plt.savefig(out_comp_png, dpi=300, bbox_inches="tight")
plt.savefig(out_comp_pdf, dpi=300, bbox_inches="tight")
plt.close()

print(f"Generated comparative DAG:")
print(f"  1. {out_comp_png}")
print(f"  2. {out_comp_pdf}")

# ====================================================================
# FIGURE 2: STANDALONE PROPOSED CANDIDATE 1 DAG (Single Panel)
# ====================================================================
fig2, ax = plt.subplots(figsize=(7.5, 7.8), dpi=300)
ax.set_xlim(-3.6, 3.6)
ax.set_ylim(-4.2, 4.2)
ax.set_aspect("equal")
ax.axis("off")

# Draw Standalone Candidate 1
ax.text(0, 3.8, "Causal Graph of CCD & Selective Triage Gate", fontsize=13, fontweight="bold", ha="center")

# Nodes
draw_node(ax, C_b, R, "$C$", COL_LATENT, dashed=True)
draw_node(ax, I_b, R, "$I$", COL_OBSERVED)
draw_node(ax, Icf_b, R, "$I_{\\mathrm{cf}}$", COL_CF)
draw_node(ax, Q_b, R, "$Q$", COL_OBSERVED)
draw_node(ax, V_b, R, "$\\mathbf{V}$", COL_OBSERVED)
draw_node(ax, Vcf_b, R, "$\\mathbf{V}_{\\mathrm{cf}}$", COL_CF)
draw_node(ax, Gam_b, R, "$\\gamma(Q)$", COL_SCALE, font_size=11)
draw_node(ax, DL_b, R, "$\\Delta \\mathbf{L}$", COL_CONTRAST, font_size=12)
draw_node(ax, Gate_b, R, "$\\tau$", COL_GATE, font_size=13, font_weight="bold")
draw_node(ax, A_b, R, "$\\hat{A}$", COL_OUTCOME, font_size=13)
draw_node(ax, R_b, R, "$\\mathcal{R}$", COL_REFERRAL, font_size=13)

# Edges
draw_dag_arrow(ax, C_b, I_b, R, R, color="#94A3B8", linestyle=":")
ax.text(mid_ci[0], mid_ci[1], "$\\times$", fontsize=18, color="red", fontweight="bold", ha="center", va="center")
ax.text(mid_ci[0] - 0.45, mid_ci[1] + 0.25, "$do(I)$", fontsize=9.5, color="red", fontweight="bold", ha="center")

draw_dag_arrow(ax, C_b, Q_b, R, R)
draw_dag_arrow(ax, I_b, V_b, R, R)
draw_dag_arrow(ax, Icf_b, Vcf_b, R, R, color="#059669")
draw_dag_arrow(ax, Q_b, V_b, R, R)
draw_dag_arrow(ax, Q_b, Vcf_b, R, R)
draw_dag_arrow(ax, Q_b, Gam_b, R, R, color="#D97706")
draw_dag_arrow(ax, V_b, DL_b, R, R)
draw_dag_arrow(ax, Vcf_b, DL_b, R, R, color="#059669")
draw_dag_arrow(ax, Gam_b, DL_b, R, R, color="#D97706")
draw_dag_arrow(ax, DL_b, Gate_b, R, R, color="#0F766E", lw=1.6)

draw_dag_arrow(ax, Gate_b, A_b, R, R, color="#2563EB", lw=1.6)
ax.text((Gate_b[0] + A_b[0])/2.0 - 0.28, (Gate_b[1] + A_b[1])/2.0, "$\\geq \\tau$", fontsize=9, color="#2563EB", fontweight="bold")

draw_dag_arrow(ax, Gate_b, R_b, R, R, color="#DC2626", lw=1.6)
ax.text((Gate_b[0] + R_b[0])/2.0 + 0.28, (Gate_b[1] + R_b[1])/2.0, "$< \\tau$", fontsize=9, color="#DC2626", fontweight="bold")

ax.text(0, -3.7, "Structural Causal Model of CI-GCI\nPhysical $do(I)$ Severance $\\cdot$ Dynamic Scaling $\\gamma(Q) \\cdot$ Decision Gate $\\tau$",
        fontsize=9.5, color="#475569", ha="center", va="top")

out_prop_png = "Manuscript TMI/fig_dag_proposal_only.png"
out_prop_pdf = "Manuscript TMI/fig_dag_proposal_only.pdf"
plt.tight_layout()
plt.savefig(out_prop_png, dpi=300, bbox_inches="tight")
plt.savefig(out_prop_pdf, dpi=300, bbox_inches="tight")
plt.close()

print(f"Generated proposal-only DAG:")
print(f"  1. {out_prop_png}")
print(f"  2. {out_prop_pdf}")
