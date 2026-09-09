#!/usr/bin/env python3
"""
Generate Causal DAG for Candidate 2:
Question-Conditioned ROI Locator (QCRL) & Generative Inpainting Pipeline
Outputs:
- Manuscript TMI/fig_qcrl_dag.png (300 DPI) & .pdf
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

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

COL_LATENT   = "#FDF3E7"  # orange!10
COL_OBSERVED = "#EAECEE"  # gray!15
COL_OUTCOME  = "#DDE3FD"  # blue!15
COL_CF       = "#E0F4EE"  # teal!15
COL_SCALE    = "#FEF9E7"  # amber!15
COL_CONTRAST = "#D1EDE8"  # teal!20
COL_GATE     = "#F5EEF8"  # purple!15
COL_MASK     = "#FEF5D1"  # yellow!20
COL_INPAINT  = "#EDE7F6"  # purple!10

def draw_node(ax, xy, radius, label, fill_color, dashed=False, lw=1.3, font_size=13, font_weight="normal"):
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
        mid = (start + end) / 2.0
        norm = np.array([-u[1], u[0]])
        label_pos = mid + norm * (dist * abs(rad) * 0.5) + np.array(label_offset)
        ax.text(label_pos[0], label_pos[1], label, color=color, fontsize=9.5, ha="center", va="center", zorder=6)
    return arrow

fig, ax = plt.subplots(figsize=(8.8, 9.4), dpi=300)
ax.set_xlim(-4.6, 4.6)
ax.set_ylim(-4.8, 4.8)
ax.set_aspect("equal")
ax.axis("off")

ax.text(0, 4.3, "Causal Graph of QCRL & Generative Inpainting Pipeline", fontsize=13.5, fontweight="bold", ha="center")
ax.text(0, 3.95, "Conditioned Anatomical Localization $\\mathbf{M}$ and Physical Intervention $do(I = I_{\\mathrm{cf}})$",
        fontsize=9.5, color="#475569", ha="center")

R = 0.52

# Node positions
C_pos    = (0.0, 2.8)
I_pos    = (-2.4, 1.4)
Q_pos    = (2.4, 1.4)

V_pos    = (-2.4, -0.2)
Qtok_pos = (2.4, -0.2)

S_pos    = (0.0, -0.2)   # Cross-attention affinity S
M_pos    = (0.0, -1.6)   # Spatial ROI Mask M

G_pos    = (-1.6, -3.0)  # Inpainting generator G_phi
Icf_pos  = (0.0, -4.1)   # Counterfactual scan I_cf

# Draw Nodes
draw_node(ax, C_pos, R, "$C$", COL_LATENT, dashed=True)
draw_node(ax, I_pos, R, "$I$", COL_OBSERVED)
draw_node(ax, Q_pos, R, "$Q$", COL_OBSERVED)

draw_node(ax, V_pos, R, "$\\mathbf{V}$", COL_OBSERVED)
draw_node(ax, Qtok_pos, R, "$\\mathbf{Q}$", COL_OBSERVED)

draw_node(ax, S_pos, R, "$\\mathbf{S}$", COL_SCALE, font_size=12)
draw_node(ax, M_pos, R, "$\\mathbf{M}$", COL_MASK, font_size=13, font_weight="bold")

draw_node(ax, G_pos, R, "$G_\\phi$", COL_INPAINT, font_size=12, font_weight="bold")
draw_node(ax, Icf_pos, R, "$I_{\\mathrm{cf}}$", COL_CF, font_size=13, font_weight="bold")

# Edges:
# 1. C -> I (Severed by intervention)
draw_dag_arrow(ax, C_pos, I_pos, R, R, color="#94A3B8", linestyle=":")
mid_ci = (np.array(C_pos) + np.array(I_pos)) / 2.0
ax.text(mid_ci[0], mid_ci[1], "$\\times$", fontsize=19, color="red", fontweight="bold", ha="center", va="center")
ax.text(mid_ci[0] - 0.48, mid_ci[1] + 0.18, "$do(I)$", fontsize=9.5, color="red", fontweight="bold", ha="center")

# 2. C -> Q
draw_dag_arrow(ax, C_pos, Q_pos, R, R)

# 3. I -> V and Q -> Q_tokens
draw_dag_arrow(ax, I_pos, V_pos, R, R)
ax.text(-2.7, 0.6, "ViT", fontsize=8.5, color="#475569", ha="right")

draw_dag_arrow(ax, Q_pos, Qtok_pos, R, R)
ax.text(2.7, 0.6, "PubMedBERT", fontsize=8.5, color="#475569", ha="left")

# 4. Cross-Attention: V -> S and Q -> S
draw_dag_arrow(ax, V_pos, S_pos, R, R, color="#D97706")
draw_dag_arrow(ax, Qtok_pos, S_pos, R, R, color="#D97706")
ax.text(0.0, 0.45, "$\\mathbf{S}_{i,j} = \\mathrm{Softmax}\\left(\\frac{(\\mathbf{Q}_i \\mathbf{W}_q)(\\mathbf{V}_j \\mathbf{W}_v)^T}{\\sqrt{D}}\\right)$",
        fontsize=8.5, color="#B45309", ha="center")

# 5. S -> M (Spatial aggregation & interpolation)
draw_dag_arrow(ax, S_pos, M_pos, R, R, color="#D97706", lw=1.6)
ax.text(0.25, -0.9, "$\\mathrm{Pool} + \\mathrm{Interp}$", fontsize=8, color="#B45309", ha="left")

# 6. Inpainting: M -> G_phi
draw_dag_arrow(ax, M_pos, G_pos, R, R, color="#7C3AED", lw=1.5)

# I -> G_phi (Arching smoothly around the outside left of V)
draw_curved_dag_arrow(ax, I_pos, G_pos, R, R, rad=-0.36, color="#7C3AED", lw=1.3, linestyle="--",
                      label="Original $I$", label_offset=(-0.45, 0.0))

# 7. Synthesis: G_phi -> I_cf
draw_dag_arrow(ax, G_pos, Icf_pos, R, R, color="#059669", lw=1.6)
ax.text(-0.7, -3.8, "$\\mathbf{M} \\odot G_\\phi(I, \\mathbf{M})$", fontsize=8.5, color="#059669", ha="center")

# Background preservation: I -> I_cf (Arching from I to I_cf)
draw_curved_dag_arrow(ax, I_pos, Icf_pos, R, R, rad=0.25, color="#059669", lw=1.4, linestyle=":",
                      label="$(1 - \\mathbf{M}) \\odot I$", label_offset=(0.25, 0.2))

# Inpainting equation box
ax.text(1.3, -2.8,
        "\\textbf{Physical} $do$-Intervention:\\\\\n"
        "$I_{\\mathrm{cf}} = (1 - \\mathbf{M}) \\odot I$\\\\\n"
        "$\\quad\\; + \\mathbf{M} \\odot G_\\phi(I, \\mathbf{M})$\\\\\n"
        "\\vspace{1pt}\\\\\n"
        "$\\bullet$ Neutralizes lesion tissue\\\\\n"
        "$\\bullet$ Preserves healthy anatomy",
        fontsize=8.8, color="#065F46", ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ECFDF5", edgecolor="#059669", lw=1.0))

plt.tight_layout()
out_dag_png = "Manuscript TMI/fig_qcrl_dag.png"
out_dag_pdf = "Manuscript TMI/fig_qcrl_dag.pdf"
plt.savefig(out_dag_png, dpi=300, bbox_inches="tight")
plt.savefig(out_dag_pdf, dpi=300, bbox_inches="tight")
plt.close()
print(f"Generated clean non-intersecting QCRL Causal DAG:")
print(f"  1. {out_dag_png}")
print(f"  2. {out_dag_pdf}")
