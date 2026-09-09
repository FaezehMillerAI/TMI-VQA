#!/usr/bin/env python3
"""
Generate Candidate 2 Diagrams:
1. Causal DAG of QCRL & Generative Inpainting Pipeline (TikZ circular style)
2. Architectural Illustration of QCRL & Generative Inpainting Pipeline
Outputs:
- Manuscript TMI/fig_qcrl_dag.png & .pdf
- Manuscript TMI/fig_qcrl_pipeline.png & .pdf
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

# ====================================================================
# PART 1: CAUSAL DAG OF CANDIDATE 2 (TikZ Style)
# ====================================================================
fig, ax = plt.subplots(figsize=(8.8, 9.6), dpi=300)
ax.set_xlim(-4.8, 4.8)
ax.set_ylim(-5.0, 5.0)
ax.set_aspect("equal")
ax.axis("off")

ax.text(0, 4.50, "Causal Graph of QCRL & Generative Inpainting Pipeline", fontsize=14, fontweight="bold", ha="center")
ax.text(0, 4.15, "Conditioned Anatomical Localization $\\mathbf{M}$ and Physical Intervention $do(I = I_{\\mathrm{cf}})$",
        fontsize=9.8, color="#475569", ha="center")

R = 0.52

# Node positions
C_pos    = (0.0, 2.9)
I_pos    = (-2.4, 1.4)
Q_pos    = (2.4, 1.4)

V_pos    = (-2.4, -0.2)
Qtok_pos = (2.4, -0.2)

S_pos    = (0.0, -0.2)   # Cross-attention affinity S
M_pos    = (0.0, -1.6)   # Spatial ROI Mask M

G_pos    = (-1.5, -3.1)  # Inpainting generator G_phi
Icf_pos  = (0.0, -4.2)   # Counterfactual scan I_cf

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
ax.text(mid_ci[0], mid_ci[1], "$\\times$", fontsize=20, color="red", fontweight="bold", ha="center", va="center")
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
ax.text(0.0, 0.50, "$\\mathbf{S}_{i,j} = \\mathrm{Softmax}\\left(\\frac{(\\mathbf{Q}_i \\mathbf{W}_q)(\\mathbf{V}_j \\mathbf{W}_v)^T}{\\sqrt{D}}\\right)$",
        fontsize=8.5, color="#B45309", ha="center")

# 5. S -> M (Spatial aggregation & interpolation)
draw_dag_arrow(ax, S_pos, M_pos, R, R, color="#D97706", lw=1.6)
ax.text(0.25, -0.9, "$\\mathrm{Pool} + \\mathrm{Interp}$", fontsize=8, color="#B45309", ha="left")

# 6. Inpainting: M -> G_phi
draw_dag_arrow(ax, M_pos, G_pos, R, R, color="#7C3AED", lw=1.5)
ax.text(-0.75, -2.35, "Mask $\\mathbf{M}$", fontsize=8.0, color="#7C3AED", ha="center", va="center", zorder=6,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#7C3AED", lw=0.7, alpha=0.95))

# 7. I -> G_phi (Curved cleanly outside V on the left)
theta_start = np.radians(205)
p_start = (I_pos[0] + R * np.cos(theta_start), I_pos[1] + R * np.sin(theta_start))
theta_end = np.radians(135)
p_end = (G_pos[0] + R * np.cos(theta_end), G_pos[1] + R * np.sin(theta_end))

arrow_ig = FancyArrowPatch(
    p_start, p_end,
    connectionstyle="arc3,rad=0.36",
    arrowstyle="-|>", mutation_scale=14,
    color="#7C3AED", linewidth=1.4, linestyle="--", zorder=3
)
ax.add_patch(arrow_ig)
ax.text(-3.35, -0.85, "Original $I$", fontsize=8.5, color="#7C3AED", fontweight="bold", ha="center", va="center", zorder=6,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#7C3AED", lw=0.8, alpha=0.95))

# 8. Synthesis: G_phi -> I_cf
draw_dag_arrow(ax, G_pos, Icf_pos, R, R, color="#059669", lw=1.6)

# Information annotation card on lower right
info_text = (
    "Physical do-Intervention:\n"
    r"$I_{\mathrm{cf}} = (1 - \mathbf{M}) \odot I + \mathbf{M} \odot G_\phi(I, \mathbf{M})$" + "\n\n"
    "• ROI Localizer: Cross-modal attention S → mask M\n"
    "• Inpainter: Replaces lesion with healthy texture\n"
    "• Preservation: Retains background anatomy strictly\n"
    "• Severance: Breaks spurious link C → I"
)
ax.text(1.1, -2.9, info_text, fontsize=8.5, color="#065F46", ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#ECFDF5", edgecolor="#059669", lw=1.2))

plt.tight_layout()
out_dag_png = "Manuscript TMI/fig_qcrl_dag.png"
out_dag_pdf = "Manuscript TMI/fig_qcrl_dag.pdf"
plt.savefig(out_dag_png, dpi=300, bbox_inches="tight")
plt.savefig(out_dag_pdf, dpi=300, bbox_inches="tight")
plt.close()

# ====================================================================
# PART 2: ARCHITECTURAL ILLUSTRATION OF QCRL & INPAINTING PIPELINE
# ====================================================================
fig2, ax2 = plt.subplots(figsize=(15.5, 8.6), dpi=300)
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 100)
ax2.axis("off")
fig2.patch.set_facecolor("#FFFFFF")
ax2.set_facecolor("#FFFFFF")

def draw_box2(x, y, w, h, bg_col, edge_col, lw=1.5, r=1.5):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={r},rounding_size=2.0",
        facecolor=bg_col, edgecolor=edge_col, linewidth=lw, zorder=2
    )
    ax2.add_patch(box)
    return box

def draw_arrow2(x1, y1, x2, y2, color="#64748B", lw=2.0, style="->"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, zorder=3
    )
    ax2.add_patch(arrow)
    return arrow

# Title & Subtitle at very top with ample breathing room
ax2.text(2.5, 96.5, "Candidate 2: Question-Conditioned ROI Locator (QCRL) & Generative Inpainting Pipeline",
         fontsize=15.5, fontweight="bold", color="#0F172A", va="top")
ax2.text(2.5, 93.0, "Cross-Modal Spatial Attention Affinity $\\mathbf{S}$, Heatmap $\\mathbf{M}$, and Physical Counterfactual Synthesis $do(I = I_{\\mathrm{cf}})$",
         fontsize=10.2, color="#475569", va="top")

# -------------------------------------------------------------
# COLUMN 1: DUAL INPUTS (x: 2.5 to 16.5)
# -------------------------------------------------------------
draw_box2(2.5, 50, 14, 34, "#F0F7FF", "#2563EB", lw=1.8)
ax2.text(9.5, 80.5, "Factual Scan $I$", fontsize=11, fontweight="bold", color="#1E40AF", ha="center")
rect_img = patches.Rectangle((4.5, 58), 10, 18, facecolor="#0F172A", edgecolor="#334155", lw=1.2, zorder=3)
ax2.add_patch(rect_img)
lung_l = patches.Ellipse((7.5, 67), 2.8, 10.5, facecolor="#1E293B", edgecolor="#475569", lw=0.8, zorder=4)
lung_r = patches.Ellipse((11.5, 67), 2.8, 10.5, facecolor="#1E293B", edgecolor="#475569", lw=0.8, zorder=4)
ax2.add_patch(lung_l)
ax2.add_patch(lung_r)
lesion = patches.Circle((11.5, 64), 2.2, facecolor="#EF4444", alpha=0.9, edgecolor="#FCA5A5", lw=1.2, zorder=5)
ax2.add_patch(lesion)
ax2.text(11.5, 64, "Lesion", fontsize=7, color="white", fontweight="bold", ha="center", va="center", zorder=6)
ax2.text(9.5, 53.0, "$I \\in \\mathbb{R}^{H \\times W \\times 3}$", fontsize=8.5, color="#475569", ha="center")

# Clinical Query Q
draw_box2(2.5, 10, 14, 32, "#FFFBEB", "#D97706", lw=1.8)
ax2.text(9.5, 38.0, "Clinical Query $Q$", fontsize=11, fontweight="bold", color="#92400E", ha="center")
ax2.text(9.5, 29.0, '"Is right lower lobe\nconsolidation present?"', fontsize=8.5, style="italic", color="#78350F", ha="center", va="center")
ax2.text(9.5, 18.0, "$Q = \\{q_1, \\dots, q_N\\}$", fontsize=8.5, color="#92400E", ha="center")
ax2.text(9.5, 13.0, "Natural Language Text", fontsize=8, color="#64748B", ha="center")

# -------------------------------------------------------------
# COLUMN 2: MULTIMODAL FEATURE EXTRACTION (x: 21 to 35)
# -------------------------------------------------------------
draw_box2(21, 50, 14, 34, "#FFFFFF", "#2563EB", lw=1.6)
ax2.text(28, 80.5, "Dual-Scale ViT", fontsize=10.5, fontweight="bold", color="#1E40AF", ha="center")
for i in range(3):
    for j in range(3):
        p = patches.Rectangle((23.5 + j*2.8, 63 + i*3.5), 2.2, 2.5, facecolor="#DBEAFE", edgecolor="#3B82F6", lw=0.8, zorder=3)
        ax2.add_patch(p)
ax2.text(28, 58.0, "Patch Tokens $\\mathbf{V}$", fontsize=9.5, fontweight="bold", color="#1E3A8A", ha="center")
ax2.text(28, 53.0, "$\\mathbf{V} \\in \\mathbb{R}^{L \\times D},\\; L=196$", fontsize=8, color="#475569", ha="center")

draw_box2(21, 10, 14, 32, "#FFFFFF", "#D97706", lw=1.6)
ax2.text(28, 38.0, "PubMedBERT", fontsize=10.5, fontweight="bold", color="#92400E", ha="center")
for t in range(4):
    tok = patches.Rectangle((23.2 + t*2.5, 24), 2.0, 6.0, facecolor="#FEF3C7", edgecolor="#F59E0B", lw=0.8, zorder=3)
    ax2.add_patch(tok)
ax2.text(28, 18.0, "Word Tokens $\\mathbf{Q}$", fontsize=9.5, fontweight="bold", color="#78350F", ha="center")
ax2.text(28, 13.0, "$\\mathbf{Q} \\in \\mathbb{R}^{N \\times D}$", fontsize=8, color="#475569", ha="center")

draw_arrow2(16.5, 67, 21.0, 67, color="#2563EB", lw=1.8)
draw_arrow2(16.5, 26, 21.0, 26, color="#D97706", lw=1.8)

# -------------------------------------------------------------
# COLUMN 3: CROSS-MODAL AFFINITY & MASK LOCALIZATION (x: 40 to 57)
# -------------------------------------------------------------
draw_box2(40, 10, 17, 74, "#FFFDF5", "#059669", lw=1.8)
ax2.text(48.5, 80.5, "QCRL Localization", fontsize=11, fontweight="bold", color="#065F46", ha="center")

ax2.text(48.5, 74.5, "Scaled Dot-Product Affinity", fontsize=8.5, fontweight="bold", color="#047857", ha="center")
ax2.text(48.5, 68.5, "$\\mathbf{S}_{i,j} = \\frac{\\exp((\\mathbf{Q}_i \\mathbf{W}_q)(\\mathbf{V}_j \\mathbf{W}_v)^T / \\sqrt{D})}{\\sum_{k} \\exp((\\mathbf{Q}_i \\mathbf{W}_q)(\\mathbf{V}_k \\mathbf{W}_v)^T / \\sqrt{D})}$",
         fontsize=7.8, color="#064E3B", ha="center")

rect_map = patches.Rectangle((43.5, 50), 10, 13, facecolor="#0F172A", edgecolor="#059669", lw=1.0, zorder=3)
ax2.add_patch(rect_map)
blob = patches.Circle((49.5, 55), 3.0, facecolor="#F59E0B", alpha=0.85, edgecolor="#FBBF24", lw=1.0, zorder=4)
ax2.add_patch(blob)
blob_core = patches.Circle((49.5, 55), 1.5, facecolor="#EF4444", alpha=0.9, zorder=5)
ax2.add_patch(blob_core)
ax2.text(48.5, 46.5, "Attn Affinity $\\mathbf{S} \\in \\mathbb{R}^{N \\times L}$", fontsize=8, color="#065F46", ha="center")

draw_arrow2(48.5, 44.5, 48.5, 39.5, color="#059669", lw=1.5)
ax2.text(48.5, 41.8, "Mean Pool + Bilinear Interp", fontsize=7.5, color="#047857", ha="center")

rect_mask = patches.Rectangle((44.5, 23), 8, 11, facecolor="#000000", edgecolor="#10B981", lw=1.2, zorder=3)
ax2.add_patch(rect_mask)
mask_spot = patches.Circle((49.5, 27.5), 2.2, facecolor="#FFFFFF", alpha=0.95, zorder=4)
ax2.add_patch(mask_spot)
ax2.text(48.5, 19.5, "Spatial Lesion Mask $\\mathbf{M}$", fontsize=8.5, fontweight="bold", color="#047857", ha="center")
ax2.text(48.5, 15.5, "$\\mathbf{M} \\in [0, 1]^{H \\times W}$", fontsize=8, color="#475569", ha="center")

draw_arrow2(35, 67, 40, 67, color="#2563EB", lw=1.6)
draw_arrow2(35, 26, 40, 26, color="#D97706", lw=1.6)

# -------------------------------------------------------------
# COLUMN 4: GENERATIVE COUNTERFACTUAL INPAINTER (x: 62 to 78)
# -------------------------------------------------------------
draw_box2(62, 10, 16, 74, "#F5F3FF", "#7C3AED", lw=1.8)
ax2.text(70, 80.5, "CFI Inpainting", fontsize=11, fontweight="bold", color="#5B21B6", ha="center")
ax2.text(70, 74.5, "Latent Inpainting $G_\\phi$", fontsize=9.5, fontweight="bold", color="#6D28D9", ha="center")

rect_gen = patches.Rectangle((64.5, 59), 11, 10, facecolor="#FFFFFF", edgecolor="#7C3AED", lw=1.5, zorder=3)
ax2.add_patch(rect_gen)
ax2.text(70, 64.0, "$G_\\phi(I, \\mathbf{M})$", fontsize=10, fontweight="bold", color="#5B21B6", ha="center", va="center")
ax2.text(70, 60.5, "Healthy Parenchyma Generator", fontsize=6.8, color="#64748B", ha="center")

ax2.text(70, 52.0, "Physical Dual Synthesis:", fontsize=8.5, fontweight="bold", color="#4C1D95", ha="center")

draw_box2(63.5, 39, 13, 9.0, "#FFFFFF", "#3B82F6", lw=1.0, r=0.5)
ax2.text(70, 44.5, "$(1 - \\mathbf{M}) \\odot I$", fontsize=9, fontweight="bold", color="#1E40AF", ha="center")
ax2.text(70, 40.5, "Healthy Background Retained", fontsize=6.8, color="#64748B", ha="center")

draw_box2(63.5, 23, 13, 9.0, "#FFFFFF", "#10B981", lw=1.0, r=0.5)
ax2.text(70, 28.5, "$\\mathbf{M} \\odot G_\\phi(I, \\mathbf{M})$", fontsize=9, fontweight="bold", color="#065F46", ha="center")
ax2.text(70, 24.5, "Pathology Inpainted to Normal", fontsize=6.8, color="#64748B", ha="center")

ax2.text(70, 35.5, "$+$", fontsize=14, fontweight="bold", color="#7C3AED", ha="center", va="center")

# Arrow from QCRL Mask M to Inpainter
draw_arrow2(57, 28, 62, 28, color="#059669", lw=1.8)

# Dedicated clean horizontal bus line for Original Scan I -> CFI Inpainting:
# Stays strictly below subtitle (y=93) in the dedicated corridor at y=88.5
p_bus_start = (9.5, 84.0)
p_bus_corner1 = (9.5, 88.5)
p_bus_corner2 = (70.0, 88.5)
p_bus_end = (70.0, 84.0)

ax2.plot([p_bus_start[0], p_bus_corner1[0], p_bus_corner2[0], p_bus_end[0]],
         [p_bus_start[1], p_bus_corner1[1], p_bus_corner2[1], p_bus_end[1]],
         color="#2563EB", linestyle="--", linewidth=1.4, zorder=3)
# Arrowhead pointing downward into Column 4
ax2.annotate("", xy=p_bus_end, xytext=(70.0, 84.8),
             arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=1.4, mutation_scale=12))

# Label pill badge centered along the bus line
badge = FancyBboxPatch((32.0, 86.8), 16.0, 3.4, boxstyle="round,pad=0.4,rounding_size=1.0",
                       facecolor="#FFFFFF", edgecolor="#2563EB", lw=1.0, zorder=5)
ax2.add_patch(badge)
ax2.text(40.0, 88.5, "Original Radiograph $I$", fontsize=8.2, fontweight="bold", color="#1E40AF", ha="center", va="center", zorder=6)

# -------------------------------------------------------------
# COLUMN 5: COUNTERFACTUAL SCAN I_cf & CAUSAL INTERVENTION (x: 83 to 98)
# -------------------------------------------------------------
draw_box2(83, 10, 14.5, 74, "#ECFDF5", "#059669", lw=2.0)
ax2.text(90.25, 80.5, "Counterfactual $I_{\\mathrm{cf}}$", fontsize=11, fontweight="bold", color="#065F46", ha="center")

rect_cf = patches.Rectangle((85.25, 58), 10, 18, facecolor="#0F172A", edgecolor="#059669", lw=1.2, zorder=3)
ax2.add_patch(rect_cf)
lung_cf_l = patches.Ellipse((88.25, 67), 2.8, 10.5, facecolor="#1E293B", edgecolor="#475569", lw=0.8, zorder=4)
lung_cf_r = patches.Ellipse((92.25, 67), 2.8, 10.5, facecolor="#1E293B", edgecolor="#475569", lw=0.8, zorder=4)
ax2.add_patch(lung_cf_l)
ax2.add_patch(lung_cf_r)
healthy_patch = patches.Circle((92.25, 64), 2.2, facecolor="#10B981", alpha=0.9, edgecolor="#6EE7B7", lw=1.2, zorder=5)
ax2.add_patch(healthy_patch)
ax2.text(92.25, 64, "Healthy", fontsize=6.8, color="white", fontweight="bold", ha="center", va="center", zorder=6)

ax2.text(90.25, 53.0, "$do(I = I_{\\mathrm{cf}})$", fontsize=10, fontweight="bold", color="#047857", ha="center")
ax2.text(90.25, 49.0, "Physical Intervention", fontsize=8, color="#065F46", ha="center")

draw_box2(84.5, 16, 12, 26, "#FFFFFF", "#059669", lw=1.0, r=0.5)
ax2.text(90.25, 38.0, "Causal Properties:", fontsize=8.5, fontweight="bold", color="#065F46", ha="center")
ax2.text(90.25, 32.0, "$\\bullet$ Lesion neutralized", fontsize=7.5, color="#1E293B", ha="center")
ax2.text(90.25, 27.0, "$\\bullet$ Anatomy preserved", fontsize=7.5, color="#1E293B", ha="center")
ax2.text(90.25, 22.0, "$\\bullet$ High PSNR / SSIM", fontsize=7.5, color="#1E293B", ha="center")

draw_arrow2(78, 44, 83, 58, color="#059669", lw=2.0)

plt.tight_layout()
out_pipe_png = "Manuscript TMI/fig_qcrl_pipeline.png"
out_pipe_pdf = "Manuscript TMI/fig_qcrl_pipeline.pdf"
plt.savefig(out_pipe_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pipe_pdf, dpi=300, bbox_inches="tight")
plt.close()

print("Generated clean Candidate 2 figures successfully!")
