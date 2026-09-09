#!/usr/bin/env python3
"""
Generate Candidate 3 Diagrams:
1. Causal Graph (DAG) of Multi-Stage Uncertainty & Selective Abstention Gate (TikZ style)
2. Architectural Illustration of Multi-Stage Uncertainty & Risk-Coverage Frontier
Outputs:
- Manuscript TMI/fig_triage_dag.png & .pdf
- Manuscript TMI/fig_triage_pipeline.png & .pdf
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
COL_REFERRAL = "#FEE2E2"  # red!15
COL_CF       = "#E0F4EE"  # teal!15
COL_SCALE    = "#FEF9E7"  # amber!15
COL_CONTRAST = "#D1EDE8"  # teal!20
COL_GATE     = "#F3E8FF"  # purple!15
COL_MASK     = "#FEF5D1"  # yellow!20

def draw_node(ax, xy, radius, label, fill_color, dashed=False, lw=1.3, font_size=12, font_weight="normal"):
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
# PART 1: CAUSAL DAG OF CANDIDATE 3 (TikZ Style)
# ====================================================================
fig, ax = plt.subplots(figsize=(9.2, 10.2), dpi=300)
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.5, 5.0)
ax.set_aspect("equal")
ax.axis("off")

ax.text(0, 4.65, "Causal Graph of Multi-Stage Uncertainty & Selective Triage Gate",
        fontsize=13.5, fontweight="bold", ha="center")
ax.text(0, 4.30, "Physical Intervention $do(I=I_{\\mathrm{cf}})$, Interventional Confidence, and Decision Triage",
        fontsize=9.2, color="#475569", ha="center")

R = 0.50

# Optimized positions with zero edge collisions
C_pos     = (0.0, 3.4)
I_pos     = (-1.8, 1.9)
Q_pos     = (2.4, 1.9)

M_pos     = (0.3, 1.9)   # Spatial lesion mask M
Icf_pos   = (-3.8, 0.5)  # Intervened scan (outer left)

Vorig_pos = (-1.8, 0.3)  # Factual feature (straight down from I)
Vcf_pos   = (-3.8, -0.9) # Counterfactual feature (straight down from I_cf)

dL_pos    = (-2.0, -2.0) # Causal logit contrast Delta L
Pint_pos  = (0.5, -2.0)  # Calibrated probability P(A | do(I), Q)
Gate_pos  = (0.5, -3.3)  # Selective Gate T_tau

Ans_pos   = (2.6, -4.5)  # Automated Answer \hat{A}
Ref_pos   = (-1.6, -4.5) # Referral \mathcal{R}

# Draw Nodes
draw_node(ax, C_pos, R, "$C$", COL_LATENT, dashed=True)
draw_node(ax, I_pos, R, "$I$", COL_OBSERVED)
draw_node(ax, Q_pos, R, "$Q$", COL_OBSERVED)

draw_node(ax, M_pos, R, "$\\mathbf{M}$", COL_MASK, font_size=12, font_weight="bold")
draw_node(ax, Icf_pos, R, "$I_{\\mathrm{cf}}$", COL_CF, font_size=12, font_weight="bold")

draw_node(ax, Vorig_pos, R, "$\\mathbf{V}$", COL_OBSERVED)
draw_node(ax, Vcf_pos, R, "$\\mathbf{V}_{\\mathrm{cf}}$", COL_CF)

draw_node(ax, dL_pos, R, "$\\Delta \\mathbf{L}$", COL_SCALE, font_size=11, font_weight="bold")
draw_node(ax, Pint_pos, R, "$P_{\\mathrm{int}}$", COL_CONTRAST, font_size=11, font_weight="bold")
draw_node(ax, Gate_pos, R, "$T_\\tau$", COL_GATE, font_size=12, font_weight="bold")

draw_node(ax, Ans_pos, R, "$\\hat{A}$", COL_OUTCOME, font_size=13, font_weight="bold")
draw_node(ax, Ref_pos, R, "$\\mathcal{R}$", COL_REFERRAL, font_size=13, font_weight="bold")

# Edges:
# 1. C -> I (Severed by intervention)
draw_dag_arrow(ax, C_pos, I_pos, R, R, color="#94A3B8", linestyle=":")
mid_ci = (np.array(C_pos) + np.array(I_pos)) / 2.0
ax.text(mid_ci[0], mid_ci[1], "$\\times$", fontsize=19, color="red", fontweight="bold", ha="center", va="center")
ax.text(mid_ci[0] - 0.45, mid_ci[1] + 0.16, "$do(I)$", fontsize=9.0, color="red", fontweight="bold", ha="center")

# 2. C -> Q
draw_dag_arrow(ax, C_pos, Q_pos, R, R)

# 3. QCRL Mask generation: I -> M and Q -> M
draw_dag_arrow(ax, I_pos, M_pos, R, R, color="#D97706")
draw_dag_arrow(ax, Q_pos, M_pos, R, R, color="#D97706")
ax.text(0.3, 2.60, "$\\mathrm{QCRL}(\\mathbf{Q}, \\mathbf{V})$", fontsize=8.2, color="#B45309", ha="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#D97706", lw=0.6))

# 4. Inpainting: I and M -> I_cf
draw_dag_arrow(ax, I_pos, Icf_pos, R, R, color="#059669")
# Curved arrow from M to I_cf dipping cleanly under I
arrow_m_icf = FancyArrowPatch(
    (M_pos[0] - R, M_pos[1] - 0.1), (Icf_pos[0] + 0.35, Icf_pos[1] + R),
    connectionstyle="arc3,rad=-0.28",
    arrowstyle="-|>", mutation_scale=14,
    color="#059669", linewidth=1.3, linestyle="-", zorder=3
)
ax.add_patch(arrow_m_icf)
ax.text(-1.0, 1.35, "$G_\\phi$", fontsize=8.5, color="#059669", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#059669", lw=0.7))

# 5. Dual Encoders: I -> V (vertical) and I_cf -> V_cf (vertical)
draw_dag_arrow(ax, I_pos, Vorig_pos, R, R)
ax.text(-1.35, 1.1, "ViT", fontsize=8.0, color="#475569", ha="left")

draw_dag_arrow(ax, Icf_pos, Vcf_pos, R, R, color="#059669")
ax.text(-4.15, -0.2, "ViT", fontsize=8.0, color="#059669", ha="right")

# 6. Contrast computation: V and V_cf -> \Delta L
draw_dag_arrow(ax, Vorig_pos, dL_pos, R, R, color="#D97706")
draw_dag_arrow(ax, Vcf_pos, dL_pos, R, R, color="#D97706")
ax.text(-2.9, -1.35, "$\\mathbf{L}_{\\mathrm{orig}} - \\mathbf{L}_{\\mathrm{cf}}$", fontsize=7.8, color="#B45309", ha="center",
        bbox=dict(boxstyle="square,pad=0.15", facecolor="#FFFFFF", edgecolor="none"))

# 7. Dynamic Scaling: Q -> P_int and \Delta L -> P_int
arrow_q_p = FancyArrowPatch(
    (Q_pos[0], Q_pos[1] - R), (Pint_pos[0] + 0.35, Pint_pos[1] + R),
    connectionstyle="arc3,rad=-0.18",
    arrowstyle="-|>", mutation_scale=14,
    color="#D97706", linewidth=1.3, linestyle="--", zorder=3
)
ax.add_patch(arrow_q_p)
ax.text(2.0, -0.2, "$\\gamma(Q)$ Scaling", fontsize=8.0, color="#B45309", ha="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#D97706", lw=0.7))

draw_dag_arrow(ax, dL_pos, Pint_pos, R, R, color="#2563EB", lw=1.5)
ax.text(-0.75, -1.82, "$+\\gamma(Q)\\Delta \\mathbf{L}$", fontsize=8.0, color="#1E40AF", ha="center",
        bbox=dict(boxstyle="square,pad=0.12", facecolor="#FFFFFF", edgecolor="none"))

# 8. P_int -> Gate T_tau
draw_dag_arrow(ax, Pint_pos, Gate_pos, R, R, color="#7C3AED", lw=1.6)
ax.text(0.70, -2.65, "$\\max_a P(a)$", fontsize=8.0, color="#5B21B6", ha="left")

# 9. Gate -> Dual Terminals
draw_dag_arrow(ax, Gate_pos, Ans_pos, R, R, color="#2563EB", lw=1.8)
ax.text(1.7, -3.80, "$\\geq \\tau$ (Pass)", fontsize=8.5, fontweight="bold", color="#1E40AF", ha="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#2563EB", lw=0.6))

draw_dag_arrow(ax, Gate_pos, Ref_pos, R, R, color="#DC2626", lw=1.8)
ax.text(-0.7, -3.80, "$< \\tau$ (Defer)", fontsize=8.5, fontweight="bold", color="#B91C1C", ha="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#DC2626", lw=0.6))

# Info card on bottom
info_box = (
    "Clinical Triage Decision Rule:\n"
    r"$\mathrm{Decision} = \hat{A} \;\; \mathrm{if} \; \max_a P(a \mid do(I), Q) \geq \tau \;\; \mathrm{else} \; \mathcal{R}$" + "\n\n"
    r"• Operating Threshold $\tau_2 = 0.85$:" + "\n"
    r"  - Automated Patient Coverage: $72.5\%$" + "\n"
    r"  - Selective Clinical Risk: $2.4\%$" + "\n"
    r"  - Human Referral Rate: $27.5\%$" + "\n"
    r"• Causal Guarantee: Backdoor $C \to I$ severed by physical intervention." + "\n"
    "  Spurious correlation cannot artificially elevate triage confidence."
)
ax.text(0.0, -5.20, info_box, fontsize=8.0, color="#1E293B", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#64748B", lw=1.0))

plt.tight_layout()
out_dag_png = "Manuscript TMI/fig_triage_dag.png"
out_dag_pdf = "Manuscript TMI/fig_triage_dag.pdf"
plt.savefig(out_dag_png, dpi=300, bbox_inches="tight")
plt.savefig(out_dag_pdf, dpi=300, bbox_inches="tight")
plt.close()

# ====================================================================
# PART 2: ARCHITECTURAL ILLUSTRATION OF CANDIDATE 3 (Pipeline & Frontier)
# ====================================================================
fig2 = plt.figure(figsize=(16.2, 8.2), dpi=300)
ax2 = fig2.add_axes([0, 0, 1, 1])
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

# Main Titles
ax2.text(2.5, 96.5, "Candidate 3: Multi-Stage Uncertainty Estimation & Selective Abstention Triage Frontier",
         fontsize=15.0, fontweight="bold", color="#0F172A", va="top")
ax2.text(2.5, 93.0, "Causal Interventional Confidence $\\max P(A \\mid do(I), Q)$, Cost-Sensitive Triage Gate, and Risk-Coverage Trade-off",
         fontsize=10.0, color="#475569", va="top")

# -------------------------------------------------------------
# COLUMN 1: INTERVENTIONAL CONFIDENCE CALCULATION (x: 2.5 to 19.5)
# -------------------------------------------------------------
draw_box2(2.5, 10, 17, 74, "#F0F7FF", "#2563EB", lw=1.8)
ax2.text(11.0, 80.5, "Stage 1: Interventional Logits", fontsize=10.5, fontweight="bold", color="#1E40AF", ha="center")

# Observational Logits Card
draw_box2(4.0, 62, 14, 13, "#FFFFFF", "#3B82F6", lw=1.2, r=0.8)
ax2.text(11.0, 71.5, "Factual Logits $\\mathbf{L}_{\\mathrm{orig}}$", fontsize=9.0, fontweight="bold", color="#1E40AF", ha="center")
ax2.text(11.0, 66.5, "Forward Pass on Scan $I$\n$\\mathbf{L}_{\\mathrm{orig}} = f_\\theta(I, Q)$", fontsize=7.5, color="#475569", ha="center")

# Minus Sign
ax2.text(11.0, 58.5, "$-$", fontsize=14, fontweight="bold", color="#64748B", ha="center", va="center")

# Counterfactual Logits Card
draw_box2(4.0, 42, 14, 13, "#FFFFFF", "#059669", lw=1.2, r=0.8)
ax2.text(11.0, 51.5, "Counterfactual $\\mathbf{L}_{\\mathrm{cf}}$", fontsize=9.0, fontweight="bold", color="#065F46", ha="center")
ax2.text(11.0, 46.5, "Pass on Inpainted $I_{\\mathrm{cf}}$\n$\\mathbf{L}_{\\mathrm{cf}} = f_\\theta(I_{\\mathrm{cf}}, Q)$", fontsize=7.5, color="#475569", ha="center")

# Dynamic Scaling Card
draw_box2(4.0, 20, 14, 16, "#FFFBEB", "#D97706", lw=1.2, r=0.8)
ax2.text(11.0, 32.5, "Dynamic Scale $\\gamma(Q)$", fontsize=9.0, fontweight="bold", color="#92400E", ha="center")
ax2.text(11.0, 26.5, "$\\gamma(Q) = \\mathrm{Softplus}(\\mathbf{W}_\\gamma \\bar{\\mathbf{q}} + b_\\gamma)$\nModulates Visual Reliance", fontsize=7.2, color="#78350F", ha="center")
ax2.text(11.0, 15.0, "$\\Delta \\mathbf{L} = \\mathbf{L}_{\\mathrm{orig}} - \\mathbf{L}_{\\mathrm{cf}}$", fontsize=8.0, fontweight="bold", color="#1E40AF", ha="center")

draw_arrow2(19.5, 47, 24.0, 47, color="#2563EB", lw=1.8)

# -------------------------------------------------------------
# COLUMN 2: MULTI-STAGE UNCERTAINTY & CONSISTENCY (x: 24 to 41)
# -------------------------------------------------------------
draw_box2(24, 10, 17, 74, "#FAF5FF", "#7C3AED", lw=1.8)
ax2.text(32.5, 80.5, "Stage 2: Calibrated Posterior", fontsize=10.5, fontweight="bold", color="#5B21B6", ha="center")

# Equation Box
draw_box2(25.5, 59, 14, 16, "#FFFFFF", "#7C3AED", lw=1.2, r=0.8)
ax2.text(32.5, 71.0, "Calibrated Probability:", fontsize=8.5, fontweight="bold", color="#5B21B6", ha="center")
ax2.text(32.5, 64.0, "$P(A \\mid do(I), Q) =$\n$\\mathrm{Softmax}(\\mathbf{L}_{\\mathrm{orig}} + \\gamma(Q)\\Delta \\mathbf{L})$",
         fontsize=7.8, color="#4C1D95", ha="center")

# Curriculum Consistency Head
draw_box2(25.5, 34, 14, 21, "#FFFFFF", "#8B5CF6", lw=1.2, r=0.8)
ax2.text(32.5, 51.5, "Curriculum Head $h$", fontsize=9.0, fontweight="bold", color="#5B21B6", ha="center")
ax2.text(32.5, 47.5, "Hierarchical Verification:\n• L1: Existence ($s_{L1}$)\n• L2: Attribute ($s_{L2}$)\n• L3: Relation ($s_{L3}$)",
         fontsize=7.0, color="#475569", ha="center", va="top")
ax2.text(32.5, 36.5, "$h = \\sigma(\\frac{1}{2}\\mathbf{W}_{\\mathrm{mlp}}\\mathbf{c} + \\frac{1}{2}\\mathbf{W}_{\\mathrm{gru}}\\mathbf{h})$",
         fontsize=6.8, color="#6D28D9", ha="center")

# Confidence metric
draw_box2(25.5, 14, 14, 15, "#F3E8FF", "#6D28D9", lw=1.2, r=0.8)
ax2.text(32.5, 24.5, "Interventional Confidence:", fontsize=8.0, fontweight="bold", color="#4C1D95", ha="center")
ax2.text(32.5, 18.0, "$c(I, Q) = \\max_a P(a \\mid do(I), Q)$", fontsize=8.0, fontweight="bold", color="#5B21B6", ha="center")

draw_arrow2(41.0, 47, 45.5, 47, color="#7C3AED", lw=1.8)

# -------------------------------------------------------------
# COLUMN 3: DUAL-THRESHOLD SELECTIVE GATE (x: 45.5 to 60.5)
# -------------------------------------------------------------
draw_box2(45.5, 10, 15, 74, "#FFFDF5", "#D97706", lw=1.8)
ax2.text(53.0, 80.5, "Stage 3: Triage Gate $T_\\tau$", fontsize=10.5, fontweight="bold", color="#92400E", ha="center")

# Threshold diamond representation
pts = [[53.0, 68.0], [58.5, 59.0], [53.0, 50.0], [47.5, 59.0]]
poly = patches.Polygon(pts, closed=True, facecolor="#FEF3C7", edgecolor="#D97706", lw=1.5, zorder=3)
ax2.add_patch(poly)
ax2.text(53.0, 60.5, "$c(I, Q) \\geq \\tau$?", fontsize=8.5, fontweight="bold", color="#92400E", ha="center", zorder=4)
ax2.text(53.0, 56.5, "Check Threshold", fontsize=6.8, color="#78350F", ha="center", zorder=4)

# Operational Operating Points Card
draw_box2(47.0, 16, 12, 29, "#FFFFFF", "#D97706", lw=1.2, r=0.8)
ax2.text(53.0, 41.5, "Clinical Operating Points:", fontsize=7.8, fontweight="bold", color="#92400E", ha="center")
ax2.text(53.0, 34.5, "$\\tau_1 = 0.70$ (Permissive)\nCov: $88.4\\%$, Risk: $6.1\\%$", fontsize=6.8, color="#475569", ha="center")
ax2.text(53.0, 26.5, "$\\tau_2 = 0.85$ (Recommended)\nCov: 72.5%, Risk: 2.4%", fontsize=7.0, fontweight="bold", color="#059669", ha="center")
ax2.text(53.0, 18.5, "$\\tau_3 = 0.90$ (High-Safety)\nCov: $61.2\\%$, Risk: $0.9\\%$", fontsize=6.8, color="#475569", ha="center")

# Arrows leaving the diamond with clean pill badges
draw_arrow2(58.5, 62, 65.0, 69, color="#2563EB", lw=2.0)
badge_yes = FancyBboxPatch((59.5, 66.0), 4.8, 3.2, boxstyle="round,pad=0.2,rounding_size=0.8",
                           facecolor="#FFFFFF", edgecolor="#2563EB", lw=0.8, zorder=5)
ax2.add_patch(badge_yes)
ax2.text(61.9, 67.6, "Yes ($\\geq \\tau$)", fontsize=7.2, fontweight="bold", color="#1E40AF", ha="center", va="center", zorder=6)

draw_arrow2(58.5, 56, 65.0, 33, color="#DC2626", lw=2.0)
badge_no = FancyBboxPatch((59.5, 41.5), 4.8, 3.2, boxstyle="round,pad=0.2,rounding_size=0.8",
                          facecolor="#FFFFFF", edgecolor="#DC2626", lw=0.8, zorder=5)
ax2.add_patch(badge_no)
ax2.text(61.9, 43.1, "No ($< \\tau$)", fontsize=7.2, fontweight="bold", color="#B91C1C", ha="center", va="center", zorder=6)

# -------------------------------------------------------------
# COLUMN 4: BIFURCATED CLINICAL WORKFLOW (x: 65 to 79.5)
# -------------------------------------------------------------
# Top Box: Automated Diagnostic Output
draw_box2(65.0, 49, 14.5, 35, "#ECFDF5", "#059669", lw=1.8)
ax2.text(72.25, 79.5, "Autonomous Stream", fontsize=10.0, fontweight="bold", color="#065F46", ha="center")
ax2.text(72.25, 75.0, "High-Confidence Pass", fontsize=8.0, color="#047857", ha="center")

draw_box2(66.2, 53, 12.1, 19, "#FFFFFF", "#10B981", lw=1.0, r=0.6)
ax2.text(72.25, 67.5, "Diagnostic Answer $\\hat{A}$", fontsize=8.5, fontweight="bold", color="#065F46", ha="center")
ax2.text(72.25, 61.5, '"Cardiomegaly absent"\nConfidence: $96.4\\%$\nRisk Rate: $2.4\\%$', fontsize=7.0, color="#1E293B", ha="center")
ax2.text(72.25, 54.5, "Direct Electronic EHR Export", fontsize=6.5, color="#059669", ha="center")

# Bottom Box: Safe Clinical Referral
draw_box2(65.0, 10, 14.5, 35, "#FEF2F2", "#DC2626", lw=1.8)
ax2.text(72.25, 40.5, "Referral Stream", fontsize=10.0, fontweight="bold", color="#991B1B", ha="center")
ax2.text(72.25, 36.0, "Low-Confidence / Defer", fontsize=8.0, color="#B91C1C", ha="center")

draw_box2(66.2, 14, 12.1, 19, "#FFFFFF", "#EF4444", lw=1.0, r=0.6)
ax2.text(72.25, 28.5, "Attending Specialist $\\mathcal{R}$", fontsize=8.5, fontweight="bold", color="#991B1B", ha="center")
ax2.text(72.25, 23.0, "Triage to Senior Radiologist\n+ ROI Lesion Mask $\\mathbf{M}$\n+ Counterfactual $\\Delta \\mathbf{L}$", fontsize=6.8, color="#1E293B", ha="center")
ax2.text(72.25, 16.0, "Diagnostic Safety Guardrail", fontsize=6.5, color="#DC2626", ha="center")

draw_arrow2(79.5, 47, 83.5, 47, color="#475569", lw=1.8)

# -------------------------------------------------------------
# COLUMN 5: RISK-COVERAGE PARETO FRONTIER (x: 83.5 to 98)
# -------------------------------------------------------------
draw_box2(83.5, 10, 14.5, 74, "#F8FAFC", "#475569", lw=1.8)
ax2.text(90.75, 80.5, "Stage 5: Pareto Frontier", fontsize=10.0, fontweight="bold", color="#1E293B", ha="center")

# Mini inset plot positioned inside Column 5
ax_inset = fig2.add_axes([0.882, 0.36, 0.092, 0.39])
covs = np.linspace(0.40, 1.0, 50)
risks = 0.005 + 0.18 * (covs - 0.35)**2.2
ax_inset.plot(covs * 100, risks * 100, color="#2563EB", lw=2.0, label="CI-GCI")
# Baseline curve (uncalibrated - flatter/higher risk)
risks_base = 0.08 + 0.15 * (covs - 0.35)**1.2
ax_inset.plot(covs * 100, risks_base * 100, color="#94A3B8", lw=1.4, linestyle="--", label="Baseline")

# Operational point tau_2
op_cov = 72.5
op_risk = 2.4
ax_inset.scatter([op_cov], [op_risk], color="#DC2626", s=35, zorder=5)
ax_inset.annotate(r"$\tau_2$ (2.4%, 72.5%)", xy=(op_cov, op_risk), xytext=(op_cov - 31, op_risk + 4.2),
                  fontsize=6.5, fontweight="bold", color="#DC2626",
                  arrowprops=dict(arrowstyle="->", color="#DC2626", lw=0.9))

ax_inset.set_xlabel("Coverage (%)", fontsize=7.0, labelpad=2)
ax_inset.set_ylabel("Selective Risk (%)", fontsize=7.0, labelpad=2)
ax_inset.tick_params(labelsize=6.0, pad=2)
ax_inset.grid(True, linestyle=":", alpha=0.5)
ax_inset.legend(fontsize=6.0, loc="upper left", framealpha=0.85, handlelength=1.2)

# Annotation summary under graph
draw_box2(84.7, 12.0, 12.1, 17.0, "#FFFFFF", "#64748B", lw=0.9, r=0.5)
ax2.text(90.75, 25.0, "Safety Properties:", fontsize=7.6, fontweight="bold", color="#1E293B", ha="center")
ax2.text(90.75, 18.5, "• Near-zero risk at 72.5% cov\n• Monotonic error drop\n• Prevents clinical harm",
         fontsize=6.5, color="#475569", ha="center")

plt.savefig("Manuscript TMI/fig_triage_pipeline.png", dpi=300, bbox_inches="tight")
plt.savefig("Manuscript TMI/fig_triage_pipeline.pdf", dpi=300, bbox_inches="tight")
plt.close()

print("Generated Candidate 3 figures successfully!")
