#!/usr/bin/env python3
"""
Generate publication-grade schematics and flowcharts for the
Autonomous SAC-MLIP High-Throughput Pipeline.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

os.makedirs("figures", exist_ok=True)

# Shared typography and color palette
COLOR_DATA = "#2b5c8f"        # Steel blue (DFT / Sampling)
COLOR_ML = "#d95f02"          # Amber orange (MACE / Training)
COLOR_KINETICS = "#7570b3"    # Royal purple (CI-NEB / Thermo)
COLOR_AL = "#e7298a"          # Magenta / Pink (Active Learning)
COLOR_BG = "#f8f9fa"          # Clean card background
COLOR_TEXT = "#212529"

# =====================================================================
# FIGURE 1: Full Architecture & 10 Coordination Shells
# =====================================================================
fig = plt.figure(figsize=(16, 10), facecolor="white")
gs = GridSpec(2, 1, height_ratios=[1.1, 1.0], hspace=0.32)

# --- Top Subplot: Pipeline Phases ---
ax_arch = fig.add_subplot(gs[0])
ax_arch.set_xlim(0, 100)
ax_arch.set_ylim(0, 42)
ax_arch.axis("off")
ax_arch.set_title("Autonomous Single-Atom Catalyst (SAC) MLIP Workflow Architecture",
                  fontsize=14, fontweight="bold", pad=15, color=COLOR_TEXT)

stages = [
    ("Phase 0 & 1", "Substrate Synthesis\n& 10 Motifs", "Divacancy build\nBFGS pre-relax", COLOR_DATA),
    ("Phase 2 & 3", "DFT PES Sampling\n(GPAW PBE-D3)", "Rattles, Approaches,\nCleavage, AIMD", COLOR_DATA),
    ("Phase 4 & 5", "Data Split &\nZero-Shot Baseline", "85/15 ExtXYZ split\nMACE-MP-0 test", COLOR_ML),
    ("Phase 6", "Programmatic MACE\nFine-Tuning", "ScaleShiftMACE\nLoss backprop", COLOR_ML),
    ("Phase 7", "CI-NEB Kinetics &\nThermochemistry", "MEP barrier ($E_a$)\n$\Delta G(T, P)$ via hTST", COLOR_KINETICS),
    ("Phase 8 & 9", "Active Learning MD\n& Publication Plots", "20 ps tripwire MD\nParity & Volcano", COLOR_AL),
]

box_w, box_h, y_box = 13.5, 26, 8
for idx, (p_tag, p_title, p_desc, col) in enumerate(stages):
    x_box = 3.5 + idx * 16.0

    # Main Card
    rect = patches.FancyBboxPatch((x_box, y_box), box_w, box_h,
                                  boxstyle="round,pad=0.8,rounding_size=1.2",
                                  facecolor=COLOR_BG, edgecolor=col, linewidth=2.0)
    ax_arch.add_patch(rect)

    # Header Tag Banner
    tag_rect = patches.FancyBboxPatch((x_box + 0.5, y_box + box_h - 4.5), box_w - 1.0, 3.8,
                                      boxstyle="round,pad=0.2,rounding_size=0.8",
                                      facecolor=col, edgecolor="none")
    ax_arch.add_patch(tag_rect)
    ax_arch.text(x_box + box_w/2, y_box + box_h - 2.6, p_tag,
                 ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")

    # Title & Subtitle
    ax_arch.text(x_box + box_w/2, y_box + box_h - 8.5, p_title,
                 ha="center", va="center", color=COLOR_TEXT, fontsize=9.5, fontweight="bold")
    ax_arch.text(x_box + box_w/2, y_box + 6.0, p_desc,
                 ha="center", va="center", color="#495057", fontsize=8.2, style="italic")

    # Forward arrows between stages
    if idx < len(stages) - 1:
        ax_arch.annotate("", xy=(x_box + box_w + 2.2, y_box + box_h/2),
                         xytext=(x_box + box_w + 0.3, y_box + box_h/2),
                         arrowprops=dict(arrowstyle="-|>", color="#6c757d", lw=2.2, mutation_scale=15))

# --- Bottom Subplot: 10 Physical Coordination Shells ---
ax_motifs = fig.add_subplot(gs[1])
ax_motifs.set_xlim(0, 100)
ax_motifs.set_ylim(0, 38)
ax_motifs.axis("off")
ax_motifs.set_title(r"First-Coordination Shell Library: $M\text{-N}_x\text{C}_{4-x}$ (Divacancy) and $M\text{-N}_x\text{C}_{3-x}$ (Single Vacancy)",
                    fontsize=12, fontweight="bold", pad=10, color=COLOR_TEXT)

motif_labels = [
    # 4-Coordinated
    (r"$M\text{-N}_4$", "4 N"), (r"$M\text{-N}_3\text{C}_1$", "3 N, 1 C"),
    (r"$M\text{-N}_2\text{C}_2^{\mathrm{cis}}$", "cis-2N, 2C"), (r"$M\text{-N}_2\text{C}_2^{\mathrm{trans}}$", "trans-2N, 2C"),
    (r"$M\text{-N}_1\text{C}_3$", "1 N, 3 C"), (r"$M\text{-C}_4$", "4 C"),
    # 3-Coordinated
    (r"$M\text{-N}_3$", "3 N"), (r"$M\text{-N}_2\text{C}_1$", "2 N, 1 C"),
    (r"$M\text{-N}_1\text{C}_2$", "1 N, 2 C"), (r"$M\text{-C}_3$", "3 C")
]

m_box_w, m_box_h = 8.5, 26
for i, (m_name, m_sub) in enumerate(motif_labels):
    x_m = 2.0 + i * 9.8
    y_m = 4.0
    is_4coord = i < 6
    border_col = COLOR_DATA if is_4coord else COLOR_KINETICS

    card = patches.FancyBboxPatch((x_m, y_m), m_box_w, m_box_h,
                                  boxstyle="round,pad=0.4,rounding_size=0.8",
                                  facecolor=COLOR_BG, edgecolor=border_col, linewidth=1.5)
    ax_motifs.add_patch(card)

    # Mini Coordination Cavity Schematic
    cx, cy = x_m + m_box_w/2, y_m + m_box_h - 11
    ax_motifs.scatter(cx, cy, color="#d95f02", s=110, zorder=3) # Metal center
    
    if is_4coord:
        coords = [(cx-1.8, cy), (cx+1.8, cy), (cx, cy-1.8), (cx, cy+1.8)]
    else:
        coords = [(cx-1.8, cy-0.9), (cx+1.8, cy-0.9), (cx, cy+1.8)]

    for ox, oy in coords:
        ax_motifs.plot([cx, ox], [cy, oy], color="#adb5bd", lw=1.2, zorder=2)
        ax_motifs.scatter(ox, oy, color="#2b5c8f", s=45, zorder=3)

    ax_motifs.text(cx, y_m + 5.5, m_name, ha="center", va="center",
                   color=COLOR_TEXT, fontsize=8.8, fontweight="bold")
    ax_motifs.text(cx, y_m + 2.2, m_sub, ha="center", va="center",
                   color="#6c757d", fontsize=7.5)

plt.savefig("figures/pipeline_architecture.png", dpi=300, bbox_inches="tight")
plt.close()
print("  -> Saved: figures/pipeline_architecture.png")


# =====================================================================
# FIGURE 2: Computational Organigram & Closed-Loop Active Learning
# =====================================================================
fig2, ax = plt.subplots(figsize=(11, 14), facecolor="white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.set_title("Autonomous Execution Organigram & Active Learning Closed-Loop",
             fontsize=14, fontweight="bold", pad=20, color=COLOR_TEXT)

def draw_block(x, y, w, h, title, subtitle, col, shape="box"):
    if shape == "box":
        patch = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
                                      facecolor=COLOR_BG, edgecolor=col, linewidth=2.0)
    elif shape == "decision":
        patch = patches.Polygon([
            [x + w/2, y + h + 1.5], [x + w + 1.5, y + h/2],
            [x + w/2, y - 1.5], [x - 1.5, y + h/2]
        ], closed=True, facecolor="#fff3cd", edgecolor="#e0a800", linewidth=2.0)
    ax.add_patch(patch)

    if subtitle:
        ax.text(x + w/2, y + h/2 + 1.2, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=COLOR_TEXT)
        ax.text(x + w/2, y + h/2 - 1.8, subtitle, ha="center", va="center",
                fontsize=8.0, color="#495057", style="italic")
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=COLOR_TEXT)

# Flowchart Layout (Top to Bottom)
draw_block(35, 92, 30, 6.0, "Master Configuration (config.yaml)", "System parameters, cutoffs, stage toggles", "#495057")
draw_block(35, 81, 30, 6.5, "Analytical Substrate Synthesis", "Divacancy pore punching + BFGS pre-relaxation", COLOR_DATA)
draw_block(35, 70, 30, 6.5, "10-Motif First-Shell Library", "Permutations of saturated and open cavities", COLOR_DATA)
draw_block(35, 59, 30, 6.5, "Ab Initio Sampling Grid (GPAW PBE-D3)", "Rattles, gas-phase approaches, bond cleavage", COLOR_DATA)
draw_block(10, 48, 28, 6.0, "SQLite Database (catalysis.db)", "Atomic positions, energies, forces, metadata", "#198754")
draw_block(55, 48, 35, 6.0, "MACE Foundation Model Baseline", "Zero-shot out-of-distribution evaluation", COLOR_ML)
draw_block(35, 37, 30, 6.5, "MACE Programmatic Fine-Tuning", "ScaleShiftMACE fitting energy & forces", COLOR_ML)
draw_block(35, 26, 30, 6.5, "Climbing-Image NEB Screening", r"Extracts minimum energy paths ($E_a$, $\Delta E$)", COLOR_KINETICS)
draw_block(35, 15, 30, 6.5, "Harmonic Thermochemistry (hTST)", r"Calculates $\Delta G^\ddagger(298\mathrm{K}, 1\mathrm{bar})$ via Vibrations", COLOR_KINETICS)
draw_block(35, 3, 30, 7.0, "Active-Learning Production MD", r"20 ps Langevin dynamics with $|F_{\max}|$ tripwire", COLOR_AL)

# Connective Flow Arrows
flow_pairs = [
    ((50, 92), (50, 87.5)),
    ((50, 81), (50, 76.5)),
    ((50, 70), (50, 65.5)),
    ((50, 59), (50, 43.5)),
    ((24, 48), (24, 40), (35, 40)),
    ((50, 37), (50, 32.5)),
    ((50, 26), (50, 21.5)),
    ((50, 15), (50, 10.0)),
]

for seg in flow_pairs:
    if len(seg) == 2:
        ax.annotate("", xy=seg[1], xytext=seg[0],
                    arrowprops=dict(arrowstyle="-|>", color="#495057", lw=1.8, mutation_scale=12))
    elif len(seg) == 3:
        ax.plot([seg[0][0], seg[1][0], seg[2][0]], [seg[0][1], seg[1][1], seg[2][1]],
                color="#495057", lw=1.8, ls="--")
        ax.annotate("", xy=seg[2], xytext=(seg[2][0]-1, seg[2][1]),
                    arrowprops=dict(arrowstyle="-|>", color="#495057", lw=1.8, mutation_scale=12))

# Active Learning Loop (Tripwire fallback from MD back to DFT)
al_path_x = [35, 4, 4, 35]
al_path_y = [6.5, 6.5, 62.25, 62.25]
ax.plot(al_path_x, al_path_y, color=COLOR_AL, lw=2.2, ls=":")
ax.annotate("", xy=(35, 62.25), xytext=(32, 62.25),
            arrowprops=dict(arrowstyle="-|>", color=COLOR_AL, lw=2.2, mutation_scale=14))
ax.text(5.5, 35, r"Uncertainty Tripwire ($|F| > 4.5\ \mathrm{eV/\AA}$) $\rightarrow$ DFT Fallback",
        rotation=90, color=COLOR_AL, fontsize=9.0, fontweight="bold")

plt.savefig("figures/pipeline_organigram.png", dpi=300, bbox_inches="tight")
plt.close()
print("  -> Saved: figures/pipeline_organigram.png")