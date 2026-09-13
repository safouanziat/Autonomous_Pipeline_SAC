import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_diagram(title, subtitle, filename, steps_data, arrows_data):
    fig, ax = plt.subplots(figsize=(10, 13), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')
    
    # Header
    rect_head = patches.FancyBboxPatch((0.8, 12.1), 8.4, 0.65, boxstyle="round,pad=0.06,rounding_size=0.08",
                                       facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.5, zorder=2)
    ax.add_patch(rect_head)
    ax.text(5.0, 12.52, title, ha='center', va='center', fontsize=12, fontweight='bold', color='#0f172a', zorder=3)
    ax.text(5.0, 12.28, subtitle, ha='center', va='center', fontsize=9.5, color='#64748b', zorder=3)

    # Draw boxes
    for s in steps_data:
        x, y, w, h = s['coords']
        bg = s.get('bg', '#ffffff')
        border = s.get('border', '#334155')
        t_col = s.get('t_col', '#0f172a')
        b_col = s.get('b_col', '#334155')
        
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.1",
                                    facecolor=bg, edgecolor=border, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        
        if s.get('title'):
            ax.text(x + w/2, y + h - 0.32, s['title'], ha='center', va='center', fontsize=10.5, fontweight='bold', color=t_col, zorder=3)
            ax.text(x + w/2, y + (h - 0.35)/2, s['text'], ha='center', va='center', fontsize=8.5, color=b_col, zorder=3)
        else:
            ax.text(x + w/2, y + h/2, s['text'], ha='center', va='center', fontsize=9.5, fontweight='bold', color=t_col, zorder=3)

    # Draw arrows
    for arr in arrows_data:
        p1 = arr['from']
        p2 = arr['to']
        color = arr.get('color', '#475569')
        ls = arr.get('ls', '-')
        ax.annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(facecolor=color, edgecolor=color, width=1.5, headwidth=6, headlength=6, linestyle=ls), zorder=1)
        if arr.get('label'):
            lx, ly = arr.get('label_pos', ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2))
            ax.text(lx, ly, arr['label'], ha='center', va='center', fontsize=8, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=color, lw=1), zorder=4)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# 1. Global Pipeline
# -------------------------------------------------------------
global_steps = [
    {'coords': (3.0, 11.1, 4.0, 0.6), 'text': 'Parse cutoffs, k-points & execution toggles', 'title': 'Phase 0: config.yaml', 'bg': '#f8fafc', 'border': '#64748b'},
    {'coords': (2.2, 9.8, 5.6, 0.8), 'text': 'Divacancy 585-D pore formation + BFGS geometry relaxation (GPAW)', 'title': 'Phase 1: Substrate & Initial Cavity', 'bg': '#eff6ff', 'border': '#3b82f6', 't_col': '#1d4ed8'},
    {'coords': (2.2, 8.4, 5.6, 0.9), 'text': '10 Pt-NxCy motifs (Saturated Pt-N4 to Open Pt-C4)\nSingle Pt atom coordination & embedding', 'title': 'Phase 2: SAC Motif Library', 'bg': '#eff6ff', 'border': '#3b82f6', 't_col': '#1d4ed8'},
    {'coords': (2.0, 6.9, 6.0, 1.0), 'text': '260 Ab Initio single-point configurations:\n• Thermal atomic rattles (0.05 - 0.15 Å)\n• Vertical z(Pt) & z(H2) scans • H-H bond cleavage', 'title': 'Phase 3: Ab Initio Sampling (GPAW PBE-D3)', 'bg': '#fff7ed', 'border': '#ea580c', 't_col': '#c2410c'},
    {'coords': (2.5, 5.5, 5.0, 0.85), 'text': 'Relational SQLite storage: positions, energies, forces\nAutomated Train / Val dataset export (.xyz)', 'title': 'Phase 4: Database Infrastructure (catalysis.db)', 'bg': '#f8fafc', 'border': '#475569', 't_col': '#1e293b'},
    {'coords': (2.0, 4.0, 6.0, 1.0), 'text': 'Supervised fine-tuning of foundation MACE model\nJoint energy and force loss optimization across 10 motifs', 'title': 'Phase 5: MLIP Training (MACE)', 'bg': '#faf5ff', 'border': '#9333ea', 't_col': '#7e22ce'},
    {'coords': (1.8, 2.5, 6.4, 1.05), 'text': 'Climbing-Image NEB: activation barrier Ea & reaction energy ΔE\nhTST harmonic thermochemistry: free energy barrier ΔG‡ (298 K, 1 bar)', 'title': 'Phases 6 & 7: Catalytic CI-NEB Screening', 'bg': '#eff6ff', 'border': '#2563eb', 't_col': '#1d4ed8'},
    {'coords': (1.8, 1.1, 6.4, 0.95), 'text': '20 ps Langevin MD (300 K) with force tripwire (|F| > 4.5 eV/Å)\nThermal stability verification + on-the-fly GPAW fallbacks', 'title': 'Phase 8: Active-Learning Production MD', 'bg': '#fffbeb', 'border': '#d97706', 't_col': '#b45309'},
    {'coords': (2.2, -0.15, 5.6, 0.8), 'text': 'Catalytic volcano plot, error distributions, MSD, z_Pt(t), and g(r)', 'title': 'Phase 9: Post-Processing & Figures', 'bg': '#f0fdf4', 'border': '#16a34a', 't_col': '#15803d'}
]
global_arrows = [
    {'from': (5.0, 11.1), 'to': (5.0, 10.6)},
    {'from': (5.0, 9.8), 'to': (5.0, 9.3)},
    {'from': (5.0, 8.4), 'to': (5.0, 7.9)},
    {'from': (5.0, 6.9), 'to': (5.0, 6.35)},
    {'from': (5.0, 5.5), 'to': (5.0, 5.0)},
    {'from': (5.0, 4.0), 'to': (5.0, 3.55)},
    {'from': (5.0, 2.5), 'to': (5.0, 2.05)},
    {'from': (5.0, 1.1), 'to': (5.0, 0.65)}
]
create_diagram("Autonomous SAC Screening Pipeline: Global Architecture",
               "End-to-end workflow from structural design to catalytic screening & dynamic validation",
               "pipeline_overview_global.png", global_steps, global_arrows)

# -------------------------------------------------------------
# 2. Phase 3: Ab Initio Sampling Grid
# -------------------------------------------------------------
sampling_steps = [
    {'coords': (3.0, 11.0, 4.0, 0.6), 'text': '10 optimized Pt-NxCy motifs at 0 K', 'title': 'Input: Relaxed Geometries', 'bg': '#eff6ff', 'border': '#3b82f6'},
    {'coords': (1.5, 9.2, 7.0, 1.2), 'text': 'Generation of 26 non-equilibrium structures per motif:\n• Gaussian coordinate rattles (σ = 0.05 to 0.15 Å)\n• Rigid vertical approach scans: z(H2) from 1.2 to 3.5 Å\n• H-H bond cleavage stretching: d(H-H) from 0.74 to 2.8 Å', 'title': 'Geometric Perturbation Engine', 'bg': '#f8fafc', 'border': '#64748b'},
    {'coords': (2.0, 7.4, 6.0, 1.2), 'text': 'Single-point DFT calculations across all perturbed frames:\n• PBE functional + Grimme D3 dispersion corrections\n• Consistent k-point sampling & tight SCF criteria (1e-5 eV)\n• Fixed unit cell boundaries without ionic relaxation', 'title': 'Ab Initio Engine (GPAW PBE-D3)', 'bg': '#fff7ed', 'border': '#ea580c', 't_col': '#c2410c'},
    {'coords': (2.0, 5.6, 6.0, 1.2), 'text': 'Systematic extraction of observables:\n• Kohn-Sham total ground-state energy (E_DFT)\n• Analytical Hellmann-Feynman atomic force tensors (3N)\n• Cell hydrostatic stress and total dipole moment', 'title': 'Tensor Parser & Extraction', 'bg': '#f8fafc', 'border': '#475569'},
    {'coords': (2.2, 3.8, 5.6, 1.1), 'text': 'Relational SQLite storage:\n• Indexed records [DB ID: 1] through [DB ID: 260]\n• Automatic Train (80%) / Val (20%) stratified partitioning\n• Export to formatted extended XYZ training datasets', 'title': 'Commit to catalysis.db', 'bg': '#f0fdf4', 'border': '#16a34a', 't_col': '#15803d'}
]
sampling_arrows = [
    {'from': (5.0, 11.0), 'to': (5.0, 10.4)},
    {'from': (5.0, 9.2), 'to': (5.0, 8.6)},
    {'from': (5.0, 7.4), 'to': (5.0, 6.8)},
    {'from': (5.0, 5.6), 'to': (5.0, 4.9)}
]
create_diagram("Phase 3 Architecture: Ab Initio Sampling Grid",
               "Generating 260 benchmark reference points for MACE foundation fine-tuning",
               "phase3_sampling_grid.png", sampling_steps, sampling_arrows)

# -------------------------------------------------------------
# 3. Phase 7: CI-NEB Screening & Thermochemistry
# -------------------------------------------------------------
neb_steps = [
    {'coords': (2.5, 11.0, 5.0, 0.65), 'text': 'Fine-tuned MACE MLIP + 10 Pt-NxCy cavities', 'title': 'Input: Trained Potential Model', 'bg': '#faf5ff', 'border': '#9333ea', 't_col': '#7e22ce'},
    {'coords': (1.8, 9.3, 6.4, 1.1), 'text': '• Initial State (IS): Physisorbed H2 above Pt single atom\n• Final State (FS): Dissociated co-adsorbed H* atoms (Pt-H & C/N-H)\n• Image generation: IDPP interpolation across 7 reaction images', 'title': 'Reaction Pathway Initialization', 'bg': '#eff6ff', 'border': '#3b82f6', 't_col': '#1d4ed8'},
    {'coords': (1.5, 7.3, 7.0, 1.4), 'text': 'Climbing-Image NEB optimization accelerated by MACE:\n• High-throughput evaluation of perpendicular forces & spring tensors\n• Upward climbing of the saddle-point image along the MEP\n• Force convergence criteria: |F_max| < 0.03 eV/Å\n• Direct extraction of activation barrier Ea and reaction energy ΔE', 'title': 'CI-NEB Transition State Resolution', 'bg': '#eff6ff', 'border': '#1d4ed8', 't_col': '#1e40af'},
    {'coords': (1.8, 5.1, 6.4, 1.5), 'text': 'Harmonic vibrational frequency analysis:\n• Finite-difference Hessian construction via MACE potential\n• Zero-point energy (ZPE) corrections\n• Partition functions (translational, rotational, vibrational)\n• Computation of Gibbs free energy barrier ΔG‡ (298.15 K, 1 bar)', 'title': 'Harmonic Transition State Theory (hTST)', 'bg': '#fff7ed', 'border': '#ea580c', 't_col': '#c2410c'},
    {'coords': (2.0, 3.2, 6.0, 1.2), 'text': '• Scaling relation: Catalytic activity vs. hydrogen binding free energy ΔG_H*\n• Pinpoint optimal single-atom cavity at the volcano peak\n• Export tabulated kinetics to results/catalytic_screening.csv', 'title': 'Catalytic Volcano Construction', 'bg': '#f0fdf4', 'border': '#16a34a', 't_col': '#15803d'}
]
neb_arrows = [
    {'from': (5.0, 11.0), 'to': (5.0, 10.4)},
    {'from': (5.0, 9.3), 'to': (5.0, 8.7)},
    {'from': (5.0, 7.3), 'to': (5.0, 6.6)},
    {'from': (5.0, 5.1), 'to': (5.0, 4.4)}
]
create_diagram("Phases 6 & 7 Architecture: CI-NEB Screening & Thermochemistry",
               "H2 dissociation barrier resolution and catalytic volcano mapping",
               "phase7_cineb_screening.png", neb_steps, neb_arrows)

# -------------------------------------------------------------
# 4. Phase 8: Active-Learning Production MD
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 14), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

def draw_box(x, y, w, h, text, title=None, bg='#ffffff', border='#334155', rx=0.1, fontsize=10, title_color='#0f172a', text_color='#334155'):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.08,rounding_size={rx}", facecolor=bg, edgecolor=border, linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    if title:
        ax.text(x + w/2, y + h - 0.35, title, ha='center', va='center', fontsize=fontsize+1, fontweight='bold', color=title_color, zorder=3)
        ax.text(x + w/2, y + (h - 0.4)/2, text, ha='center', va='center', fontsize=fontsize, color=text_color, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=title_color, zorder=3)

# Header
draw_box(0.8, 13.1, 8.4, 0.7, "On-the-fly hybrid MACE / GPAW Born-Oppenheimer coupling (T = 300 K)", 
         title="Phase 8: Active-Learning Production Molecular Dynamics", bg='#f8fafc', border='#cbd5e1', title_color='#0f172a', text_color='#64748b')

# Start
draw_box(3.6, 12.1, 2.8, 0.5, "Time step: t = n × Δt", bg='#0f172a', border='#0f172a', title_color='#ffffff')
ax.annotate('', xy=(5.0, 11.2), xytext=(5.0, 12.1), arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6, headlength=6), zorder=1)

# MACE
draw_box(2.2, 10.3, 5.6, 0.9, "High-throughput evaluation: Energy E_MACE and forces {F_i, MACE}", 
         title="MACE MLIP Inference Engine", bg='#eff6ff', border='#3b82f6', title_color='#1d4ed8', text_color='#1e40af')
ax.annotate('', xy=(5.0, 9.4), xytext=(5.0, 10.3), arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6, headlength=6), zorder=1)

# Diamond Tripwire
diamond = patches.Polygon([[5.0, 9.4], [7.3, 8.6], [5.0, 7.8], [2.7, 8.6]], closed=True, facecolor='#fffbeb', edgecolor='#f59e0b', linewidth=2, zorder=2)
ax.add_patch(diamond)
ax.text(5.0, 8.85, "Uncertainty Tripwire Test", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#92400e', zorder=3)
ax.text(5.0, 8.55, "max_i ||F_i, MACE|| > 4.5 eV/Å ?", ha='center', va='center', fontsize=10, fontweight='bold', color='#b45309', zorder=3)
ax.text(5.0, 8.25, "Confidence domain boundary verification", ha='center', va='center', fontsize=8.5, color='#78350f', zorder=3)

# NO branch (Fast Path)
ax.plot([2.7, 1.2, 1.2, 3.0], [8.6, 8.6, 3.6, 3.6], color='#047857', lw=2, zorder=1)
ax.annotate('', xy=(3.0, 3.6), xytext=(2.9, 3.6), arrowprops=dict(facecolor='#047857', edgecolor='#047857', width=1.5, headwidth=6, headlength=6), zorder=1)
draw_box(1.4, 8.75, 1.1, 0.35, "NO (Stable)", bg='#ecfdf5', border='#10b981', fontsize=8.5, title_color='#047857')
ax.text(0.95, 6.0, "Retain: F_step = F_MACE", ha='center', va='center', rotation=90, fontsize=9, fontweight='bold', color='#065f46')

# YES branch (DFT Fallback)
ax.plot([7.3, 8.7, 8.7], [8.6, 8.6, 7.6], color='#b45309', lw=2, zorder=1)
ax.annotate('', xy=(8.7, 7.6), xytext=(8.7, 7.7), arrowprops=dict(facecolor='#b45309', edgecolor='#b45309', width=1.5, headwidth=6, headlength=6), zorder=1)
draw_box(7.25, 8.75, 1.4, 0.35, "YES (Uncertain)", bg='#fef2f2', border='#ef4444', fontsize=8.5, title_color='#b91c1c')

draw_box(5.8, 6.8, 3.6, 0.75, "Instantaneous structure freeze at step n", 
         title="Active Learning Alert", bg='#fef2f2', border='#ef4444', title_color='#991b1b', text_color='#7f1d1d')
ax.annotate('', xy=(7.6, 5.85), xytext=(7.6, 6.8), arrowprops=dict(facecolor='#b45309', edgecolor='#b45309', width=1.5, headwidth=6, headlength=6), zorder=1)

draw_box(5.8, 4.85, 3.6, 1.0, "Kohn-Sham SCF convergence\nHellmann-Feynman gradients → E_DFT, F_DFT", 
         title="Single-Point GPAW (DFT)", bg='#fff7ed', border='#ea580c', title_color='#9a3412', text_color='#7c2d12')
ax.annotate('', xy=(7.6, 4.0), xytext=(7.6, 4.85), arrowprops=dict(facecolor='#b45309', edgecolor='#b45309', width=1.5, headwidth=6, headlength=6), zorder=1)

draw_box(5.8, 3.25, 3.6, 0.75, "catalysis.db commit (Gen-2 dataset)\nOverride: F_step = F_DFT", 
         title="Database Update & Force Substitution", bg='#f8fafc', border='#64748b', title_color='#1e293b', text_color='#334155')

ax.plot([5.8, 5.4], [3.62, 3.62], color='#b45309', lw=2, zorder=1)
ax.annotate('', xy=(5.4, 3.62), xytext=(5.5, 3.62), arrowprops=dict(facecolor='#b45309', edgecolor='#b45309', width=1.5, headwidth=6, headlength=6), zorder=1)

# Langevin Integrator
draw_box(2.8, 3.1, 2.6, 1.05, "a_i = F_i / m_i  |  Langevin Thermostat (300 K)\nUpdate: r(t + Δt) and v(t + Δt)", 
         title="MD Time Integrator", bg='#f1f5f9', border='#334155', title_color='#0f172a', text_color='#1e293b')

ax.annotate('', xy=(4.1, 2.4), xytext=(4.1, 3.1), arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6, headlength=6), zorder=1)
draw_box(2.7, 1.95, 2.8, 0.45, "Write frame to trajectory.traj", bg='#ffffff', border='#94a3b8', fontsize=9, title_color='#334155')

ax.annotate('', xy=(4.1, 1.45), xytext=(4.1, 1.95), arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=6, headlength=6), zorder=1)
diamond_end = patches.Polygon([[4.1, 1.45], [5.6, 1.05], [4.1, 0.65], [2.6, 1.05]], closed=True, facecolor='#f8fafc', edgecolor='#64748b', linewidth=1.5, zorder=2)
ax.add_patch(diamond_end)
ax.text(4.1, 1.15, "n == N_total ?", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1e293b', zorder=3)
ax.text(4.1, 0.9, "(e.g., 500 or 5000)", ha='center', va='center', fontsize=8, color='#64748b', zorder=3)

# Loopback
ax.plot([2.6, 0.3, 0.3, 3.6], [1.05, 1.05, 12.35, 12.35], color='#64748b', lw=1.5, ls='--', zorder=1)
ax.annotate('', xy=(3.6, 12.35), xytext=(3.5, 12.35), arrowprops=dict(facecolor='#64748b', edgecolor='#64748b', width=1, headwidth=5, headlength=5), zorder=1)
draw_box(0.5, 1.15, 1.4, 0.3, "NO: n = n + 1", bg='#f1f5f9', border='#cbd5e1', fontsize=7.5, title_color='#475569')

# Completion to Phase 9
ax.plot([5.6, 7.6, 7.6], [1.05, 1.05, 1.4], color='#2563eb', lw=2, zorder=1)
ax.annotate('', xy=(7.6, 1.4), xytext=(7.6, 1.3), arrowprops=dict(facecolor='#2563eb', edgecolor='#2563eb', width=1.5, headwidth=6, headlength=6), zorder=1)
draw_box(5.8, 1.15, 0.8, 0.3, "YES", bg='#eff6ff', border='#93c5fd', fontsize=8, title_color='#1d4ed8')

draw_box(6.0, 1.4, 3.4, 1.05, "• Mean Squared Displacement: MSD(t)\n• Metal anchoring stability: z_Pt(t)\n• Radial distribution function: g(r)", 
         title="Phase 9: Post-Processing", bg='#f0fdf4', border='#16a34a', title_color='#15803d', text_color='#166534', fontsize=8.5)

plt.tight_layout()
plt.savefig('phase8_active_learning_md.png', dpi=300, bbox_inches='tight')
plt.close()

print("All 4 English diagrams generated successfully at 300 DPI.")
