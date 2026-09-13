import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_diagram(title, subtitle, filename, steps_data, arrows_data):
    fig, ax = plt.subplots(figsize=(10, 13), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')
    
    rect_head = patches.FancyBboxPatch((0.8, 12.1), 8.4, 0.65, boxstyle="round,pad=0.06,rounding_size=0.08",
                                       facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.5, zorder=2)
    ax.add_patch(rect_head)
    ax.text(5.0, 12.52, title, ha='center', va='center', fontsize=12, fontweight='bold', color='#0f172a', zorder=3)
    ax.text(5.0, 12.28, subtitle, ha='center', va='center', fontsize=9.5, color='#64748b', zorder=3)

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

    for arr in arrows_data:
        p1, p2 = arr['from'], arr['to']
        color = arr.get('color', '#475569')
        ax.annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(facecolor=color, edgecolor=color, width=1.5, headwidth=6, headlength=6), zorder=1)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# 1. Phase 1 & 2
phase12_steps = [
    {'coords': (2.8, 11.0, 4.4, 0.65), 'text': 'Pristine 4x4 Graphene Supercell\nVacuum gap = 15 Å (z-axis)', 'title': 'Input: Graphene Lattice', 'bg': '#f8fafc', 'border': '#64748b'},
    {'coords': (1.8, 9.2, 6.4, 1.2), 'text': '• Removal of 2 adjacent carbon atoms (Divacancy)\n• Stone-Wales local reconstruction: 5-8-5 pore formation\n• Initial unrelaxed pore: C2vac substrate', 'title': 'Cavity Creation: 585-D Divacancy', 'bg': '#eff6ff', 'border': '#3b82f6', 't_col': '#1d4ed8'},
    {'coords': (1.8, 7.3, 6.4, 1.25), 'text': 'GPAW DFT electronic and ionic minimization:\n• PBE functional + Grimme D3 dispersion\n• BFGS quasi-Newton line search (|F_max| < 0.02 eV/Å)\n• Out-of-plane buckling and edge-stress relaxation', 'title': 'DFT Geometry Relaxation (585-D)', 'bg': '#fff7ed', 'border': '#ea580c', 't_col': '#c2410c'},
    {'coords': (1.5, 5.2, 7.0, 1.4), 'text': 'Combinatorial substitution of edge carbon atoms by nitrogen:\n• Systematic N-doping (x = 0 to 4 nitrogen atoms)\n• Coordination motifs: Pt-N4, Pt-N3C1, Pt-N2C2 (cis/trans),\n  Pt-N1C3, and un-doped Pt-C4 (10 distinct configurations)', 'title': 'Combinatorial Edge Heteroatom Doping', 'bg': '#faf5ff', 'border': '#9333ea', 't_col': '#7e22ce'},
    {'coords': (1.8, 3.1, 6.4, 1.4), 'text': 'Single platinum atom insertion into coordinating center:\n• Positioned at cavity centroid (initial z ~ 1.2 Å)\n• DFT spin-polarized minimization of Pt-NxCy complexes\n• Trapping energy calculation: E_trap = E_tot - (E_sub + E_Pt)', 'title': 'Pt Single-Atom Insertion & Anchoring', 'bg': '#f0fdf4', 'border': '#16a34a', 't_col': '#15803d'},
    {'coords': (2.2, 1.6, 5.6, 0.85), 'text': 'Export optimized motifs to structures/motifs/\nIndexed records registered for Phase 3 Ab Initio sampling', 'title': 'SAC Motif Library Committed', 'bg': '#f8fafc', 'border': '#475569', 't_col': '#1e293b'}
]
phase12_arrows = [
    {'from': (5.0, 11.0), 'to': (5.0, 10.4)}, {'from': (5.0, 9.2), 'to': (5.0, 8.55)},
    {'from': (5.0, 7.3), 'to': (5.0, 6.6)}, {'from': (5.0, 5.2), 'to': (5.0, 4.5)},
    {'from': (5.0, 3.1), 'to': (5.0, 2.45)}
]
create_diagram("Phases 1 & 2 Architecture: Substrate & SAC Motif Library",
               "Generation of 585-D divacancy pore and combinatorial Pt-NxCy cavities",
               "phase1_2_substrate_motifs_en.png", phase12_steps, phase12_arrows)

# 2. Phase 4 & 5
phase45_steps = [
    {'coords': (2.5, 11.0, 5.0, 0.65), 'text': '260 Ab Initio single-point configurations (GPAW PBE-D3)', 'title': 'Input: Phase 3 Sampling Grid', 'bg': '#fff7ed', 'border': '#ea580c', 't_col': '#c2410c'},
    {'coords': (1.8, 9.3, 6.4, 1.1), 'text': '• SQLite schema with ASE Atoms storage (atomic numbers, cells, pbc)\n• Custom key-value pairs: e_tot, f_norm_max, motif_id, perturbation_type\n• Integrity verification & automated backup', 'title': 'Database Schema & Storage (catalysis.db)', 'bg': '#f8fafc', 'border': '#475569', 't_col': '#1e293b'},
    {'coords': (1.8, 7.4, 6.4, 1.25), 'text': 'Stratified partitioning preserving motif balance:\n• Training subset (80% / 208 structures) -> train.xyz\n• Validation subset (20% / 52 structures) -> val.xyz\n• ExtXYZ formatting with virial stress and per-atom forces', 'title': 'Dataset Partitioning & Formatting', 'bg': '#eff6ff', 'border': '#3b82f6', 't_col': '#1d4ed8'},
    {'coords': (1.5, 5.2, 7.0, 1.5), 'text': 'Higher-order equivariant message passing neural network:\n• MACE foundation model transfer learning / fine-tuning\n• Dual loss function: L = w_E * MSE(E) + w_F * MSE(F)\n• Radial cutoff rc = 5.0 Å, spherical harmonics L_max = 2\n• Optimizer: AMSGrad / Adam with learning rate scheduling', 'title': 'MACE Supervised Fine-Tuning', 'bg': '#faf5ff', 'border': '#9333ea', 't_col': '#7e22ce'},
    {'coords': (1.8, 3.0, 6.4, 1.5), 'text': 'Model generalization assessment on unseen validation set:\n• Energy RMSE target: < 3 meV/atom\n• Force RMSE target: < 50 meV/Å\n• Parity scatter plots (DFT vs MACE) generation\n• Export compiled TorchScript / ASE Calculator: mace_sac.model', 'title': 'Validation & Model Compilation', 'bg': '#f0fdf4', 'border': '#16a34a', 't_col': '#15803d'}
]
phase45_arrows = [
    {'from': (5.0, 11.0), 'to': (5.0, 10.4)}, {'from': (5.0, 9.3), 'to': (5.0, 8.65)},
    {'from': (5.0, 7.4), 'to': (5.0, 6.7)}, {'from': (5.0, 5.2), 'to': (5.0, 4.5)}
]
create_diagram("Phases 4 & 5 Architecture: Database & MACE Training",
               "Data curation into catalysis.db and supervised equivariant MLIP fine-tuning",
               "phase4_5_database_mace_training_en.png", phase45_steps, phase45_arrows)

# 3. Phase 9
phase9_steps = [
    {'coords': (2.2, 11.0, 5.6, 0.65), 'text': '• neb_data.csv (CI-NEB kinetics)  • trajectory.traj (MD frames)\n• catalysis.db (DFT reference) • mace_sac.model (Parity logs)', 'title': 'Input: Complete Pipeline Data Artifacts', 'bg': '#f8fafc', 'border': '#475569'},
    {'coords': (1.8, 9.2, 6.4, 1.2), 'text': '• Regression parity curves: DFT vs MACE Energy & Forces\n• Residual distribution histograms\n• Force component error decomposition (Fx, Fy, Fz)', 'title': 'Panel A: MLIP Quality & Parity Metrics', 'bg': '#faf5ff', 'border': '#9333ea', 't_col': '#7e22ce'},
    {'coords': (1.8, 7.3, 6.4, 1.25), 'text': '• Minimum Energy Paths (MEP) along the 7 reaction images\n• Direct comparison of Ea and ΔE across the 10 Pt-NxCy motifs\n• Gibbs free energy profiles ΔG(reaction coordinate) at 298.15 K', 'title': 'Panel B: Reaction Profiles & Barrier Analysis', 'bg': '#eff6ff', 'border': '#2563eb', 't_col': '#1d4ed8'},
    {'coords': (1.5, 5.3, 7.0, 1.35), 'text': '• Sabatier volcano plot: Log(turnover frequency / rate) vs ΔG_H*\n• Microkinetic model overlay with experimental Pt(111) benchmarks\n• Cavity coordination trend ranking: optimal peak motif identification', 'title': 'Panel C: Catalytic Volcano Relationship', 'bg': '#fff7ed', 'border': '#ea580c', 't_col': '#c2410c'},
    {'coords': (1.5, 3.2, 7.0, 1.45), 'text': '• Pt vertical anchor fluctuation z_Pt(t) (demetalation tracking)\n• Mean Squared Displacement: MSD(t) extraction of diffusion regimes\n• Radial distribution functions g(r) for Pt-N, Pt-C, and C-C bond pairs\n• Statistical fallback count & trajectory temperature preservation', 'title': 'Panel D: 300 K Dynamic Stability Analysis', 'bg': '#f0fdf4', 'border': '#16a34a', 't_col': '#15803d'}
]
phase9_arrows = [
    {'from': (5.0, 11.0), 'to': (5.0, 10.4)}, {'from': (5.0, 9.2), 'to': (5.0, 8.55)},
    {'from': (5.0, 7.3), 'to': (5.0, 6.65)}, {'from': (5.0, 5.3), 'to': (5.0, 4.65)}
]
create_diagram("Phase 9 Architecture: Post-Processing & Scientific Figures",
               "Automated multi-panel figure generation and publication-ready analytics",
               "phase9_postprocessing_figures_en.png", phase9_steps, phase9_arrows)

print("Phases 1-2, 4-5, and 9 successfully rendered.")
