import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_diagram(title, subtitle, filename, steps_data, arrows_data, side_notes=None):
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

    # Draw steps
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

# 1. Pipeline Global
global_steps = [
    {'coords': (3.0, 11.1, 4.0, 0.6), 'text': 'Lecture des cutoffs, k-points & toggles', 'title': 'Phase 0 : config.yaml', 'bg': '#f8fafc', 'border': '#64748b'},
    {'coords': (2.2, 9.8, 5.6, 0.8), 'text': 'Pore divacancy 585-D + Relaxation BFGS (GPAW)', 'title': 'Phase 1 : Substrat & Cavité initiale', 'bg': '#eff6ff', 'border': '#3b82f6', 't_col': '#1d4ed8'},
    {'coords': (2.2, 8.4, 5.6, 0.9), 'text': '10 motifs Pt-NxCy (Saturés Pt-N4 à ouverts Pt-C4)\nInsertion du single atom de Pt', 'title': 'Phase 2 : Bibliothèque des 10 Motifs SAC', 'bg': '#eff6ff', 'border': '#3b82f6', 't_col': '#1d4ed8'},
    {'coords': (2.0, 6.9, 6.0, 1.0), 'text': '260 configurations DFT ponctuelles :\n• Rattles thermiques (0.05-0.15 Å)\n• Approches z(Pt) & z(H2) • Clivage H-H', 'title': 'Phase 3 : Échantillonnage Ab Initio (GPAW PBE-D3)', 'bg': '#fff7ed', 'border': '#ea580c', 't_col': '#c2410c'},
    {'coords': (2.5, 5.5, 5.0, 0.85), 'text': 'Stockage SQLite atomic positions, energies, forces\nExport train / val splits (.xyz)', 'title': 'Phase 4 : Base de Données (catalysis.db)', 'bg': '#f8fafc', 'border': '#475569', 't_col': '#1e293b'},
    {'coords': (2.0, 4.0, 6.0, 1.0), 'text': 'Fine-tuning du modèle de fondation MACE\nFitting supervisé E & F sur les 10 cavités', 'title': 'Phase 5 : Entraînement MLIP (MACE)', 'bg': '#faf5ff', 'border': '#9333ea', 't_col': '#7e22ce'},
    {'coords': (1.8, 2.5, 6.4, 1.05), 'text': 'Extraction barrière Ea & ΔE (Climbing-Image NEB)\nThermochimie harmonique hTST : ΔG*(298K, 1 bar)', 'title': 'Phase 6 & 7 : Criblage Catalytique & Profils NEB', 'bg': '#eff6ff', 'border': '#2563eb', 't_col': '#1d4ed8'},
    {'coords': (1.8, 1.1, 6.4, 0.95), 'text': '20 ps Langevin (300 K) + Tripwire (|F| > 4.5 eV/Å)\nValidation tenue mécanique & fallbacks GPAW', 'title': 'Phase 8 : Active-Learning Production MD', 'bg': '#fffbeb', 'border': '#d97706', 't_col': '#b45309'},
    {'coords': (2.2, -0.15, 5.6, 0.8), 'text': 'Courbe de volcan catalytique, barres d\'erreur, MSD, z_Pt(t), g(r)', 'title': 'Phase 9 : Post-Traitement & Publication', 'bg': '#f0fdf4', 'border': '#16a34a', 't_col': '#15803d'}
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
create_diagram("Architecture Globale du Pipeline Autonome SAC", "De la génération de substrat au criblage catalytique et à la validation dynamique", "pipeline_overview_global.png", global_steps, global_arrows)

# 2. Phase 3
sampling_steps = [
    {'coords': (3.0, 11.0, 4.0, 0.6), 'text': '10 structures Pt-NxCy relaxées à 0 K', 'title': 'Entrée : Motifs Optimisés', 'bg': '#eff6ff', 'border': '#3b82f6'},
    {'coords': (1.5, 9.2, 7.0, 1.2), 'text': 'Génération de 26 configurations non-équilibrées par motif :\n• Rattles gaussiens (σ = 0.05 à 0.15 Å)\n• Approches H2 selon z (1.2 Å à 3.5 Å)\n• Élongations d(H-H) de 0.74 Å à 2.8 Å (clivage)', 'title': 'Générateur de Perturbations Géométriques', 'bg': '#f8fafc', 'border': '#64748b'},
    {'coords': (2.0, 7.4, 6.0, 1.2), 'text': 'Calcul Single-Point DFT sur chaque structure :\n• Fonctionnelle PBE + corrections de dispersion D3\n• Mode LCAO / Grid selon config.yaml\n• Convergence SCF stricte (1e-5 eV)', 'title': 'Moteur Ab Initio GPAW (260 Calculs)', 'bg': '#fff7ed', 'border': '#ea580c', 't_col': '#c2410c'},
    {'coords': (2.0, 5.6, 6.0, 1.2), 'text': 'Extraction systématique des tenseurs :\n• Énergie totale DFT (E_tot)\n• Tenseur de forces atomiques 3N (F_x, F_y, F_z)\n• Stress hydrostatique & dipôle', 'title': 'Parseur de Tenseurs Hellmann-Feynman', 'bg': '#f8fafc', 'border': '#475569'},
    {'coords': (2.2, 3.8, 5.6, 1.1), 'text': 'Stockage relationnel SQLite :\n• Indexation [DB ID: 1] à [DB ID: 260]\n• Séparation automatique Train (80%) / Val (20%)\n• Export vers training_dataset.xyz', 'title': 'Insertion dans catalysis.db', 'bg': '#f0fdf4', 'border': '#16a34a', 't_col': '#15803d'}
]
sampling_arrows = [
    {'from': (5.0, 11.0), 'to': (5.0, 10.4)},
    {'from': (5.0, 9.2), 'to': (5.0, 8.6)},
    {'from': (5.0, 7.4), 'to': (5.0, 6.8)},
    {'from': (5.0, 5.6), 'to': (5.0, 4.9)}
]
create_diagram("Détail Phase 3 : Échantillonnage Ab Initio (Sampling Grid)", "Génération des 260 points de référence pour l'entraînement du potentiel MACE", "phase3_sampling_grid.png", sampling_steps, sampling_arrows)

# 3. Phase 7
neb_steps = [
    {'coords': (2.5, 11.0, 5.0, 0.65), 'text': 'Potentiel fine-tuné MACE + 10 cavités Pt-NxCy', 'title': 'Entrée : Modèle Entraîné', 'bg': '#faf5ff', 'border': '#9333ea', 't_col': '#7e22ce'},
    {'coords': (1.8, 9.3, 6.4, 1.1), 'text': '• État Initial (IS) : H2 physisorbé au-dessus du Pt\n• État Final (FS) : 2 H dissociés chimisorbés (Pt-H & C/N-H)\n• Interpolation linéaire / IDPP de 7 images intermédiaires', 'title': 'Construction du Chemin Réactionnel Initial', 'bg': '#eff6ff', 'border': '#3b82f6', 't_col': '#1d4ed8'},
    {'coords': (1.5, 7.3, 7.0, 1.4), 'text': 'Optimisation par Climbing-Image NEB pilotée par MACE :\n• Calcul ultra-rapide des forces orthogonales & ressorts\n• Ascension de l\'image centrale au col (Saddle Point)\n• Tolérance de convergence : |F_max| < 0.03 eV/Å\n• Extraction barrière Ea et enthalpie de réaction ΔE', 'title': 'Algorithme CI-NEB (Recherche du Saddle Point)', 'bg': '#eff6ff', 'border': '#1d4ed8', 't_col': '#1e40af'},
    {'coords': (1.8, 5.1, 6.4, 1.5), 'text': 'Analyse des fréquences vibrationnelles harmoniques :\n• Matrice hessienne par différences finies via MACE\n• Énergie de point zéro (ZPE)\n• Fonctions de partition translation, rotation, vibration\n• Calcul de l\'énergie libre d\'activation ΔG‡ (298.15 K, 1 bar)', 'title': 'Thermochimie Harmonique (hTST)', 'bg': '#fff7ed', 'border': '#ea580c', 't_col': '#c2410c'},
    {'coords': (2.0, 3.2, 6.0, 1.2), 'text': '• Traçage de l\'activité en fonction du descripteur ΔG_H*\n• Identification du motif optimal au sommet du volcan\n• Export des tables récapitulatives dans results/neb_data.csv', 'title': 'Diagramme de Volcan Catalytique', 'bg': '#f0fdf4', 'border': '#16a34a', 't_col': '#15803d'}
]
neb_arrows = [
    {'from': (5.0, 11.0), 'to': (5.0, 10.4)},
    {'from': (5.0, 9.3), 'to': (5.0, 8.7)},
    {'from': (5.0, 7.3), 'to': (5.0, 6.6)},
    {'from': (5.0, 5.1), 'to': (5.0, 4.4)}
]
create_diagram("Détail Phases 6 & 7 : Criblage CI-NEB & Thermochimie", "Calcul des barrières de dissociation H2 et détermination du profil volcanique", "phase7_cineb_screening.png", neb_steps, neb_arrows)

print("Images générées avec succès dans le répertoire courant.")
