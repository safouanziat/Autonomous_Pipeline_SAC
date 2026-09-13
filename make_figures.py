# make_figures.py
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read, Trajectory
from mace.calculators import MACECalculator, mace_mp

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = cfg.get("mace_mlip", {}).get("device", "cpu")
model_path = "mace_sac_model.model"

print("1. Evaluating validation set parity...")
val_atoms = read("mace_val.xyz", index=":")
calc_ft = MACECalculator(model_paths=model_path, device=device)
calc_zs = mace_mp(model="small", device=device)

dft_e, ft_e, zs_e = [], [], []
dft_f, ft_f = [], []

for atoms in val_atoms:
    dft_e.append(atoms.get_potential_energy())
    dft_f.append(atoms.get_forces())
    
    atoms.calc = calc_ft
    ft_e.append(atoms.get_potential_energy())
    ft_f.append(atoms.get_forces())
    
    atoms.calc = calc_zs
    zs_e.append(atoms.get_potential_energy())

dft_e = np.array(dft_e)
ft_e = np.array(ft_e)
zs_e = np.array(zs_e)
dft_f = np.concatenate(dft_f).flatten()
ft_f = np.concatenate(ft_f).flatten()

mae_ft_e = np.mean(np.abs(ft_e - dft_e))
mae_zs_e = np.mean(np.abs(zs_e - dft_e))
mae_ft_f = np.mean(np.abs(ft_f - dft_f))

print(f"   -> FT Energy MAE: {mae_ft_e*1000:.1f} meV | Zero-Shot MAE: {mae_zs_e*1000:.1f} meV")
print(f"   -> FT Forces MAE: {mae_ft_f*1000:.1f} meV/Å")

# ----------------- PARITY PLOT -----------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(dft_e, zs_e, color="#d62728", marker="^", s=45, alpha=0.5, label=f"Zero-Shot MACE-MP-0 (MAE={mae_zs_e*1000:.1f} meV)")
axes[0].scatter(dft_e, ft_e, color="#1f77b4", s=45, alpha=0.9, label=f"Fine-Tuned MACE (MAE={mae_ft_e*1000:.1f} meV)")
lims_e = [min(dft_e) - 0.5, max(dft_e) + 0.5]
axes[0].plot(lims_e, lims_e, "k--", alpha=0.7)
axes[0].set_xlim(lims_e)
axes[0].set_ylim(lims_e)
axes[0].set_xlabel("DFT Energy (eV)", fontsize=12)
axes[0].set_ylabel("Predicted Energy (eV)", fontsize=12)
axes[0].set_title("Energy Parity (Held-Out Test Set)", fontsize=13, fontweight="bold")
axes[0].legend(frameon=True)

axes[1].scatter(dft_f, ft_f, color="#2ca02c", s=15, alpha=0.4, label=f"Fine-Tuned MACE (MAE={mae_ft_f*1000:.1f} meV/Å)")
lims_f = [min(dft_f), max(dft_f)]
axes[1].plot(lims_f, lims_f, "k--", alpha=0.7)
axes[1].set_xlim(lims_f)
axes[1].set_ylim(lims_f)
axes[1].set_xlabel("DFT Atomic Forces (eV/Å)", fontsize=12)
axes[1].set_ylabel("Predicted Forces (eV/Å)", fontsize=12)
axes[1].set_title("Forces Parity", fontsize=13, fontweight="bold")
axes[1].legend(frameon=True)

plt.tight_layout()
plt.savefig("figures/parity_plot.png", dpi=300)
plt.close()
print("   -> figures/parity_plot.png written.")

# ----------------- REACTION PROFILES -----------------
import glob

neb_files = sorted(glob.glob("neb_*.traj"))
print(f"2. Plotting reaction profiles from {len(neb_files)} CI-NEB trajectories...")

fig, ax = plt.subplots(figsize=(10, 6))
for traj_path in neb_files:
    motif = traj_path.replace("neb_", "").replace(".traj", "")
    traj = Trajectory(traj_path)
    # The last recorded images in the trajectory file are the converged band
    n_images = cfg.get("kinetics_and_thermo", {}).get("neb_images", 3) + 2
    converged_band = [traj[i] for i in range(-n_images, 0)]
    energies = np.array([img.get_potential_energy() for img in converged_band])
    energies -= energies[0]
    e_act = max(energies)
    rxn_coord = np.linspace(0, 1, len(energies))
    ax.plot(rxn_coord, energies, marker="o", label=f"{motif} (Ea={e_act:.2f} eV)")

ax.set_xlabel("Reaction Coordinate (Normalized)", fontsize=12)
ax.set_ylabel("Relative Energy (eV)", fontsize=12)
ax.set_title(f"Catalytic {cfg['adsorbate']['type']} Activation Across Coordination Motifs", fontsize=13, fontweight="bold")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)
plt.tight_layout()
plt.savefig("figures/reaction_profiles.png", dpi=300)
plt.close()
print("   -> figures/reaction_profiles.png written.")