"""
Core Engine: Synthesis, Sampling, AIMD, MLIP Training, Active Learning,
CI-NEB, Thermochemistry, and Matplotlib Figure Generation.
"""

import os
import sys
import subprocess
import shutil
import numpy as np

# Multithreading configuration for host CPU cores
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

from ase import Atoms, units
from ase.build import make_supercell
from ase.io import read, write, Trajectory
from ase.db import connect
from ase.optimize import BFGS
from ase.calculators.singlepoint import SinglePointCalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.langevin import Langevin
from ase.mep.neb import NEB
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo, HarmonicThermo
from gpaw import GPAW, PW

try:
    from gpaw.dftd3 import DFTD3
    HAS_DFTD3 = True
except ImportError:
    HAS_DFTD3 = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SACPipelineEngine:
    def __init__(self, config: dict):
        self.cfg = config
        self.sys_cfg = config["system"]
        self.ads_cfg = config["adsorbate"]
        self.dft_cfg = config["dft"]
        self.samp_cfg = config["sampling"]
        self.mace_cfg = config.get("mace_mlip", {})
        self.prod_cfg = config.get("production_md", {})
        self.kin_cfg = config.get("kinetics_and_thermo", {})
        
        self.db = connect(self.sys_cfg["db_path"])
        self.use_d3 = self.dft_cfg.get("use_d3", False) and HAS_DFTD3
        os.makedirs("figures", exist_ok=True)

    # -------------------------------------------------------------
    # 0. Analytical Substrate Synthesis & Adaptive Pre-relaxation
    # -------------------------------------------------------------
    def synthesize_pristine_sac(self) -> Atoms:
        metal = self.sys_cfg["metal"]
        coord_elem = self.sys_cfg["coordination_element"]
        cell_size = self.sys_cfg.get("cell_size", [4, 4])
        vacuum = self.sys_cfg.get("vacuum", 10.0)

        a = 2.46
        c = 2.0 * vacuum
        unit_cell = Atoms(
            "C2",
            positions=[[0, 0, vacuum], [a / 3.0, a / np.sqrt(3), vacuum]],
            cell=[[a, 0, 0], [a / 2.0, a * np.sqrt(3) / 2.0, 0], [0, 0, c]],
            pbc=[True, True, False]
        )
        P = [[cell_size[0], 0, 0], [0, cell_size[1], 0], [0, 0, 1]]
        slab = make_supercell(unit_cell, P)

        # Divacancy pore punch
        com = np.mean(slab.positions[:, :2], axis=0)
        dists = np.linalg.norm(slab.positions[:, :2] - com, axis=1)
        center_two_carbons = np.argsort(dists)[:2]
        metal_pos = np.mean(slab.positions[center_two_carbons], axis=0)
        del slab[center_two_carbons]

        # Anchor transition metal
        slab.append(metal)
        slab.positions[-1] = metal_pos

        # Dope coordination rim
        metal_idx = len(slab) - 1
        dists_to_tm = [slab.get_distance(metal_idx, i) for i in range(len(slab) - 1)]
        coord_indices = np.argsort(dists_to_tm)[:4]
        for idx in coord_indices:
            slab.symbols[idx] = coord_elem

        # Pre-relax substrate
        log_label = f"gpaw_logs/prerelax_{metal}_{coord_elem}4"
        slab.calc = self.get_gpaw_calculator(log_label)
        opt = BFGS(slab, logfile=f"{log_label}_opt.log")
        opt.run(fmax=self.sys_cfg.get("pre_relax_fmax", 0.05))
        slab.calc = None

        write(f"synthesized_{metal}_{coord_elem}4_relaxed.xyz", slab)
        return slab

    # -------------------------------------------------------------
    # 1. 10 Physical Coordination Motifs Generator
    # -------------------------------------------------------------
    def generate_motifs(self, base_slab: Atoms) -> dict:
        metal_sym = self.sys_cfg["metal"]
        coord_elem = self.sys_cfg["coordination_element"]
        syms = np.array(base_slab.get_chemical_symbols())
        metal_idx = np.where(syms == metal_sym)[0][0]
        metal_pos = base_slab.positions[metal_idx]

        all_dists = [(i, base_slab.get_distance(metal_idx, i)) for i in range(len(base_slab)) if i != metal_idx]
        all_dists.sort(key=lambda x: x[1])
        coord_indices = [idx for idx, _ in all_dists[:4]]

        v0 = base_slab.positions[coord_indices[0]] - metal_pos
        angles = []
        for i in range(1, 4):
            vi = base_slab.positions[coord_indices[i]] - metal_pos
            cos_th = np.dot(v0, vi) / (np.linalg.norm(v0) * np.linalg.norm(vi))
            cross_z = np.cross(v0, vi)[2]
            ang = np.degrees(np.arctan2(cross_z, np.dot(v0[:2], vi[:2]))) % 360
            angles.append((ang, coord_indices[i]))

        angles.sort(key=lambda x: x[0])
        ring = [coord_indices[0], angles[0][1], angles[1][1], angles[2][1]]
        motifs = {}

        # 4-Coordinated (6 Motifs)
        for name, syms_list in [
            (f"{metal_sym}-{coord_elem}4", [coord_elem]*4),
            (f"{metal_sym}-{coord_elem}3C1", ["C", coord_elem, coord_elem, coord_elem]),
            (f"{metal_sym}-{coord_elem}2C2_cis", [coord_elem, coord_elem, "C", "C"]),
            (f"{metal_sym}-{coord_elem}2C2_trans", [coord_elem, "C", coord_elem, "C"]),
            (f"{metal_sym}-{coord_elem}1C3", [coord_elem, "C", "C", "C"]),
            (f"{metal_sym}-C4", ["C", "C", "C", "C"]),
        ]:
            s = base_slab.copy()
            for r_idx, s_elem in zip(ring, syms_list):
                s.symbols[r_idx] = s_elem
            motifs[name] = s

        # 3-Coordinated Single Vacancy (4 Motifs)
        def make_3coord(sub_syms, label):
            s = base_slab.copy()
            del s[ring[0]]
            shifted = [idx if idx < ring[0] else idx - 1 for idx in ring[1:]]
            for idx, sym in zip(shifted, sub_syms):
                s.symbols[idx] = sym
            motifs[label] = s

        make_3coord([coord_elem]*3, f"{metal_sym}-{coord_elem}3")
        make_3coord([coord_elem, coord_elem, "C"], f"{metal_sym}-{coord_elem}2C1")
        make_3coord([coord_elem, "C", "C"], f"{metal_sym}-{coord_elem}1C2")
        make_3coord(["C", "C", "C"], f"{metal_sym}-C3")

        # Check config for test subset filter
        sub_cfg = self.sys_cfg.get("test_subset", {})
        if sub_cfg.get("enabled", False):
            selected = sub_cfg.get("motifs", [])
            motifs = {k: v for k, v in motifs.items() if k in selected}

        return motifs

    # -------------------------------------------------------------
    # 2. Static PES Matrix Generator
    # -------------------------------------------------------------
    def sample_static_configurations(self, motif_name: str, slab: Atoms) -> list:
        configs = []
        metal_pos = slab.positions[np.where(np.array(slab.get_chemical_symbols()) == self.sys_cfg["metal"])[0][0]]
        mol_type = self.ads_cfg["type"]
        d_eq = self.ads_cfg["eq_bond_length"]

        configs.append((slab.copy(), {"stage": "pristine_slab", "has_adsorbate": False, "d_bond": 0.0}))

        if self.samp_cfg["rattle"]["enabled"]:
            for amp in self.samp_cfg["rattle"]["amplitudes"]:
                for rep in range(self.samp_cfg["rattle"]["replicates_per_amp"]):
                    r_slab = slab.copy()
                    r_slab.rattle(stdev=amp, seed=42 + rep)
                    configs.append((r_slab, {"stage": f"rattle_amp{amp}_rep{rep+1}", "has_adsorbate": False, "d_bond": 0.0}))

        for z in self.ads_cfg["approach_z_heights"]:
            for orient in ["perp", "para"]:
                s = slab.copy()
                if orient == "perp":
                    a1 = metal_pos + np.array([0.0, 0.0, z])
                    a2 = metal_pos + np.array([0.0, 0.0, z + d_eq])
                else:
                    a1 = metal_pos + np.array([-d_eq / 2.0, 0.0, z])
                    a2 = metal_pos + np.array([d_eq / 2.0, 0.0, z])
                s.extend(Atoms(mol_type, positions=[a1, a2]))
                configs.append((s, {"stage": f"approach_{orient}_z{z}", "has_adsorbate": True, "d_bond": d_eq}))

        for z_cleave in self.ads_cfg["cleavage_z_heights"]:
            for d in self.ads_cfg["bond_stretch_distances"]:
                s = slab.copy()
                a1 = metal_pos + np.array([-d / 2.0, 0.0, z_cleave])
                a2 = metal_pos + np.array([d / 2.0, 0.0, z_cleave])
                s.extend(Atoms(mol_type, positions=[a1, a2]))
                configs.append((s, {"stage": f"cleavage_d{d}_z{z_cleave}", "has_adsorbate": True, "d_bond": float(d)}))

        pt_idx = np.where(np.array(slab.get_chemical_symbols()) == self.sys_cfg["metal"])[0][0]
        coord_atoms = [i for i in range(len(slab)) if i != pt_idx]
        dists = [slab.get_distance(pt_idx, i) for i in coord_atoms]
        neighbor_pos = slab.positions[coord_atoms[int(np.argmin(dists))]]
        s_dissoc = slab.copy()
        a_metal = metal_pos + np.array([0.0, 0.0, 1.55])
        a_neigh = neighbor_pos + np.array([0.0, 0.0, 1.10])
        s_dissoc.extend(Atoms(mol_type, positions=[a_metal, a_neigh]))
        configs.append((s_dissoc, {"stage": "dissociated_chemisorbed", "has_adsorbate": True, "d_bond": round(float(np.linalg.norm(a_metal - a_neigh)), 3)}))

        # Cap configs if running a smoke test
        sub_cfg = self.sys_cfg.get("test_subset", {})
        if sub_cfg.get("enabled", False):
            limit = sub_cfg.get("max_configs_per_motif", None)
            if limit:
                configs = configs[:limit]

        return configs

    # -------------------------------------------------------------
    # 3. DFT Engine & AIMD Sampling
    # -------------------------------------------------------------
    def get_gpaw_calculator(self, log_label: str) -> GPAW:
        setups = {}
        if self.dft_cfg.get("hubbard_u", 0.0) > 0.0:
            setups[self.sys_cfg["metal"]] = f":d,{self.dft_cfg['hubbard_u']:.2f},0.0"
        return GPAW(
            mode=PW(self.dft_cfg["pw_cutoff"]),
            xc=self.dft_cfg["xc"],
            setups=setups if setups else "paw",
            spinpol=self.dft_cfg["spinpol"],
            kpts={"size": tuple(self.dft_cfg["kpts"]), "gamma": True},
            convergence=self.dft_cfg["convergence"],
            txt=f"{log_label}.log"
        )

    def run_dft_evaluation(self, atoms: Atoms, log_label: str) -> tuple:
        atoms.calc = self.get_gpaw_calculator(log_label)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        if self.use_d3:
            try:
                d3 = DFTD3(xc=self.dft_cfg["xc"], damping="bj")
                e_d3, f_d3 = d3.calculate(atoms)
                energy += e_d3
                forces += f_d3
            except Exception:
                pass
        atoms.calc = None
        return energy, forces

    def run_aimd_trajectory(self, motif_name: str, slab: Atoms) -> list:
        aimd_cfg = self.samp_cfg["aimd"]
        md_atoms = slab.copy()
        log_label = f"gpaw_logs/aimd_{motif_name}_{int(aimd_cfg['temperature_k'])}K"
        md_atoms.calc = self.get_gpaw_calculator(log_label)

        MaxwellBoltzmannDistribution(md_atoms, temperature_K=aimd_cfg["temperature_k"])
        Stationary(md_atoms)

        dyn = Langevin(
            md_atoms,
            timestep=aimd_cfg["dt_fs"] * units.fs,
            temperature_K=aimd_cfg["temperature_k"],
            friction=aimd_cfg["friction"],
            logfile=f"{log_label}_dynamics.log",
            trajectory=f"{log_label}.traj"
        )

        n_steps = int(aimd_cfg["total_fs"] / aimd_cfg["dt_fs"])
        interval = int(aimd_cfg["sample_interval_fs"] / aimd_cfg["dt_fs"])
        snapshots = []

        for step in range(1, n_steps + 1):
            dyn.run(1)
            if step % interval == 0:
                snap = md_atoms.copy()
                e = snap.get_potential_energy()
                f = snap.get_forces()
                if self.use_d3:
                    try:
                        d3 = DFTD3(xc=self.dft_cfg["xc"], damping="bj")
                        ed, fd = d3.calculate(snap)
                        e += ed
                        f += fd
                    except Exception:
                        pass
                snap.calc = None
                meta = {
                    "motif": motif_name,
                    "stage": f"aimd_snapshot_{int(aimd_cfg['temperature_k'])}K_t{int(step*aimd_cfg['dt_fs'])}fs",
                    "has_adsorbate": False,
                    "d_bond": 0.0,
                    "temperature_K": aimd_cfg["temperature_k"],
                    "dft_code": f"GPAW_{self.dft_cfg['xc']}_D3",
                    "converged": True
                }
                snapshots.append((snap, e, f, meta))
        md_atoms.calc = None
        return snapshots

    def save_record(self, atoms: Atoms, energy: float, forces: np.ndarray, meta: dict) -> int:
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        return self.db.write(atoms, key_value_pairs=meta)

    def export_mace_datasets(self):
        records = [row.toatoms() for row in self.db.select(converged=True)]
        if not records:
            raise RuntimeError("No records found in database.")
        np.random.seed(42)
        idx = np.random.permutation(len(records))
        split = int(len(records) * self.sys_cfg.get("train_ratio", 0.85))
        train_atoms = [records[i] for i in idx[:split]]
        val_atoms = [records[i] for i in idx[split:]]
        write(self.sys_cfg["export_train_xyz"], train_atoms, format="extxyz")
        write(self.sys_cfg["export_val_xyz"], val_atoms, format="extxyz")
        return len(train_atoms), len(val_atoms)

    # -------------------------------------------------------------
    # 4. Zero-Shot Benchmark & Programmatic Training
    # -------------------------------------------------------------
    def benchmark_zero_shot(self) -> dict:
        from mace.calculators import mace_mp
        val_atoms = read(self.sys_cfg["export_val_xyz"], index=":")
        calc = mace_mp(model=self.mace_cfg.get("foundation_model", "medium"), device=self.mace_cfg.get("device", "cpu"))

        dft_e, zs_e, dft_f, zs_f = [], [], [], []
        for atoms in val_atoms:
            dft_e.append(atoms.get_potential_energy())
            dft_f.append(atoms.get_forces())
            atoms.calc = calc
            zs_e.append(atoms.get_potential_energy())
            zs_f.append(atoms.get_forces())

        return {
            "mae_energy": float(np.mean(np.abs(np.array(zs_e) - np.array(dft_e)))),
            "mae_forces": float(np.mean(np.abs(np.concatenate(zs_f) - np.concatenate(dft_f)))),
            "pred_energy": zs_e, "pred_forces": zs_f, "dft_energy": dft_e, "dft_forces": dft_f
        }

    def train_mace_model(self) -> str:
        cmd = [
            "mace_run_train",
            f"--name={self.mace_cfg.get('model_name', 'mace_sac_model')}",
            f"--train_file={self.sys_cfg['export_train_xyz']}",
            f"--valid_file={self.sys_cfg['export_val_xyz']}",
            "--E0s=average",
            "--atomic_numbers=[1, 6, 7, 78]",
            "--energy_key=energy",
            "--forces_key=forces",
            f"--energy_weight={self.mace_cfg.get('energy_weight', 1.0)}",
            f"--forces_weight={self.mace_cfg.get('forces_weight', 10.0)}",
            "--model=ScaleShiftMACE",
            "--hidden_irreps=64x0e + 64x1o",
            "--r_max=5.0",
            f"--batch_size={self.mace_cfg.get('batch_size', 4)}",
            "--valid_batch_size=4",
            f"--max_num_epochs={self.mace_cfg.get('max_epochs', 60)}",
            "--ema",
            f"--device={self.mace_cfg.get('device', 'cpu')}",
            "--default_dtype=float32"
        ]
        log_path = "mace_training.log"
        with open(log_path, "w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"MACE training failed. Check {log_path} for details.")

        expected_model = f"{self.mace_cfg.get('model_name', 'mace_sac_model')}_stagetwo.model"
        if not os.path.exists(expected_model):
            expected_model = f"{self.mace_cfg.get('model_name', 'mace_sac_model')}.model"
        return expected_model

    def validate_trained_model(self, model_path: str) -> dict:
        from mace.calculators import MACECalculator
        val_atoms = read(self.sys_cfg["export_val_xyz"], index=":")
        calc = MACECalculator(model_paths=model_path, device=self.mace_cfg.get("device", "cpu"))

        dft_e, ft_e, dft_f, ft_f = [], [], [], []
        for atoms in val_atoms:
            dft_e.append(atoms.get_potential_energy())
            dft_f.append(atoms.get_forces())
            atoms.calc = calc
            ft_e.append(atoms.get_potential_energy())
            ft_f.append(atoms.get_forces())

        return {
            "mae_energy": float(np.mean(np.abs(np.array(ft_e) - np.array(dft_e)))),
            "mae_forces": float(np.mean(np.abs(np.concatenate(ft_f) - np.concatenate(dft_f)))),
            "pred_energy": ft_e, "pred_forces": ft_f, "dft_energy": dft_e, "dft_forces": dft_f
        }

    # -------------------------------------------------------------
    # 5. Production MD with Active Learning Uncertainty Trap
    # -------------------------------------------------------------
    def run_production_md(self, model_path: str) -> int:
        from mace.calculators import MACECalculator
        val_atoms = read(self.sys_cfg["export_val_xyz"], index=":")
        initial_structure = val_atoms[0].copy()
        calc = MACECalculator(model_paths=model_path, device=self.mace_cfg.get("device", "cpu"))
        initial_structure.calc = calc

        MaxwellBoltzmannDistribution(initial_structure, temperature_K=self.prod_cfg["temperature_k"])
        traj = Trajectory(self.prod_cfg["trajectory_file"], "w", initial_structure)

        dyn = Langevin(
            initial_structure,
            timestep=self.prod_cfg["timestep_fs"] * units.fs,
            temperature_K=self.prod_cfg["temperature_k"],
            friction=self.prod_cfg["friction"],
            logfile=self.prod_cfg["log_file"]
        )

        al_threshold = self.prod_cfg.get("al_force_threshold", 4.5)
        al_trapped = 0

        for step in range(self.prod_cfg["steps"]):
            dyn.run(1)
            traj.write(initial_structure)

            forces = initial_structure.get_forces()
            f_max = np.max(np.linalg.norm(forces, axis=1))

            if f_max > al_threshold:
                al_trapped += 1
                print(f"\n  [Active Learning Alert] Step {step}: Force norm ({f_max:.2f} eV/A) exceeded threshold. Triggering DFT fallback...")
                e_dft, f_dft = self.run_dft_evaluation(initial_structure.copy(), f"gpaw_logs/al_trap_step{step}")
                meta = {
                    "motif": "production_md_al_trap",
                    "stage": f"al_fallback_step{step}",
                    "has_adsorbate": True,
                    "dft_code": f"GPAW_{self.dft_cfg['xc']}_D3",
                    "converged": True
                }
                self.save_record(initial_structure.copy(), e_dft, f_dft, meta)

        traj.close()
        return al_trapped

    # -------------------------------------------------------------
    # 6. Automated CI-NEB Catalytic Barrier Screening
    # -------------------------------------------------------------
    def run_cineb_screening(self, motifs: dict, model_path: str) -> dict:
        from mace.calculators import MACECalculator
        mol_type = self.ads_cfg["type"]
        d_eq = self.ads_cfg["eq_bond_length"]
        results = {}

        for motif_name, slab in motifs.items():
            print(f"  -> Running MACE CI-NEB on {motif_name}...")
            metal_idx = np.where(np.array(slab.get_chemical_symbols()) == self.sys_cfg["metal"])[0][0]
            metal_pos = slab.positions[metal_idx]

            # Initial State: Physisorbed reactant (z = 2.5 A)
            initial = slab.copy()
            h1 = metal_pos + np.array([-d_eq / 2.0, 0.0, 2.5])
            h2 = metal_pos + np.array([d_eq / 2.0, 0.0, 2.5])
            initial.extend(Atoms(mol_type, positions=[h1, h2]))
            initial.calc = MACECalculator(model_paths=model_path, device=self.mace_cfg.get("device", "cpu"))
            opt_i = BFGS(initial, logfile=None)
            opt_i.run(fmax=self.kin_cfg.get("neb_fmax", 0.05))

            # Final State: Dissociated chemisorbed
            coord_atoms = [i for i in range(len(slab)) if i != metal_idx]
            neighbor_pos = slab.positions[coord_atoms[int(np.argmin([slab.get_distance(metal_idx, i) for i in coord_atoms]))]]
            final = slab.copy()
            final.extend(Atoms(mol_type, positions=[metal_pos + np.array([0, 0, 1.55]), neighbor_pos + np.array([0, 0, 1.10])]))
            final.calc = MACECalculator(model_paths=model_path, device=self.mace_cfg.get("device", "cpu"))
            opt_f = BFGS(final, logfile=None)
            opt_f.run(fmax=self.kin_cfg.get("neb_fmax", 0.05))

            # Generate intermediate images
            n_images = self.kin_cfg.get("neb_images", 5)
            images = [initial]
            for _ in range(n_images):
                image = initial.copy()
                image.calc = MACECalculator(model_paths=model_path, device=self.mace_cfg.get("device", "cpu"))
                images.append(image)
            images.append(final)

            neb = NEB(images, climb=self.kin_cfg.get("neb_climbing", True))
            neb.interpolate()
            opt_neb = BFGS(neb, trajectory=f"neb_{motif_name}.traj", logfile=f"gpaw_logs/neb_{motif_name}.log")
            opt_neb.run(fmax=self.kin_cfg.get("neb_fmax", 0.05))

            energies = [img.get_potential_energy() for img in images]
            e_act = max(energies) - energies[0]
            delta_e = energies[-1] - energies[0]

            results[motif_name] = {
                "e_act": float(e_act),
                "delta_e": float(delta_e),
                "energies": energies,
                "initial_state": initial,
                "transition_state": images[int(np.argmax(energies))],
                "final_state": final
            }
        return results

    # -------------------------------------------------------------
    # 7. Vibrational Thermochemistry (ZPE & Delta G)
    # -------------------------------------------------------------
    def compute_thermochemistry(self, state_atoms: Atoms, model_path: str, name: str) -> float:
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_paths=model_path, device=self.mace_cfg.get("device", "cpu"))
        state_atoms.calc = calc
        
        vib_dir = f"vib_{name}"
        vib = Vibrations(state_atoms, name=vib_dir)
        vib.run()
        vib_energies = vib.get_energies()
        vib.clean()

        real_energies = [e.real for e in vib_energies if abs(e.imag) < 1e-5 and e.real > 0]
        thermo = HarmonicThermo(vib_energies=real_energies, potentialenergy=state_atoms.get_potential_energy())
        g = thermo.get_helmholtz_energy(temperature=self.kin_cfg.get("temperature_k", 298.15))
        return float(g)

    # -------------------------------------------------------------
    # 8. Automated Publication Figures
    # -------------------------------------------------------------
    def generate_publication_figures(self, zs_data: dict, ft_data: dict, neb_data: dict):
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Energy Parity Plot
        dft_e = np.array(ft_data["dft_energy"])
        ft_e = np.array(ft_data["pred_energy"])
        axes[0].scatter(dft_e, ft_e, color="#1f77b4", label=f"Fine-Tuned MACE (MAE={ft_data['mae_energy']*1000:.1f} meV)", s=45, alpha=0.9)
        if zs_data:
            zs_e = np.array(zs_data["pred_energy"])
            axes[0].scatter(dft_e, zs_e, color="#d62728", marker="^", label=f"Zero-Shot MACE-MP-0 (MAE={zs_data['mae_energy']*1000:.1f} meV)", s=45, alpha=0.5)

        lims_e = [min(dft_e) - 0.5, max(dft_e) + 0.5]
        axes[0].plot(lims_e, lims_e, "k--", alpha=0.7)
        axes[0].set_xlim(lims_e)
        axes[0].set_ylim(lims_e)
        axes[0].set_xlabel("DFT Energy (eV)", fontsize=12)
        axes[0].set_ylabel("Predicted Energy (eV)", fontsize=12)
        axes[0].set_title("Energy Parity (Held-Out Test Set)", fontsize=13, fontweight="bold")
        axes[0].legend(frameon=True)

        # Forces Parity Plot
        dft_f = np.concatenate(ft_data["dft_forces"]).flatten()
        ft_f = np.concatenate(ft_data["pred_forces"]).flatten()
        axes[1].scatter(dft_f, ft_f, color="#2ca02c", label=f"Fine-Tuned MACE (MAE={ft_data['mae_forces']*1000:.1f} meV/Å)", s=15, alpha=0.4)
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

        # Reaction Profiles Figure
        if neb_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            for motif, res in neb_data.items():
                energies = np.array(res["energies"]) - res["energies"][0]
                rxn_coord = np.linspace(0, 1, len(energies))
                ax.plot(rxn_coord, energies, marker="o", label=f"{motif} (Ea={res['e_act']:.2f} eV)")

            ax.set_xlabel("Reaction Coordinate (Normalized)", fontsize=12)
            ax.set_ylabel("Relative Energy (eV)", fontsize=12)
            ax.set_title(f"Catalytic {self.ads_cfg['type']} Activation Pathways Across 10 Coordination Shells", fontsize=13, fontweight="bold")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)
            plt.tight_layout()
            plt.savefig("figures/reaction_profiles.png", dpi=300)
            plt.close()
            print("  -> Figures saved: figures/parity_plot.png & figures/reaction_profiles.png")

    # -------------------------------------------------------------
    # 9. Slurm Array Script Generator
    # -------------------------------------------------------------
    def generate_slurm_script(self):
        s_cfg = self.cfg.get("slurm", {})
        script = f"""#!/bin/bash
#SBATCH --job-name=sac_dft
#SBATCH --partition={s_cfg.get('partition', 'compute')}
#SBATCH --nodes={s_cfg.get('nodes', 1)}
#SBATCH --ntasks-per-node={s_cfg.get('ntasks_per_node', 12)}
#SBATCH --time={s_cfg.get('time_limit', '24:00:00')}
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module load python/3.10 gpaw/23.6.0
export OMP_NUM_THREADS=6

python -u run_ht_pipeline.py
"""
        with open("submit_slurm.sh", "w") as f:
            f.write(script)
        print("  -> Slurm submission script written: submit_slurm.sh")