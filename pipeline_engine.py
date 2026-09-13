"""
Core Engine: Synthesis, Sampling, AIMD, MLIP Training, Active Learning,
CI-NEB, Thermochemistry, and Matplotlib Figure Generation.
"""

import os
import sys
import glob
import re
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
        calc = mace_mp(model=self.mace_cfg.get("foundation_model", "small"), device=self.mace_cfg.get("device", "cpu"))

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

        al_threshold = self.prod_cfg.get("al_force_threshold", 2.5)
        force_abort = self.prod_cfg.get("force_abort_threshold", 8.0)
        cooldown_steps = self.prod_cfg.get("al_cooldown_steps", 25)
        z_cutoff = self.prod_cfg.get("al_z_cutoff", 4.0)

        al_trapped = 0
        last_dft_step = -100

        for step in range(self.prod_cfg["steps"]):
            dyn.run(1)
            traj.write(initial_structure)

            pos = initial_structure.get_positions()
            syms = np.array(initial_structure.get_chemical_symbols())

            # 1. Boundary check: lightweight adsorbate (H) desorbed into vacuum
            h_mask = (syms == "H")
            c_mask = (syms == "C")
            if np.any(h_mask) and np.any(c_mask):
                z_slab = np.mean(pos[c_mask, 2])
                z_h = pos[h_mask, 2]
                if np.any(z_h > z_slab + z_cutoff) or np.any(z_h < z_slab - 2.0):
                    print(f"\n  [MD Safety] H2 desorbed or drifted into vacuum at step {step} (z > {z_cutoff} Å). Stopping trajectory cleanly.")
                    break

            # 2. Force checks
            forces = initial_structure.get_forces()
            f_max = np.max(np.linalg.norm(forces, axis=1))

            # Severe non-physical explosion or nuclear clash
            if f_max > force_abort:
                print(f"\n  [MD Safety] Unphysical force spike ({f_max:.2f} eV/A > {force_abort} eV/A) at step {step}. Aborting trajectory.")
                break

            # Informative OOD region: DFT fallback with stride cooldown
            if f_max > al_threshold:
                if step - last_dft_step >= cooldown_steps:
                    al_trapped += 1
                    last_dft_step = step
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
    # 6. Automated CI-NEB Catalytic Barrier Screening (IDPP Robust)
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

            # Initial State: Physisorbed reactant (z ~ 2.3 A)
            initial = slab.copy()
            h1 = metal_pos + np.array([-d_eq / 2.0, 0.0, 2.3])
            h2 = metal_pos + np.array([d_eq / 2.0, 0.0, 2.3])
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
            n_images = self.kin_cfg.get("neb_images", 3)
            images = [initial]
            for _ in range(n_images):
                image = initial.copy()
                image.calc = MACECalculator(model_paths=model_path, device=self.mace_cfg.get("device", "cpu"))
                images.append(image)
            images.append(final)

            # Climbing Image NEB with IDPP interpolation
            neb = NEB(images, climb=self.kin_cfg.get("neb_climbing", True))
            neb.interpolate("idpp")
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
    # 8. Automated Publication Figures (All 4 Plots, Polished)
    # -------------------------------------------------------------
    def generate_publication_figures(self, zs_data: dict = None, ft_data: dict = None, neb_data: dict = None):
        from mace.calculators import MACECalculator, mace_mp

        model_name = self.mace_cfg.get("model_name", "mace_sac_model")
        model_path = f"{model_name}_stagetwo.model"
        if not os.path.exists(model_path):
            model_path = f"{model_name}.model"

        val_path = self.sys_cfg.get("export_val_xyz", "mace_val.xyz")
        device = self.mace_cfg.get("device", "cpu")

        # 1. Parity Data Resolution: compute from disk if missing/empty
        if not ft_data and os.path.exists(val_path) and os.path.exists(model_path):
            print("  -> Evaluating test set parity from disk artifacts...")
            val_atoms = read(val_path, index=":")
            calc_ft = MACECalculator(model_paths=model_path, device=device)
            calc_zs = mace_mp(model=self.mace_cfg.get("foundation_model", "small"), device=device)

            dft_e_pa, ft_e_pa, zs_e_pa = [], [], []
            dft_f, ft_f = [], []

            for atoms in val_atoms:
                n_at = len(atoms)
                e_dft = atoms.get_potential_energy()
                f_dft = atoms.get_forces()

                atoms.calc = calc_ft
                e_ft = atoms.get_potential_energy()
                f_ft = atoms.get_forces()

                atoms.calc = calc_zs
                e_zs = atoms.get_potential_energy()

                dft_e_pa.append(e_dft / n_at)
                ft_e_pa.append(e_ft / n_at)
                zs_e_pa.append(e_zs / n_at)
                dft_f.append(f_dft)
                ft_f.append(f_ft)

            dft_e_pa = np.array(dft_e_pa)
            ft_e_pa = np.array(ft_e_pa)
            zs_e_pa = np.array(zs_e_pa)

            ft_data = {
                "dft_energy_pa": dft_e_pa,
                "pred_energy_pa": ft_e_pa,
                "dft_forces": dft_f,
                "pred_forces": ft_f,
                "mae_energy_pa": float(np.mean(np.abs(ft_e_pa - dft_e_pa))),
                "mae_forces": float(np.mean(np.abs(np.concatenate(ft_f) - np.concatenate(dft_f))))
            }
            zs_data = {
                "pred_energy_pa": zs_e_pa,
                "mae_energy_pa": float(np.mean(np.abs(zs_e_pa - dft_e_pa)))
            }

        # 2. CI-NEB Data Resolution: parse neb_*.traj files if missing/empty
        if not neb_data:
            neb_files = sorted(glob.glob("neb_*.traj"))
            if neb_files:
                print(f"  -> Reading {len(neb_files)} CI-NEB trajectories from disk...")
                neb_data = {}
                n_images = self.kin_cfg.get("neb_images", 3)
                total_band = n_images + 2

                for traj_path in neb_files:
                    motif = os.path.basename(traj_path).replace("neb_", "").replace(".traj", "")
                    try:
                        traj = Trajectory(traj_path)
                        n_frames = len(traj)
                        if n_frames < total_band:
                            continue

                        converged_band = [traj[i] for i in range(-total_band, 0)]
                        energies = []
                        for img in converged_band:
                            try:
                                energies.append(img.get_potential_energy())
                            except Exception:
                                img.calc = MACECalculator(model_paths=model_path, device=device)
                                energies.append(img.get_potential_energy())

                        energies = np.array(energies)
                        e_act = max(energies) - energies[0]
                        delta_e = energies[-1] - energies[0]
                        neb_data[motif] = {
                            "energies": energies.tolist(),
                            "e_act": float(e_act),
                            "delta_e": float(delta_e)
                        }
                    except Exception as err:
                        print(f"  (!) Warning: Could not parse {traj_path}: {err}")

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # =========================================================================
        # FIGURE 1: Parity Plots (Energy Normalized per Atom & Cartesian Forces)
        # =========================================================================
        if ft_data:
            dft_e = np.array(ft_data["dft_energy_pa"])
            ft_e = np.array(ft_data["pred_energy_pa"])
            dft_f = np.concatenate(ft_data["dft_forces"]).flatten()
            ft_f = np.concatenate(ft_data["pred_forces"]).flatten()

            fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

            # Strict Monotonic Limits for Per-Atom Energy
            all_e = [dft_e, ft_e]
            if zs_data:
                zs_e = np.array(zs_data["pred_energy_pa"])
                all_e.append(zs_e)
            e_min, e_max = float(np.min(all_e)) - 0.05, float(np.max(all_e)) + 0.05

            axes[0].plot([e_min, e_max], [e_min, e_max], "k--", lw=1.2, alpha=0.7, label="Ideal Parity (y = x)")
            if zs_data:
                axes[0].scatter(dft_e, zs_e, color="#d62728", marker="^", s=45, alpha=0.6,
                                label=f"Zero-Shot MACE-MP-0 (MAE={zs_data['mae_energy_pa']*1000:.1f} meV/atom)")
            axes[0].scatter(dft_e, ft_e, color="#1f77b4", marker="o", s=50, edgecolors="k", lw=0.5,
                            label=f"Fine-Tuned MACE (MAE={ft_data['mae_energy_pa']*1000:.1f} meV/atom)")
            axes[0].set_xlim(e_min, e_max)
            axes[0].set_ylim(e_min, e_max)
            axes[0].set_xlabel("DFT Energy (eV/atom)", fontsize=11, fontweight="bold")
            axes[0].set_ylabel("Predicted Energy (eV/atom)", fontsize=11, fontweight="bold")
            axes[0].set_title("Energy Parity (Held-Out Test Set)", fontsize=12, fontweight="bold")
            axes[0].legend(frameon=True, loc="upper left")

            # Forces Parity
            f_min, f_max = float(np.min([dft_f, ft_f])), float(np.max([dft_f, ft_f]))
            axes[1].plot([f_min, f_max], [f_min, f_max], "k--", lw=1.2, alpha=0.7, label="Ideal Parity (y = x)")
            axes[1].scatter(dft_f, ft_f, color="#2ca02c", s=18, alpha=0.45, edgecolors="none",
                            label=f"Fine-Tuned MACE (MAE={ft_data['mae_forces']*1000:.1f} meV/Å)")
            axes[1].set_xlim(f_min, f_max)
            axes[1].set_ylim(f_min, f_max)
            axes[1].set_xlabel("DFT Cartesian Forces (eV/Å)", fontsize=11, fontweight="bold")
            axes[1].set_ylabel("Predicted Forces (eV/Å)", fontsize=11, fontweight="bold")
            axes[1].set_title("Forces Parity", fontsize=12, fontweight="bold")
            axes[1].legend(frameon=True, loc="upper left")

            plt.tight_layout()
            plt.savefig("figures/parity_plot.png")
            plt.close()
            print("  -> Output written: figures/parity_plot.png")

        # =========================================================================
        # FIGURE 2: Reaction Profiles Across Coordination Motifs
        # =========================================================================
        if neb_data:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            for motif, res in neb_data.items():
                energies = np.array(res["energies"]) - res["energies"][0]
                rxn_coord = np.linspace(0, 1, len(energies))
                ax.plot(rxn_coord, energies, marker="o", lw=1.8, label=f"{motif} (Ea={res['e_act']:.2f} eV)")

            ax.set_xlabel("Reaction Coordinate (Normalized)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Relative Energy (eV)", fontsize=11, fontweight="bold")
            ax.set_title(f"Catalytic {self.ads_cfg['type']} Activation Across Coordination Shells", fontsize=12, fontweight="bold")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)
            plt.tight_layout()
            plt.savefig("figures/reaction_profiles.png")
            plt.close()
            print("  -> Output written: figures/reaction_profiles.png")

        # =========================================================================
        # FIGURE 3: Volcano Plot (Strict Single- vs Divacancy Categorization)
        # =========================================================================
        if neb_data:
            single_vac_motifs = ["Pt-C3", "Pt-N1C2", "Pt-N2C1", "Pt-N3"]
            
            def sort_key(name):
                n_m = re.search(r"N(\d+)", name)
                n = int(n_m.group(1)) if n_m else (3 if name == "Pt-N3" else (4 if name == "Pt-N4" else 0))
                is_single = name in single_vac_motifs
                return (0 if is_single else 1, n, name)

            sorted_motifs = sorted(neb_data.keys(), key=sort_key)
            ea_vals = [neb_data[m]["e_act"] for m in sorted_motifs]
            de_vals = [neb_data[m].get("delta_e", neb_data[m]["energies"][-1] - neb_data[m]["energies"][0]) for m in sorted_motifs]

            fig, ax1 = plt.subplots(figsize=(11, 5.5), dpi=300)
            ax2 = ax1.twinx()

            x_pos = np.arange(len(sorted_motifs))

            ax1.plot(x_pos, ea_vals, color="#1f77b4", marker="s", lw=2, ms=7, label=r"Activation Barrier ($E_a$)")
            ax1.set_ylabel(r"$E_a$ (eV)", color="#1f77b4", fontsize=11, fontweight="bold")
            ax1.tick_params(axis="y", labelcolor="#1f77b4")
            ax1.set_ylim(-0.1, max(ea_vals) * 1.15)

            ax2.plot(x_pos, de_vals, color="#d62728", marker="o", lw=2, ms=7, ls="--", label=r"Reaction Energy ($\Delta E$)")
            ax2.set_ylabel(r"$\Delta E$ (eV)", color="#d62728", fontsize=11, fontweight="bold")
            ax2.tick_params(axis="y", labelcolor="#d62728")
            ax2.axhline(0.0, color="gray", ls=":", lw=1.2, alpha=0.8)

            # Strict 4 single-vac motifs (indices 0, 1, 2, 3)
            n_single = sum(1 for m in sorted_motifs if m in single_vac_motifs)
            ax1.axvspan(-0.5, n_single - 0.5, color="#2ca02c", alpha=0.08, label="Single-Vacancy (3-coord)")
            ax1.axvspan(n_single - 0.5, len(sorted_motifs) - 0.5, color="#ff7f0e", alpha=0.08, label="Divacancy (4-coord)")

            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(sorted_motifs, rotation=35, ha="right", fontweight="bold")
            ax1.set_xlabel("Coordination Motif (Grouped by Vacancy Type & N-Content)", fontsize=11, fontweight="bold")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, fontsize=9.5)

            plt.title("Catalytic Activation and Reaction Energies Across Coordination Motifs", fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig("figures/volcano_plot.png")
            plt.close()
            print("  -> Output written: figures/volcano_plot.png")

        # =========================================================================
        # FIGURE 4: Active Learning Production MD Stability & Traps
        # =========================================================================
        md_traj_path = self.prod_cfg.get("trajectory_file", "production_md.traj")
        if os.path.exists(md_traj_path):
            try:
                traj = Trajectory(md_traj_path)
                dt_fs = self.prod_cfg.get("timestep_fs", 0.5)
                al_thresh = self.prod_cfg.get("al_force_threshold", 2.5)

                times, pot_energies, temps, max_forces = [], [], [], []
                for step_i, atoms in enumerate(traj):
                    times.append(step_i * dt_fs)
                    pot_energies.append(atoms.get_potential_energy())
                    ekin = atoms.get_kinetic_energy()
                    temps.append((2.0 * ekin) / (3.0 * len(atoms) * units.kB))
                    try:
                        max_forces.append(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
                    except Exception:
                        max_forces.append(0.0)

                times = np.array(times)
                pot_energies = np.array(pot_energies) - pot_energies[0]
                temps = np.array(temps)
                max_forces = np.array(max_forces)

                fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, dpi=300)

                # Subplot 1: Potential Energy Drift
                axes[0].plot(times, pot_energies, color="#1f77b4", lw=1.5)
                axes[0].set_ylabel(r"$\Delta E_{\mathrm{pot}}$ (eV)", fontsize=11, fontweight="bold")
                axes[0].set_title("Active Learning Production MD Dynamics & Stability", fontsize=12, fontweight="bold")

                # Subplot 2: Instantaneous Temperature & Target
                axes[1].plot(times, temps, color="#ff7f0e", lw=1.2, label="Instantaneous T")
                axes[1].axhline(self.prod_cfg.get("temperature_k", 300), color="black", ls="--", alpha=0.7, label="Target 300 K")
                axes[1].set_ylabel("Temperature (K)", fontsize=11, fontweight="bold")
                axes[1].legend(loc="upper right", frameon=True)

                # Subplot 3: Max Force & AL Threshold
                axes[2].plot(times, max_forces, color="#2ca02c", lw=1.2, label=r"$F_{\max}$")
                axes[2].axhline(al_thresh, color="red", ls="--", lw=1.5, label=f"AL DFT Fallback ({al_thresh} eV/Å)")
                axes[2].set_ylabel("Max Force (eV/Å)", fontsize=11, fontweight="bold")
                axes[2].set_xlabel("Simulation Time (fs)", fontsize=11, fontweight="bold")
                axes[2].legend(loc="upper right", frameon=True)

                # Trap Annotation
                if len(times) > 0:
                    axes[2].annotate(f"Trapped at t = {times[-1]:.1f} fs\n(Desorption / AL alert)",
                                     xy=(times[-1], max_forces[-1]), xytext=(times[-1] * 0.65, max_forces[-1] + 1.0),
                                     arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
                                     fontsize=9.5, fontweight="bold",
                                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6))

                plt.tight_layout()
                plt.savefig("figures/md_trajectory.png")
                plt.close()
                print("  -> Output written: figures/md_trajectory.png")
            except Exception as e:
                print(f"  (!) Warning: Could not generate MD trajectory plot: {e}")

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