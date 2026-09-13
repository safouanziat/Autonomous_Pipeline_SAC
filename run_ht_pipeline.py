#!/usr/bin/env python3
"""
Autonomous Master Orchestrator with Granular Stage Control.
Usage:
    python run_ht_pipeline.py                      # Uses config.yaml booleans
    python run_ht_pipeline.py --only neb figures   # Runs ONLY NEB and plotting
    python run_ht_pipeline.py --skip dft aimd      # Skips DFT, starts from export/train
"""

import os
import sys
import time
import argparse
import yaml
from ase.io import read
from pipeline_engine import SACPipelineEngine


def parse_arguments(config_stages: dict):
    parser = argparse.ArgumentParser(description="Autonomous SAC-MLIP Pipeline Runner")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[
            "synth", "motifs", "dft", "aimd", "export",
            "zeroshot", "train", "validate", "neb", "md", "figures"
        ],
        help="Execute ONLY specified stages (overrides config.yaml)"
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=[
            "synth", "motifs", "dft", "aimd", "export",
            "zeroshot", "train", "validate", "neb", "md", "figures"
        ],
        help="Skip specified stages"
    )
    args = parser.parse_args()

    key_map = {
        "synth": "synthesize_substrate",
        "motifs": "generate_motifs",
        "dft": "static_dft",
        "aimd": "aimd",
        "export": "export_dataset",
        "zeroshot": "zero_shot_baseline",
        "train": "train_mace",
        "validate": "validate_mace",
        "neb": "cineb_screening",
        "md": "production_md",
        "figures": "generate_figures",
    }

    stages = config_stages.copy()

    if args.only:
        for k in stages:
            stages[k] = False
        for short_name in args.only:
            stages[key_map[short_name]] = True

    if args.skip:
        for short_name in args.skip:
            stages[key_map[short_name]] = False

    return stages


def main():
    print("=================================================================")
    print("   AUTONOMOUS PLATFORM FOR SINGLE-ATOM CATALYSIS & MACE MLIP     ")
    print("=================================================================")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    stages = parse_arguments(config.get("stages", {}))
    engine = SACPipelineEngine(config)
    os.makedirs("gpaw_logs", exist_ok=True)

    if config["system"].get("executor") == "slurm" and not os.environ.get("SLURM_JOB_ID"):
        print("\n[Slurm Mode Enabled] Generating batch script submit_slurm.sh...")
        engine.generate_slurm_script()
        print("Submit to your cluster via: sbatch submit_slurm.sh")
        return

    print("\n[Active Execution Plan]")
    for stage_name, active in stages.items():
        status = "ENABLED " if active else "DISABLED"
        print(f"  - {stage_name:<25} : {status}")

    # -------------------------------------------------------------
    # PHASE 0 & 1: Geometry Foundation
    # -------------------------------------------------------------
    metal = config['system']['metal']
    coord_elem = config['system']['coordination_element']
    synth_file = f"synthesized_{metal}_{coord_elem}4_relaxed.xyz"

    if stages.get("synthesize_substrate"):
        print(f"\n[Phase 0/9] In-silico synthesis of {metal}-{coord_elem}4...")
        t0 = time.time()
        pristine_slab = engine.synthesize_pristine_sac()
        print(f"  -> Substrate ready ({time.time() - t0:.1f}s).")
    else:
        print(f"\n[Phase 0/9] Skipped. Loading existing '{synth_file}'...")
        if not os.path.exists(synth_file):
            print(f"(!) Error: '{synth_file}' not found. Run Phase 0 first.")
            sys.exit(1)
        pristine_slab = read(synth_file)

    if stages.get("generate_motifs"):
        print("\n[Phase 1/9] Constructing 10 physical coordination motifs...")
        motifs = engine.generate_motifs(pristine_slab)
        print(f"  -> Motifs: {list(motifs.keys())}")
    else:
        print("\n[Phase 1/9] Skipped. Deriving motifs in-memory...")
        motifs = engine.generate_motifs(pristine_slab)

    # -------------------------------------------------------------
    # PHASE 2: Static DFT
    # -------------------------------------------------------------
    if stages.get("static_dft"):
        print(f"\n[Phase 2/9] Evaluating Static DFT Configurations ({config['adsorbate']['type']})...")
        for motif_name, slab in motifs.items():
            dataset = engine.sample_static_configurations(motif_name, slab)
            for idx, (atoms, meta) in enumerate(dataset):
                meta["motif"] = motif_name
                meta["dft_code"] = f"GPAW_{config['dft']['xc']}_D3"
                stage = meta["stage"]

                if len(list(engine.db.select(motif=motif_name, stage=stage))) > 0:
                    continue

                log_label = f"gpaw_logs/{motif_name}_{stage}"
                print(f"  [{motif_name}] {stage:<30} ... ", end="", flush=True)
                t_calc = time.time()
                try:
                    energy, forces = engine.run_dft_evaluation(atoms, log_label=log_label)
                    meta["converged"] = True
                    row_id = engine.save_record(atoms, energy, forces, meta)
                    print(f"OK ({time.time()-t_calc:.1f}s) | E = {energy:.4f} eV [DB ID: {row_id}]")
                except Exception as err:
                    print(f"FAILED: {err}")
    else:
        print("\n[Phase 2/9] Static DFT Phase Skipped.")

    # -------------------------------------------------------------
    # PHASE 3: AIMD Sampling
    # -------------------------------------------------------------
    if stages.get("aimd"):
        print("\n[Phase 3/9] Running Langevin AIMD Sampling...")
        temp_k = int(config["sampling"]["aimd"]["temperature_k"])
        total_fs = int(config["sampling"]["aimd"]["total_fs"])
        for motif_name, slab in motifs.items():
            check_stage = f"aimd_snapshot_{temp_k}K_t{total_fs}fs"
            if len(list(engine.db.select(motif=motif_name, stage=check_stage))) > 0:
                continue
            print(f"  -> AIMD on {motif_name} ({temp_k} K)...")
            try:
                snaps = engine.run_aimd_trajectory(motif_name, slab)
                for s_atoms, e, f, meta in snaps:
                    engine.save_record(s_atoms, e, f, meta)
            except Exception as err:
                print(f"  (!) AIMD failed on {motif_name}: {err}")
    else:
        print("\n[Phase 3/9] AIMD Phase Skipped.")

    # -------------------------------------------------------------
    # PHASE 4: Dataset Export
    # -------------------------------------------------------------
    if stages.get("export_dataset"):
        print("\n[Phase 4/9] Exporting Train/Validation ExtXYZ Datasets...")
        n_train, n_val = engine.export_mace_datasets()
        print(f"  -> {n_train} training frames, {n_val} validation frames.")
    else:
        print("\n[Phase 4/9] Dataset Export Skipped.")

    # -------------------------------------------------------------
    # PHASE 5: Zero-Shot Baseline
    # -------------------------------------------------------------
    zs_metrics = None
    if stages.get("zero_shot_baseline"):
        print("\n[Phase 5/9] Evaluating Foundation MACE-MP-0 Baseline...")
        try:
            zs_metrics = engine.benchmark_zero_shot()
            print(f"  -> Zero-Shot Energy MAE : {zs_metrics['mae_energy']*1000:.1f} meV")
            print(f"  -> Zero-Shot Forces MAE : {zs_metrics['mae_forces']*1000:.1f} meV/A")
        except Exception as e:
            print(f"  (!) Zero-shot benchmark skipped: {e}")
    else:
        print("\n[Phase 5/9] Zero-Shot Baseline Skipped.")

    # -------------------------------------------------------------
    # PHASE 6: MACE Fine-Tuning & Validation
    # -------------------------------------------------------------
    model_name = config.get("mace_mlip", {}).get("model_name", "mace_sac_model")
    trained_model = f"{model_name}_stagetwo.model"
    if not os.path.exists(trained_model):
        trained_model = f"{model_name}.model"

    if stages.get("train_mace"):
        print("\n[Phase 6/9] Fine-Tuning MACE on Bounded Dataset...")
        trained_model = engine.train_mace_model()
        print(f"  -> Potential compiled: {trained_model}")
    else:
        print(f"\n[Phase 6/9] Training Skipped. Using model: {trained_model}")

    ft_metrics = None
    if stages.get("validate_mace"):
        print("\n  Validating model against held-out DFT points...")
        try:
            ft_metrics = engine.validate_trained_model(trained_model)
            print(f"  -> Fine-Tuned Energy MAE : {ft_metrics['mae_energy']*1000:.1f} meV")
            print(f"  -> Fine-Tuned Forces MAE : {ft_metrics['mae_forces']*1000:.1f} meV/A")
            if zs_metrics:
                print(f"  ==> ACCURACY BOOST: {zs_metrics['mae_energy']/ft_metrics['mae_energy']:.1f}x (Energy)")
        except Exception as e:
            print(f"  (!) Validation skipped: {e}")

    # -------------------------------------------------------------
    # PHASE 7: CI-NEB Catalytic Barrier Screening
    # -------------------------------------------------------------
    neb_results = {}
    if stages.get("cineb_screening"):
        print("\n[Phase 7/9] Running MACE CI-NEB Pathways & Thermochemistry...")
        neb_results = engine.run_cineb_screening(motifs, trained_model)

        print("\n  ================================================================================")
        print(f"  CATALYTIC VOLCANO SCREENING TABLE ({config['kinetics_and_thermo']['temperature_k']} K, {config['kinetics_and_thermo']['pressure_bar']} bar)")
        print("  ================================================================================")
        print(f"  {'Motif':<18} | {'E_a (eV)':<10} | {'Delta_E (eV)':<12} | {'Delta_G (eV)':<18}")
        print("  --------------------------------------------------------------------------------")

        for motif, res in neb_results.items():
            delta_g_str = "N/A"
            if config["kinetics_and_thermo"].get("calc_free_energy", True):
                try:
                    g_init = engine.compute_thermochemistry(res["initial_state"], trained_model, f"{motif}_init")
                    g_ts = engine.compute_thermochemistry(res["transition_state"], trained_model, f"{motif}_ts")
                    delta_g_str = f"{g_ts - g_init:.2f}"
                except Exception:
                    pass
            print(f"  {motif:<18} | {res['e_act']:<10.2f} | {res['delta_e']:<12.2f} | {delta_g_str:<18}")
        print("  ================================================================================")
    else:
        print("\n[Phase 7/9] CI-NEB Screening Skipped.")

    # -------------------------------------------------------------
    # PHASE 8: Production MD
    # -------------------------------------------------------------
    if stages.get("production_md"):
        print(f"\n[Phase 8/9] Launching Production MD at {config['production_md']['temperature_k']} K...")
        al_trapped = engine.run_production_md(trained_model)
        print(f"  -> MD complete. Trapped {al_trapped} high-uncertainty configurations.")
    else:
        print("\n[Phase 8/9] Production MD Skipped.")

    # -------------------------------------------------------------
    # PHASE 9: Figures
    # -------------------------------------------------------------
    if stages.get("generate_figures"):
        print("\n[Phase 9/9] Generating Publication Figures...")
        if ft_metrics and neb_results:
            engine.generate_publication_figures(zs_metrics, ft_metrics, neb_results)
        else:
            print("  (!) Need validation data and NEB data to generate full figures. Skipping.")
    else:
        print("\n[Phase 9/9] Figure Generation Skipped.")

    print("\n=================================================================")
    print("EXECUTION COMPLETED ACCORDING TO STAGE PLAN.")
    print("=================================================================\n")


if __name__ == "__main__":
    main()