"""
Empirical Experiment Runner: Multi-Technique Box-Scale Interventions on CPU.

Benchmarks multiple box-scaling techniques against real IDD-PeD manifests
and trajectory forecasting models to evaluate latent behavioral priors vs. depth artifacts.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.iddped_loader import IDDPeDTrajectoryDataset
from src.intervention.box_scale import (
    apply_ground_anchored_scale,
    apply_height_only_scale,
    apply_centroid_scale,
)
from src.models.trajectory_probe import ConstantVelocityPredictor, NeuralBBoxPredictor
from src.evaluation.metrics import compute_hesitation_and_depth_diagnostics


def run_multi_technique_experiment(
    data_path: str = None,
    num_samples: int = 50,
    output_dir: str = "results",
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print(" DISCRETIO: Empirical Multi-Technique Box-Scale Benchmark (CPU Mode)")
    print(" Comparing: Ground-Anchored Proportional, Height-Only, Sweep & Centroid Control")
    print("=" * 80)

    # 1. Load dataset (loads local IDD-PeD cache or calibrated benchmark tracks)
    dataset = IDDPeDTrajectoryDataset(data_path=data_path, t_obs=15, t_pred=30)
    obs_batch, gt_pred_batch = dataset.get_batch(batch_size=num_samples)

    # 2. Check Curated Child Reference Manifests
    manifest_path = "manifests/selected_child_clips.json"
    child_clips_info = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            child_clips = json.load(f)
        child_clips_info = {
            "total_curated_clips": len(child_clips),
            "outcomes": pd.Series([c["outcome"] for c in child_clips]).value_counts().to_dict(),
            "mean_ped_speed_mps": float(np.mean([c["ped_speed_mps"] for c in child_clips])),
            "mean_ttc_s": float(np.mean([c["ttc_s"] for c in child_clips])),
        }
        print(f"\n[Manifest] Loaded {len(child_clips)} curated IDD-PeD child benchmark events.")
        print(f"  Outcomes: {child_clips_info['outcomes']}")
        print(f"  Empirical Mean Child Speed: {child_clips_info['mean_ped_speed_mps']:.2f} m/s")

    # 3. Define Empirical Intervention Techniques
    techniques = {
        "Tech 1: Ground-Anchored Proportional (α=0.65)": lambda b: apply_ground_anchored_scale(b, scale=0.65, preserve_aspect_ratio=True),
        "Tech 2: Ground-Anchored Height-Only (α=0.65)": lambda b: apply_height_only_scale(b, scale=0.65),
        "Tech 3: Moderate Scale (α=0.80)": lambda b: apply_ground_anchored_scale(b, scale=0.80, preserve_aspect_ratio=True),
        "Tech 4: Extreme Child Scale (α=0.50)": lambda b: apply_ground_anchored_scale(b, scale=0.50, preserve_aspect_ratio=True),
        "Tech 5: Centroid Scaling (Control Baseline)": lambda b: apply_centroid_scale(b, scale=0.65),
    }

    # 4. Models to Evaluate (all run on CPU)
    models = {
        "Constant Velocity (Kinematic)": ConstantVelocityPredictor(),
        "Neural BBox LSTM (Learned)": NeuralBBoxPredictor(),
    }

    benchmark_results = {}
    summary_rows = []

    for model_name, model in models.items():
        print(f"\n---> Evaluating Model: {model_name}...")
        pred_orig = model.predict(obs_batch, t_pred=30)
        model_results = {}

        for tech_name, tech_fn in techniques.items():
            intervened_obs = tech_fn(obs_batch)
            pred_intervened = model.predict(intervened_obs, t_pred=30)

            diag = compute_hesitation_and_depth_diagnostics(pred_orig, pred_intervened)
            model_results[tech_name] = diag

            summary_rows.append({
                "Model": model_name,
                "Technique": tech_name,
                "D_tau (px)": round(diag["trajectory_divergence_d_tau"], 2),
                "Speed Ratio": round(diag["speed_ratio"], 3),
                "Foot Shift (px)": round(diag["foot_y_shift"], 2),
                "Perspective Preserved": not diag["perspective_depth_confusion"],
                "Finding": "Depth Confused" if diag["perspective_depth_confusion"] else ("Scale Invariant" if diag["scale_invariant"] else "Behavioral Shift")
            })

            print(f"  [{tech_name}]: D_tau = {diag['trajectory_divergence_d_tau']:.2f}px | Speed Ratio = {diag['speed_ratio']:.3f} | Foot Drift = {diag['foot_y_shift']:.2f}px")

        benchmark_results[model_name] = model_results

    # 5. Format & Save Output
    results_payload = {
        "experiment_metadata": {
            "num_samples": num_samples,
            "device": "CPU",
            "t_obs": 15,
            "t_pred": 30,
            "dataset": "IDD-PeD",
        },
        "curated_child_manifest": child_clips_info,
        "benchmark_results": benchmark_results,
        "summary_table": summary_rows,
    }

    result_path = os.path.join(output_dir, "box_scale_results.json")
    with open(result_path, "w") as f:
        json.dump(results_payload, f, indent=2)

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print(" EMPIRICAL SUMMARY TABLE:")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    return results_payload


if __name__ == "__main__":
    run_multi_technique_experiment()
