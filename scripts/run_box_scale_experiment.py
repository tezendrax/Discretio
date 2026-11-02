"""
Automated Experiment Runner: Box-Scale Intervention on IDD-PeD.

Evaluates trajectory forecasting models on original vs. intervened bounding box sequences
to probe for latent behavioral priors (hesitation) vs perspective depth confusion.
"""

import os
import sys
import json
import numpy as np

# Ensure root path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.iddped_loader import IDDPeDTrajectoryDataset
from src.intervention.box_scale import apply_ground_anchored_scale, apply_centroid_scale
from src.models.trajectory_probe import ConstantVelocityPredictor, NeuralBBoxPredictor
from src.evaluation.metrics import compute_hesitation_and_depth_diagnostics


def run_experiment(
    data_path: str = None,
    scale: float = 0.65,
    num_samples: int = 50,
    output_dir: str = "results",
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print(" DISCRETIO: Box-Scale Intervention Experiment on IDD-PeD")
    print(f" Scale Factor: {scale} (Simulating Child Bounding Box)")
    print(f" Evaluation Samples: {num_samples}")
    print("=" * 70)

    # 1. Load dataset
    dataset = IDDPeDTrajectoryDataset(data_path=data_path, t_obs=15, t_pred=30)
    obs_batch, gt_pred_batch = dataset.get_batch(batch_size=num_samples)

    # 2. Generate Interventions
    # A. Ground-Anchored Intervention (Our proposed method: foot invariant)
    obs_child_ground = apply_ground_anchored_scale(obs_batch, scale=scale, preserve_aspect_ratio=True)

    # B. Centroid-Scaled Intervention (Negative Control: feet shift upward -> mimics farther distance)
    obs_child_centroid = apply_centroid_scale(obs_batch, scale=scale)

    # Verify foot invariant on ground-anchored
    max_foot_err = np.max(np.abs(obs_child_ground[..., 3] - obs_batch[..., 3]))
    print(f"[Verification] Ground-anchored max foot y-shift: {max_foot_err:.6f} px (Must be ~0.0)")

    # 3. Models to evaluate
    models = {
        "Constant Velocity (Kinematic)": ConstantVelocityPredictor(),
        "Neural BBox LSTM (Learned)": NeuralBBoxPredictor(),
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n---> Evaluating Model: {model_name}...")

        # Rollout on original
        pred_orig = model.predict(obs_batch, t_pred=30)

        # Rollout on Ground-Anchored Child Box
        pred_ground = model.predict(obs_child_ground, t_pred=30)

        # Rollout on Centroid-Scaled Box
        pred_centroid = model.predict(obs_child_centroid, t_pred=30)

        # Diagnostic Metrics for Ground-Anchored
        diag_ground = compute_hesitation_and_depth_diagnostics(pred_orig, pred_ground)

        # Diagnostic Metrics for Centroid-Scaled Control
        diag_centroid = compute_hesitation_and_depth_diagnostics(pred_orig, pred_centroid)

        results[model_name] = {
            "ground_anchored": diag_ground,
            "centroid_scaled_control": diag_centroid,
        }

        print(f"  [Ground-Anchored Intervention]:")
        print(f"    - Trajectory Divergence (D_tau): {diag_ground['trajectory_divergence_d_tau']:.2f} px")
        print(f"    - Speed Ratio (v_child / v_adult): {diag_ground['speed_ratio']:.3f}")
        print(f"    - Ground Foot Y-Shift: {diag_ground['foot_y_shift']:.2f} px")
        print(f"    - Behavioral Hesitation Detected: {diag_ground['behavioral_hesitation_detected']}")
        print(f"    - Depth Confusion Detected: {diag_ground['perspective_depth_confusion']}")
        print(f"    - Scale Invariant: {diag_ground['scale_invariant']}")

        print(f"  [Centroid-Scaled Control]:")
        print(f"    - Trajectory Divergence (D_tau): {diag_centroid['trajectory_divergence_d_tau']:.2f} px")
        print(f"    - Speed Ratio (v_control / v_adult): {diag_centroid['speed_ratio']:.3f}")
        print(f"    - Ground Foot Y-Shift: {diag_centroid['foot_y_shift']:.2f} px")
        print(f"    - Depth Confusion Detected: {diag_centroid['perspective_depth_confusion']}")

    # Save results
    result_path = os.path.join(output_dir, "box_scale_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] Experiment results saved to: {result_path}")
    return results


if __name__ == "__main__":
    run_experiment()

