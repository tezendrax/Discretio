"""
DISCRETIO Diagnostic Metrics for Box-Scale Intervention.

Calculates:
  1. Trajectory Divergence (D_tau)
  2. Hesitation / Velocity Shift Ratio
  3. Ground-Contact Perspective Drift (Depth Confusion Diagnostic)
  4. Jensen-Shannon Divergence (JSD) & Calibration Score
"""

from typing import Dict, List, Tuple, Union
import numpy as np
from scipy.spatial.distance import jensenshannon


def compute_trajectory_divergence(
    pred_orig: np.ndarray,
    pred_intervened: np.ndarray,
    use_foot_point: bool = True,
) -> float:
    """
    Computes average displacement divergence (D_tau) between original
    and intervened rollouts over the prediction horizon.
    
    Args:
        pred_orig: Shape (T_pred, 4) or (B, T_pred, 4)
        pred_intervened: Shape (T_pred, 4) or (B, T_pred, 4)
        use_foot_point: If True, uses (center_x, y2); else uses centroid.
        
    Returns:
        d_tau: Mean Euclidean distance across all future timesteps.
    """
    p_orig = np.asarray(pred_orig, dtype=np.float32)
    p_inter = np.asarray(pred_intervened, dtype=np.float32)
    
    if p_orig.ndim == 2:
        p_orig = np.expand_dims(p_orig, 0)
        p_inter = np.expand_dims(p_inter, 0)
        
    if use_foot_point:
        pts_orig = np.stack([(p_orig[..., 0] + p_orig[..., 2]) / 2.0, p_orig[..., 3]], axis=-1)
        pts_inter = np.stack([(p_inter[..., 0] + p_inter[..., 2]) / 2.0, p_inter[..., 3]], axis=-1)
    else:
        pts_orig = np.stack([(p_orig[..., 0] + p_orig[..., 2]) / 2.0, (p_orig[..., 1] + p_orig[..., 3]) / 2.0], axis=-1)
        pts_inter = np.stack([(p_inter[..., 0] + p_inter[..., 2]) / 2.0, (p_inter[..., 1] + p_inter[..., 3]) / 2.0], axis=-1)
        
    diff = np.linalg.norm(pts_orig - pts_inter, axis=-1)  # (B, T_pred)
    return float(np.mean(diff))


def compute_hesitation_and_depth_diagnostics(
    pred_orig: np.ndarray,
    pred_intervened: np.ndarray,
    fps: float = 30.0,
) -> Dict[str, float]:
    """
    Diagnoses whether rollout shift indicates a Behavioral Prior (hesitation)
    or Perspective Depth Confusion (misjudged distance).
    
    Args:
        pred_orig: Shape (B, T_pred, 4) or (T_pred, 4)
        pred_intervened: Shape (B, T_pred, 4) or (T_pred, 4)
        fps: Frames per second.
        
    Returns:
        dict containing speed_ratio, foot_y_shift, is_depth_confused, is_hesitating.
    """
    p_orig = np.asarray(pred_orig, dtype=np.float32)
    p_inter = np.asarray(pred_intervened, dtype=np.float32)
    
    if p_orig.ndim == 2:
        p_orig = np.expand_dims(p_orig, 0)
        p_inter = np.expand_dims(p_inter, 0)
        
    # Foot positions
    foot_orig = np.stack([(p_orig[..., 0] + p_orig[..., 2]) / 2.0, p_orig[..., 3]], axis=-1)
    foot_inter = np.stack([(p_inter[..., 0] + p_inter[..., 2]) / 2.0, p_inter[..., 3]], axis=-1)
    
    dt = 1.0 / fps
    vel_orig = np.diff(foot_orig, axis=1) / dt  # (B, T_pred - 1, 2)
    vel_inter = np.diff(foot_inter, axis=1) / dt
    
    speed_orig = np.linalg.norm(vel_orig, axis=-1)  # (B, T_pred - 1)
    speed_inter = np.linalg.norm(vel_inter, axis=-1)
    
    mean_speed_orig = float(np.mean(speed_orig)) + 1e-6
    mean_speed_inter = float(np.mean(speed_inter))
    speed_ratio = mean_speed_inter / mean_speed_orig
    
    # Vertical ground-point shift: (y2_inter - y2_orig)
    # Negative shift indicates feet moved upward toward horizon (farther away)
    foot_y_shift = float(np.mean(foot_inter[..., 1] - foot_orig[..., 1]))
    
    # Trajectory divergence
    d_tau = compute_trajectory_divergence(p_orig, p_inter, use_foot_point=True)
    
    return {
        "trajectory_divergence_d_tau": d_tau,
        "mean_speed_orig": mean_speed_orig,
        "mean_speed_intervened": mean_speed_inter,
        "speed_ratio": speed_ratio,
        "foot_y_shift": foot_y_shift,
        # Interpretations
        "behavioral_hesitation_detected": bool(speed_ratio < 0.90),
        "perspective_depth_confusion": bool(foot_y_shift < -5.0),
        "scale_invariant": bool(d_tau < 1.0),
    }


def compute_calibration_score(p_model: np.ndarray, e_empirical: np.ndarray) -> float:
    """
    Computes prior calibration score: C = 1 - JSD(P, E).
    
    Args:
        p_model: Discrete probability distribution from model rollouts.
        e_empirical: Discrete probability distribution from real data.
        
    Returns:
        calibration: Score in [0, 1].
    """
    p = np.asarray(p_model, dtype=np.float64)
    e = np.asarray(e_empirical, dtype=np.float64)
    
    p = p / np.sum(p)
    e = e / np.sum(e)
    
    # scipy.spatial.distance.jensenshannon returns square root of JSD, so square it
    jsd_val = float(jensenshannon(p, e, base=2.0) ** 2)
    return float(np.clip(1.0 - jsd_val, 0.0, 1.0))

