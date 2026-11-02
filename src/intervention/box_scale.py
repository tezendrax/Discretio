"""
Box-Scale Intervention Engine for Trajectory Prediction Models.

This module implements controlled counterfactual bounding-box transformations
on pedestrian trajectory sequences to probe whether models encode latent behavioral
priors (e.g. child / vulnerability hesitation) or suffer from perspective depth confusion.
"""

from typing import Tuple, Union
import numpy as np


def apply_ground_anchored_scale(
    bboxes: np.ndarray,
    scale: float = 0.65,
    preserve_aspect_ratio: bool = True,
    width_scale: Union[float, None] = None,
) -> np.ndarray:
    """
    Applies a ground-anchored box-scale intervention.
    
    The intervention reduces the height (and optionally width) of the bounding box
    while holding the ground contact point (the foot position: center-x, bottom-y)
    strictly invariant.
    
    Why:
      In monocular perspective, distance from camera to flat ground is determined
      by the vertical coordinate of the ground contact point (y2). By keeping y2
      unchanged and lowering y1, the physical ground position is preserved.
      A model understanding scene geometry should interpret this as a shorter/younger
      actor (e.g. child) at the SAME ground distance, rather than an adult farther away.
      
    Args:
        bboxes: Array of shape (..., 4) with coordinates [x1, y1, x2, y2].
        scale: Scale factor for height (default: 0.65 for typical child vs adult height).
        preserve_aspect_ratio: If True, scales width by the same factor.
        width_scale: Optional custom scale factor for width. If None, uses `scale`.
        
    Returns:
        intervened_bboxes: Array with same shape as input with transformed coordinates.
    """
    boxes = np.asarray(bboxes, dtype=np.float32)
    orig_shape = boxes.shape
    boxes = boxes.reshape(-1, 4)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    width = x2 - x1
    height = y2 - y1
    x_foot = (x1 + x2) / 2.0
    y_foot = y2  # Ground contact anchor
    
    w_scale = scale if width_scale is None else width_scale
    if not preserve_aspect_ratio and width_scale is None:
        w_scale = 1.0
        
    new_h = height * scale
    new_w = width * w_scale
    
    # Coordinates anchored at (x_foot, y_foot)
    new_y2 = y_foot
    new_y1 = y_foot - new_h
    new_x1 = x_foot - (new_w / 2.0)
    new_x2 = x_foot + (new_w / 2.0)
    
    intervened = np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)
    return intervened.reshape(orig_shape)


def apply_centroid_scale(
    bboxes: np.ndarray,
    scale: float = 0.65,
) -> np.ndarray:
    """
    Applies a centroid-anchored box-scale intervention (Control Baseline).
    
    Shrinks the box around its 2D geometric center.
    This serves as a negative control: because the bottom coordinate y2 moves upward
    toward the horizon, this explicitly mimics an object that is farther away in perspective.
    
    Args:
        bboxes: Array of shape (..., 4) with coordinates [x1, y1, x2, y2].
        scale: Scale factor (default: 0.65).
        
    Returns:
        intervened_bboxes: Array with same shape as input.
    """
    boxes = np.asarray(bboxes, dtype=np.float32)
    orig_shape = boxes.shape
    boxes = boxes.reshape(-1, 4)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    width = x2 - x1
    height = y2 - y1
    x_c = (x1 + x2) / 2.0
    y_c = (y1 + y2) / 2.0
    
    new_w = width * scale
    new_h = height * scale
    
    new_x1 = x_c - (new_w / 2.0)
    new_x2 = x_c + (new_w / 2.0)
    new_y1 = y_c - (new_h / 2.0)
    new_y2 = y_c + (new_h / 2.0)
    
    intervened = np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)
    return intervened.reshape(orig_shape)


def extract_kinematics(bboxes: np.ndarray, fps: float = 30.0) -> dict:
    """
    Extracts frame-by-frame kinematics from a bounding box sequence.
    
    Args:
        bboxes: Array of shape (T, 4) in [x1, y1, x2, y2] format.
        fps: Video frame rate.
        
    Returns:
        Dictionary containing foot coordinates, centroid coordinates,
        velocities, and speed profiles.
    """
    boxes = np.asarray(bboxes, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    foot_pts = np.stack([(x1 + x2) / 2.0, y2], axis=1)  # (T, 2)
    centroids = np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0], axis=1)  # (T, 2)
    
    dt = 1.0 / fps
    foot_vel = np.diff(foot_pts, axis=0) / dt  # (T-1, 2)
    foot_speed = np.linalg.norm(foot_vel, axis=1)  # (T-1,)
    
    width = x2 - x1
    height = y2 - y1

    return {
        "foot_pts": foot_pts,
        "centroids": centroids,
        "foot_vel": foot_vel,
        "foot_speed": foot_speed,
        "mean_speed": float(np.mean(foot_speed)) if len(foot_speed) > 0 else 0.0,
        "width": width,
        "height": height,
        "mean_height": float(np.mean(height)),
    }
