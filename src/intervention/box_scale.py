"""
Box-Scale Intervention Engine for Trajectory Prediction Models.

This module implements controlled counterfactual bounding-box transformations
on pedestrian trajectory sequences to probe whether models encode latent behavioral
priors (e.g. child / vulnerability hesitation) or suffer from perspective depth confusion.

Supports multiple empirical techniques runnable efficiently on CPU:
  1. Ground-Anchored Proportional Scaling (locks y2 and x_foot, scales w and h)
  2. Ground-Anchored Height-Only Scaling (locks y2 and x_foot, scales h, preserves w)
  3. Ground-Anchored Aspect-Adjusted Scaling (scales w and h with custom ratio)
  4. Centroid Scaling (Control Baseline: shrinks around center, shifts y2)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def apply_ground_anchored_scale(
    bboxes: np.ndarray,
    scale: float = 0.65,
    preserve_aspect_ratio: bool = True,
    width_scale: Optional[float] = None,
) -> np.ndarray:
    """
    Technique 1: Ground-Anchored Proportional Scaling.
    
    Reduces height and width while holding the ground contact point (y2)
    and horizontal center (x_foot) strictly invariant.
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
    
    # Coordinates anchored strictly at (x_foot, y_foot)
    new_y2 = y_foot
    new_y1 = y_foot - new_h
    new_x1 = x_foot - (new_w / 2.0)
    new_x2 = x_foot + (new_w / 2.0)
    
    intervened = np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)
    return intervened.reshape(orig_shape)


def apply_height_only_scale(
    bboxes: np.ndarray,
    scale: float = 0.65,
) -> np.ndarray:
    """
    Technique 2: Ground-Anchored Height-Only Scaling.
    
    Reduces only vertical height (y1 dropped toward y2) while keeping width
    completely unchanged. Simulates shorter stature without altering horizontal profile.
    """
    return apply_ground_anchored_scale(
        bboxes, scale=scale, preserve_aspect_ratio=False, width_scale=1.0
    )


def apply_centroid_scale(
    bboxes: np.ndarray,
    scale: float = 0.65,
) -> np.ndarray:
    """
    Technique 3: Centroid-Anchored Scaling (Control Baseline).
    
    Shrinks the box around its 2D center point.
    Because the bottom coordinate y2 moves upward toward the horizon,
    this explicitly simulates what an object looks like when farther away in perspective.
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
    """Extracts frame-by-frame kinematics from a bounding box sequence."""
    boxes = np.asarray(bboxes, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    foot_pts = np.stack([(x1 + x2) / 2.0, y2], axis=1)
    centroids = np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0], axis=1)
    
    dt = 1.0 / fps
    foot_vel = np.diff(foot_pts, axis=0) / dt
    foot_speed = np.linalg.norm(foot_vel, axis=1)
    
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
