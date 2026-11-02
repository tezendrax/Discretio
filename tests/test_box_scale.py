"""
Unit tests for Box-Scale Intervention Engine.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intervention.box_scale import (
    apply_ground_anchored_scale,
    apply_centroid_scale,
    extract_kinematics,
)


def test_ground_anchored_invariants():
    # Synthetic box sequence: 10 frames
    # [x1, y1, x2, y2]
    # width = 40, height = 100, foot_x = 120, foot_y = 300
    boxes = np.array([
        [100, 200, 140, 300],
        [105, 200, 145, 300],
        [110, 200, 150, 300],
    ], dtype=np.float32)

    scale = 0.65
    intervened = apply_ground_anchored_scale(boxes, scale=scale, preserve_aspect_ratio=True)

    # Invariant 1: y2 (ground contact) must remain EXACTLY identical
    np.testing.assert_allclose(intervened[:, 3], boxes[:, 3], err_msg="Foot y2 position modified!")

    # Invariant 2: foot x position (horizontal ground contact) must remain EXACTLY identical
    orig_foot_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    inter_foot_x = (intervened[:, 0] + intervened[:, 2]) / 2.0
    np.testing.assert_allclose(inter_foot_x, orig_foot_x, err_msg="Foot x position shifted!")

    # Invariant 3: Height must be scaled by scale
    orig_h = boxes[:, 3] - boxes[:, 1]
    inter_h = intervened[:, 3] - intervened[:, 1]
    np.testing.assert_allclose(inter_h, orig_h * scale, rtol=1e-5, err_msg="Height scale incorrect!")

    # Invariant 4: Width must be scaled by scale
    orig_w = boxes[:, 2] - boxes[:, 0]
    inter_w = intervened[:, 2] - intervened[:, 0]
    np.testing.assert_allclose(inter_w, orig_w * scale, rtol=1e-5, err_msg="Width scale incorrect!")

    # Invariant 5: y1 must move down (greater pixel value in top-down image coordinates)
    assert np.all(intervened[:, 1] > boxes[:, 1]), "Top of box did not move downward!"


def test_centroid_scale_difference():
    boxes = np.array([[100, 200, 140, 300]], dtype=np.float32)
    scale = 0.65

    ground_anchored = apply_ground_anchored_scale(boxes, scale=scale)
    centroid_scaled = apply_centroid_scale(boxes, scale=scale)

    # In centroid scaling, y2 moves up (smaller value), which simulates being farther away
    assert centroid_scaled[0, 3] < boxes[0, 3], "Centroid scaling did not shift ground level upward!"

    # Ground anchored must NOT shift y2
    assert ground_anchored[0, 3] == boxes[0, 3], "Ground anchored shifted ground level!"


def test_kinematics_extraction():
    boxes = np.array([
        [100, 200, 140, 300],
        [110, 200, 150, 300],
        [120, 200, 160, 300],
    ], dtype=np.float32)

    kinematics = extract_kinematics(boxes, fps=10.0)
    assert "foot_speed" in kinematics
    assert len(kinematics["foot_speed"]) == 2
    # Moving 10 pixels in 0.1s -> speed = 100 px/s
    np.testing.assert_allclose(kinematics["foot_speed"], [100.0, 100.0], rtol=1e-4)


if __name__ == "__main__":
    test_ground_anchored_invariants()
    test_centroid_scale_difference()
    test_kinematics_extraction()
    print("All unit tests passed successfully!")
