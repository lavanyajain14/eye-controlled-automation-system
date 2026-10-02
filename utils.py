"""
utils.py — Geometric helpers: EAR computation, iris centre, gaze normalisation.
"""

import numpy as np


def _lm_to_px(lm, idx: int, w: int, h: int) -> "tuple[float, float]":
    """Convert a normalised landmark to pixel coordinates."""
    return lm[idx].x * w, lm[idx].y * h


def compute_ear(lm, indices: list, w: int, h: int) -> float:
    """
    Eye Aspect Ratio (EAR) — Soukupova & Čech (2016).

    EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 · ‖p1-p4‖)

    indices order: [p1, p2, p3, p4, p5, p6]
      p1, p4 = horizontal corners
      p2, p6 = upper/lower vertical pair 1
      p3, p5 = upper/lower vertical pair 2
    """
    pts = np.array([_lm_to_px(lm, i, w, h) for i in indices], dtype=np.float32)

    # Vertical distances
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    C = np.linalg.norm(pts[0] - pts[3])

    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


def get_iris_center(lm, iris_indices: list, w: int, h: int) -> "tuple[float, float]":
    """Average of iris landmark positions → iris centre in pixels."""
    pts = np.array([_lm_to_px(lm, i, w, h) for i in iris_indices], dtype=np.float32)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def normalize_gaze(
    iris_center: "tuple[float, float]",
    eye_corners: "list[tuple[int,int]]",
) -> "tuple[float, float]":
    """
    Map iris centre relative to eye corners → normalised gaze [-1, +1].

    Returns (gaze_x, gaze_y):
      gaze_x: -1 = looking far left, +1 = looking far right
      gaze_y: -1 = looking up,       +1 = looking down
    """
    (lx, ly), (rx, ry) = eye_corners[0], eye_corners[1]

    eye_width  = abs(rx - lx) or 1
    eye_height = eye_width * 0.4   # approximate

    mid_x = (lx + rx) / 2
    mid_y = (ly + ry) / 2

    gaze_x = (iris_center[0] - mid_x) / (eye_width  / 2)
    gaze_y = (iris_center[1] - mid_y) / (eye_height / 2)

    return (
        float(np.clip(gaze_x, -1, 1)),
        float(np.clip(gaze_y, -1, 1)),
    )
