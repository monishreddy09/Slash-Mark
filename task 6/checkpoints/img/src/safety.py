from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import cv2
import numpy as np


class Action(IntEnum):
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2


@dataclass
class SafetyDecision:
    obstacle_detected: bool
    recommended_action: Action
    confidence: float
    debug_score: float


def _central_roi(img: np.ndarray, top_ratio: float = 0.35, bottom_ratio: float = 0.85):
    h, w = img.shape[:2]
    y1 = int(h * top_ratio)
    y2 = int(h * bottom_ratio)
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def depth_safety_check(depth_m: np.ndarray, stop_distance_m: float = 2.5) -> SafetyDecision:
    """
    depth_m should be a metric depth image where larger = farther.
    """
    roi, _ = _central_roi(depth_m)
    # Ignore invalid/inf values.
    valid = roi[np.isfinite(roi) & (roi > 0)]
    if valid.size == 0:
        return SafetyDecision(False, Action.STRAIGHT, 0.0, float("inf"))

    min_depth = float(np.percentile(valid, 5))
    obstacle = min_depth < stop_distance_m

    # Compare left vs right free space in the lower half.
    h, w = roi.shape[:2]
    left = roi[:, : w // 2]
    right = roi[:, w // 2 :]
    left_score = float(np.nanmean(left[np.isfinite(left)])) if np.isfinite(left).any() else 0.0
    right_score = float(np.nanmean(right[np.isfinite(right)])) if np.isfinite(right).any() else 0.0

    if obstacle:
        recommended = Action.RIGHT if right_score > left_score else Action.LEFT
        confidence = float(np.clip((stop_distance_m - min_depth) / stop_distance_m, 0.0, 1.0))
    else:
        recommended = Action.STRAIGHT
        confidence = float(np.clip((min_depth - stop_distance_m) / max(stop_distance_m, 1e-3), 0.0, 1.0))

    return SafetyDecision(obstacle, recommended, confidence, min_depth)


def rgb_safety_check(rgb: np.ndarray, edge_threshold: float = 0.12) -> SafetyDecision:
    """
    Fallback obstacle heuristic when only RGB is available.
    Uses edge density in the lower-center region as a conservative proxy.
    """
    roi, _ = _central_roi(rgb)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edge_density = float(edges.mean() / 255.0)

    h, w = gray.shape[:2]
    left = edges[:, : w // 2].mean() / 255.0 if w > 1 else edge_density
    right = edges[:, w // 2 :].mean() / 255.0 if w > 1 else edge_density

    obstacle = edge_density > edge_threshold
    if obstacle:
        recommended = Action.RIGHT if right < left else Action.LEFT
        confidence = float(np.clip((edge_density - edge_threshold) / max(edge_threshold, 1e-3), 0.0, 1.0))
    else:
        recommended = Action.STRAIGHT
        confidence = float(np.clip((edge_threshold - edge_density) / max(edge_threshold, 1e-3), 0.0, 1.0))

    return SafetyDecision(obstacle, recommended, confidence, edge_density)
