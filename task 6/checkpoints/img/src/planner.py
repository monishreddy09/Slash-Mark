from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from .safety import Action, SafetyDecision


@dataclass
class PlannerConfig:
    confidence_threshold: float = 0.45
    obstacle_confidence_threshold: float = 0.20
    smoothing_window: int = 5


class HybridPlanner:
    """
    Fuses:
      - network steering logits
      - obstacle safety override
      - temporal smoothing/hysteresis
    """

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()
        self.history: Deque[int] = deque(maxlen=self.config.smoothing_window)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        logits = logits - logits.max()
        exps = np.exp(logits)
        return exps / exps.sum()

    def choose_action(
        self,
        logits: np.ndarray,
        safety: SafetyDecision | None = None,
    ) -> tuple[Action, dict]:
        probs = self._softmax(np.asarray(logits).reshape(-1))
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))

        self.history.append(pred)
        smoothed = int(np.bincount(np.array(self.history), minlength=3).argmax())

        chosen = Action(smoothed)
        reason = "network"

        if conf < self.config.confidence_threshold:
            chosen = Action.STRAIGHT
            reason = "low_confidence"

        if safety is not None and safety.obstacle_detected and safety.confidence >= self.config.obstacle_confidence_threshold:
            chosen = safety.recommended_action
            reason = "safety_override"

        debug = {
            "probabilities": probs.tolist(),
            "predicted_class": pred,
            "smoothed_class": smoothed,
            "confidence": conf,
            "reason": reason,
            "safety_obstacle": bool(safety.obstacle_detected) if safety else False,
            "safety_score": float(safety.confidence) if safety else 0.0,
        }
        return chosen, debug
