from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import time

import cv2
import numpy as np

from .lane import LaneDetector
from .pid import PIDController
from .perception import TrafficLightDetector, VehicleDetector


class AutonomousDrivingSystem:
    """
    Combined lane detection + perception + PID steering demo.
    The output is an annotated RGB frame and control metadata.
    """

    def __init__(
        self,
        vehicle_model_dir: Optional[Path] = None,
        pid_gains: Tuple[float, float, float] = (0.35, 0.01, 0.20),
    ) -> None:
        self.lane = LaneDetector()
        self.pid = PIDController(*pid_gains, output_limits=(-1.0, 1.0), integral_limit=1.5)
        self.vehicle = VehicleDetector(vehicle_model_dir) if vehicle_model_dir is not None else VehicleDetector(None)
        self.traffic = TrafficLightDetector()
        self._last_t = None

    def _dt(self) -> float:
        now = time.time()
        if self._last_t is None:
            self._last_t = now
            return 1 / 30.0
        dt = now - self._last_t
        self._last_t = now
        return max(dt, 1 / 120.0)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
        dt = self._dt()

        lane_rgb, lane_metrics = self.lane.process_frame(frame_bgr)
        steering = float(self.pid.update(error=lane_metrics["offset_m"], dt=dt))

        # Basic speed policy: slow down on curves or when a vehicle is present.
        curvature = lane_metrics["curvature_m"]
        base_throttle = 0.55
        curve_penalty = np.clip(1200.0 / max(curvature, 1.0), 0.0, 0.35)
        throttle = float(np.clip(base_throttle - curve_penalty, 0.2, 0.65))

        traffic = self.traffic.detect(frame_bgr)
        vehicle_result = self.vehicle.detect(frame_bgr)
        vehicle_count = len(vehicle_result.get("boxes", []))

        if traffic.state == "RED":
            throttle = 0.0
        elif traffic.state == "YELLOW":
            throttle *= 0.35

        if vehicle_count > 0:
            throttle *= 0.8

        overlay = cv2.cvtColor(lane_rgb, cv2.COLOR_RGB2BGR)

        annotated_vehicle = vehicle_result.get("annotated")
        if annotated_vehicle is not None:
            overlay = cv2.addWeighted(overlay, 0.75, annotated_vehicle, 0.25, 0)

        text_color = (255, 255, 255)
        cv2.putText(overlay, f"Steering: {steering:+.2f}", (40, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Throttle: {throttle:.2f}", (40, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Traffic light: {traffic.state} ({traffic.confidence:.2f})", (40, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Vehicles: {vehicle_count}", (40, 325),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)

        meta = {
            "steering": steering,
            "throttle": throttle,
            "lane": lane_metrics,
            "traffic_light": {"state": traffic.state, "confidence": traffic.confidence},
            "vehicle_count": vehicle_count,
        }
        return overlay, meta
