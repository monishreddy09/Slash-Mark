from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from .dataset import class_to_angle
from .safety import rgb_safety_check, depth_safety_check, Action


@dataclass
class ActuationConfig:
    speed_mps: float = 1.0
    turn_radians: float = 0.3141592653589793
    altitude_m: float = -2.0
    command_duration_s: float = 0.5


class FlightController:
    def __init__(self, client=None, config: ActuationConfig | None = None) -> None:
        self.client = client
        self.config = config or ActuationConfig()

    def preprocess(self, rgb: np.ndarray, size: tuple[int, int] = (72, 128)) -> torch.Tensor:
        img = cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0)

    def capture_airsim(self, include_depth: bool = False):
        if self.client is None:
            raise RuntimeError("AirSim client is not connected.")
        import airsim  # optional dependency

        raw = self.client.simGetImage("0", airsim.ImageType.Scene)
        if raw is None:
            raise RuntimeError("AirSim did not return an RGB image.")
        arr = np.frombuffer(airsim.string_to_uint8_array(raw), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode AirSim image.")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth = None
        if include_depth:
            resp = self.client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)]
            )[0]
            depth = np.array(resp.image_data_float, dtype=np.float32).reshape(resp.height, resp.width)
        return rgb, depth

    def act_airsim(self, action: Action):
        if self.client is None:
            raise RuntimeError("AirSim client is not connected.")
        import airsim

        yaw_delta = {Action.LEFT: -self.config.turn_radians,
                     Action.STRAIGHT: 0.0,
                     Action.RIGHT: self.config.turn_radians}[action]

        pitch, roll, yaw = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        new_yaw = yaw + yaw_delta
        vx = self.config.speed_mps * float(np.cos(new_yaw))
        vy = self.config.speed_mps * float(np.sin(new_yaw))

        self.client.moveByVelocityZAsync(
            vx,
            vy,
            self.config.altitude_m,
            self.config.command_duration_s,
            airsim.DrivetrainType.ForwardOnly,
            airsim.YawMode(False, 0),
        ).join()

    @staticmethod
    def safety_from_frame(rgb: np.ndarray, depth: np.ndarray | None = None):
        return depth_safety_check(depth) if depth is not None else rgb_safety_check(rgb)
