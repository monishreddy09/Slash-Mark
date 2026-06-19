from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .controller import FlightController
from .dataset import class_to_action_name
from .model import build_model
from .planner import HybridPlanner, PlannerConfig
from .safety import Action


def load_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(num_classes=3).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def predict_action(model, frame_rgb, device, img_size):
    x = cv2.resize(frame_rgb, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).cpu().numpy().reshape(-1)
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--confidence_threshold", type=float, default=0.45)
    parser.add_argument("--obstacle_confidence_threshold", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    img_h, img_w = ckpt.get("img_size", [72, 128])

    planner = HybridPlanner(
        PlannerConfig(
            confidence_threshold=args.confidence_threshold,
            obstacle_confidence_threshold=args.obstacle_confidence_threshold,
        )
    )

    try:
        import airsim
    except Exception as e:
        raise RuntimeError(
            "AirSim is required for infer_airsim.py. Install it and run an AirSim environment."
        ) from e

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    if client.getMultirotorState().landed_state == airsim.LandedState.Landed:
        client.takeoffAsync().join()

    controller = FlightController(client)

    print("Starting inference loop. Press Ctrl+C to stop.")
    while True:
        frame_rgb, depth = controller.capture_airsim(include_depth=args.use_depth)
        logits = predict_action(model, frame_rgb, device, (img_h, img_w))
        safety = controller.safety_from_frame(frame_rgb, depth if args.use_depth else None)
        action, debug = planner.choose_action(logits, safety=safety)

        print(
            f"action={action.name:8s} "
            f"net={debug['predicted_class']} "
            f"conf={debug['confidence']:.3f} "
            f"reason={debug['reason']} "
            f"obstacle={debug['safety_obstacle']} "
            f"safety={debug['safety_score']:.3f}"
        )
        controller.act_airsim(action)

if __name__ == "__main__":
    main()
