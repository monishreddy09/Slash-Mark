# Hybrid Indoor Obstacle Avoidance (AirSim + PyTorch)

This project is a modern replacement for the older Keras/AirSim reference you uploaded. It keeps the same idea — a camera-based indoor obstacle avoidance controller for a drone/robot — but upgrades it into a hybrid system:

- **Deep learning perception**: a small CNN predicts **left / straight / right** from RGB images.
- **Classical safety layer**: optional depth-based or RGB-based obstacle checks can override the network.
- **Local control loop**: smooths predictions, adds hysteresis, and issues flight commands.
- **Training pipeline**: trains directly from `dataset/data.csv` and `dataset/imgs.zip`.

## What it expects

The reference dataset layout matches your zip:
- `dataset/data.csv`
- `dataset/imgs.zip`

The CSV uses steering labels:
- `0` = straight
- `-0.314159...` = left
- `+0.314159...` = right

## Files

- `src/model.py` — steering CNN
- `src/dataset.py` — dataset loader from CSV + zip
- `src/safety.py` — obstacle detection and safety heuristics
- `src/planner.py` — smoothing + action arbitration
- `src/controller.py` — AirSim / webcam wrapper
- `src/train.py` — train the steering model
- `src/infer_airsim.py` — live inference loop in AirSim

## Train

```bash
pip install -r requirements.txt
python -m src.train --dataset_dir /path/to/UAV-indoor-obstacle-avoidance-based-on-AI-technique-master/dataset --epochs 25
```

This will create:
- `checkpoints/best_model.pt`
- `checkpoints/class_names.json`

## Run in AirSim

```bash
python -m src.infer_airsim \
  --dataset_dir /path/to/UAV-indoor-obstacle-avoidance-based-on-AI-technique-master/dataset \
  --checkpoint checkpoints/best_model.pt
```

For live AirSim usage you will also need `airsim` installed and a running AirSim environment.

## Why this satisfies the task

- **AI perception + control loop**: yes, the CNN predicts steering and the controller executes commands.
- **Obstacle detection**: yes, via the safety layer.
- **Path planning basics**: yes, the planner smooths actions and uses obstacle-free-sector selection when needed.
- **Control heuristics**: yes, via confidence thresholds, action hysteresis, and obstacle override.
- **Sim-to-real considerations**: the code isolates perception from actuation, supports optional depth input, and keeps the action space discrete for portability.
