from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LEFT_ANGLE = -0.3141592653589793
RIGHT_ANGLE = 0.3141592653589793
STRAIGHT_ANGLE = 0.0


def angle_to_class(angle: float, tol: float = 1e-4) -> int:
    if abs(angle - STRAIGHT_ANGLE) <= tol:
        return 0
    if angle < 0:
        return 1
    return 2


def class_to_angle(cls: int) -> float:
    return {0: STRAIGHT_ANGLE, 1: LEFT_ANGLE, 2: RIGHT_ANGLE}[int(cls)]


def class_to_action_name(cls: int) -> str:
    return {0: "straight", 1: "left", 2: "right"}[int(cls)]


@dataclass
class SampleRef:
    path: str
    class_id: int


class AirSimSteeringDataset(Dataset):
    """
    Loads samples from the reference CSV and either:
      - extracted images under dataset/imgs/..., or
      - directly from dataset/imgs.zip
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        img_size: Tuple[int, int] = (72, 128),
        augment: bool = False,
        use_zip: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.img_size = img_size
        self.augment = augment

        csv_path = self.dataset_dir / "data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV: {csv_path}")

        self.samples: List[SampleRef] = []
        df = pd.read_csv(csv_path, header=None)
        for _, row in df.iterrows():
            img_path = str(row[0])
            angle = float(row[4])
            self.samples.append(SampleRef(img_path, angle_to_class(angle)))

        self.zip_file = None
        if use_zip:
            zip_path = self.dataset_dir / "imgs.zip"
            if zip_path.exists():
                self.zip_file = zipfile.ZipFile(zip_path, "r")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_image(self, rel_path: str) -> np.ndarray:
        extracted = self.dataset_dir / rel_path
        if extracted.exists():
            data = extracted.read_bytes()
        elif self.zip_file is not None:
            # zip contains entries like imgs/1.png
            with self.zip_file.open(rel_path, "r") as f:
                data = f.read()
        else:
            raise FileNotFoundError(
                f"Could not find {rel_path} in extracted dataset or imgs.zip"
            )

        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image: {rel_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _augment(self, img: np.ndarray, cls: int) -> tuple[np.ndarray, int]:
        # Random horizontal flip with steering inversion.
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
            if cls == 1:
                cls = 2
            elif cls == 2:
                cls = 1

        # Mild brightness and contrast jitter for sim-to-real robustness.
        if np.random.rand() < 0.5:
            alpha = 0.8 + np.random.rand() * 0.4  # contrast
            beta = int(np.random.uniform(-12, 12))  # brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Small blur/noise injection, light enough not to destroy labels.
        if np.random.rand() < 0.25:
            k = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        return img, cls

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = self._read_image(sample.path)
        cls = sample.class_id

        if self.augment:
            img, cls = self._augment(img, cls)

        img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        x = torch.from_numpy(img)
        y = torch.tensor(cls, dtype=torch.long)
        return x, y


def get_class_distribution(dataset: AirSimSteeringDataset) -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    for s in dataset.samples:
        counts[s.class_id] += 1
    return counts


def train_val_split_indices(n: int, val_fraction: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idxs = np.arange(n)
    rng.shuffle(idxs)
    val_n = int(round(n * val_fraction))
    val_idx = idxs[:val_n].tolist()
    train_idx = idxs[val_n:].tolist()
    return train_idx, val_idx
