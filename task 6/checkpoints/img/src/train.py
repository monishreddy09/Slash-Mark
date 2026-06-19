from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .dataset import (
    AirSimSteeringDataset,
    get_class_distribution,
    train_val_split_indices,
)
from .model import build_model


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def compute_class_weights(dataset: AirSimSteeringDataset, indices: list[int]) -> torch.Tensor:
    counts = np.zeros(3, dtype=np.float32)
    for idx in indices:
        cls = dataset.samples[idx].class_id
        counts[cls] += 1

    counts[counts == 0] = 1.0
    weights = counts.sum() / (3.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_h", type=int, default=72)
    parser.add_argument("--img_w", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)

    dataset = AirSimSteeringDataset(
        dataset_dir=args.dataset_dir,
        img_size=(args.img_h, args.img_w),
        augment=False,
        use_zip=True,
    )
    print("Class distribution:", get_class_distribution(dataset))

    train_idx, val_idx = train_val_split_indices(len(dataset), val_fraction=0.2, seed=args.seed)
    train_ds = AirSimSteeringDataset(dataset_dir=args.dataset_dir, img_size=(args.img_h, args.img_w), augment=True, use_zip=True)
    val_ds = AirSimSteeringDataset(dataset_dir=args.dataset_dir, img_size=(args.img_h, args.img_w), augment=False, use_zip=True)

    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(val_ds, val_idx)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=3).to(device)

    class_weights = compute_class_weights(dataset, train_idx).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"
    metadata_path = out_dir / "class_names.json"

    metadata_path.write_text(json.dumps({
        "class_names": ["straight", "left", "right"],
        "img_size": [args.img_h, args.img_w],
    }, indent=2))

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_total = 0
        running_correct = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                running_correct += int((pred == y).sum().item())
                running_total += int(x.size(0))
                running_loss += float(loss.item()) * x.size(0)

            pbar.set_postfix(loss=running_loss / max(running_total, 1), acc=running_correct / max(running_total, 1))

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        print(f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}")

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": "SteeringNet",
                "img_size": [args.img_h, args.img_w],
                "class_names": ["straight", "left", "right"],
            }, best_path)
            print(f"Saved best checkpoint to {best_path}")

    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
