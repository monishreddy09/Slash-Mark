#!/usr/bin/env python3
"""
Cats vs Dogs CNN trainer.

Supports three dataset layouts:
1) --train-dir and --val-dir are provided
2) data_dir/train and data_dir/test exist
3) data_dir contains class subfolders directly, and an internal validation split is used

Expected class folder names:
- Cat / Dog
or
- cats / dogs
or any two class folder names (binary classification)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN to classify cat vs dog images.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root dataset directory.")
    parser.add_argument("--train-dir", type=str, default=None, help="Optional explicit training directory.")
    parser.add_argument("--val-dir", type=str, default=None, help="Optional explicit validation directory.")
    parser.add_argument("--image-size", type=int, default=128, help="Square image size used for training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum number of training epochs.")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Internal split if no val dir is available.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Where to save the model and plots.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_dirs(args: argparse.Namespace) -> Tuple[Path, Optional[Path]]:
    if args.train_dir and args.val_dir:
        return Path(args.train_dir), Path(args.val_dir)

    data_dir = Path(args.data_dir)

    # Repo-style layout: data/train and data/test
    train_candidate = data_dir / "train"
    val_candidate = data_dir / "test"
    if train_candidate.exists() and val_candidate.exists():
        return train_candidate, val_candidate

    # Generic layout: class folders are directly under data_dir, use internal split.
    return data_dir, None


def build_generators(
    train_dir: Path,
    val_dir: Optional[Path],
    image_size: int,
    batch_size: int,
    validation_split: float,
    seed: int,
):
    if val_dir is not None:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
        )
        val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
        train_gen = train_datagen.flow_from_directory(
            str(train_dir),
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode="binary",
            shuffle=True,
            seed=seed,
        )
        val_gen = val_datagen.flow_from_directory(
            str(val_dir),
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode="binary",
            shuffle=False,
        )
    else:
        # Train/validation split from a single directory with class subfolders.
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            validation_split=validation_split,
        )
        val_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=validation_split)

        train_gen = train_datagen.flow_from_directory(
            str(train_dir),
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode="binary",
            subset="training",
            shuffle=True,
            seed=seed,
        )
        val_gen = val_datagen.flow_from_directory(
            str(train_dir),
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode="binary",
            subset="validation",
            shuffle=False,
            seed=seed,
        )
    return train_gen, val_gen


def build_model(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.30)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.50)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="cat_dog_cnn")
    return model


def plot_training_curves(history: tf.keras.callbacks.History, output_path: Path) -> None:
    hist = history.history
    acc_key = "accuracy" if "accuracy" in hist else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in hist else "val_acc"

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(hist.get("loss", []), label="train")
    plt.plot(hist.get("val_loss", []), label="validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Crossentropy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist.get(acc_key, []), label="train")
    plt.plot(hist.get(val_acc_key, []), label="validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=30, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir, val_dir = resolve_dirs(args)
    train_gen, val_gen = build_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    class_indices = train_gen.class_indices
    class_names = [None] * len(class_indices)
    for name, idx in class_indices.items():
        class_names[idx] = name

    model = build_model((args.image_size, args.image_size, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    checkpoint_path = output_dir / "best_model.keras"
    cbs = [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=cbs,
        verbose=1,
    )

    # Save final model and class mapping.
    final_model_path = output_dir / "cat_dog_cnn_model.keras"
    model.save(final_model_path)

    with open(output_dir / "class_indices.json", "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=2)

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    plot_training_curves(history, output_dir / "training_curves.png")

    # Confusion matrix and classification report on validation data
    val_gen.reset()
    val_probs = model.predict(val_gen, verbose=0).ravel()
    val_pred = (val_probs >= 0.5).astype(int)
    y_true = val_gen.classes

    cm = confusion_matrix(y_true, val_pred)
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    report = classification_report(
        y_true,
        val_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("\nValidation classification report:\n")
    print(report)

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Saved model to: {final_model_path}")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
