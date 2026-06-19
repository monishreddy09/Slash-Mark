#!/usr/bin/env python3
"""Predict cat vs dog for a single image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a single cat/dog image.")
    parser.add_argument("--model", type=str, required=True, help="Path to a saved .keras or .h5 model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image to classify.")
    parser.add_argument("--image-size", type=int, default=128, help="Image size used during training.")
    parser.add_argument("--class-a", type=str, default="Cat", help="Label for output below 0.5.")
    parser.add_argument("--class-b", type=str, default="Dog", help="Label for output at or above 0.5.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = tf.keras.models.load_model(args.model)
    image = load_img(args.image, target_size=(args.image_size, args.image_size))
    arr = img_to_array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr, verbose=0)[0][0])
    label = args.class_b if prob >= 0.5 else args.class_a

    print(f"Predicted class: {label}")
    print(f"Probability of {args.class_b}: {prob:.4f}")


if __name__ == "__main__":
    main()
