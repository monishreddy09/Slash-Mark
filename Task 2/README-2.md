# Cats vs Dogs CNN Classifier

This project builds a binary CNN model for classifying cat and dog images using TensorFlow/Keras.

## What it includes

- Python + TensorFlow/Keras
- CNN layers
- Data generators with augmentation
- Train/validation split support
- Model fitting with callbacks
- Validation curves
- Confusion matrix
- Single-image prediction script

## Dataset layout options

### Option 1: separate train/validation folders
```text
data/
  train/
    Cat/
    Dog/
  test/
    Cat/
    Dog/
```

### Option 2: one folder with class subfolders
```text
data/
  Cat/
  Dog/
```

The training script will automatically split it into train and validation sets.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train_cat_dog_cnn.py --data-dir data --epochs 25 --image-size 128
```

Or with explicit folders:

```bash
python train_cat_dog_cnn.py --train-dir data/train --val-dir data/test
```

## Outputs

The script saves these files in `outputs/`:

- `cat_dog_cnn_model.keras`
- `best_model.keras`
- `training_curves.png`
- `confusion_matrix.png`
- `class_indices.json`
- `training_history.json`

## Predict one image

```bash
python predict_cat_dog.py --model outputs/cat_dog_cnn_model.keras --image path/to/image.jpg
```

## Notes

- The model is a custom CNN, not a transfer-learning model.
- You can improve accuracy later by increasing image size, training epochs, or tuning augmentation.
