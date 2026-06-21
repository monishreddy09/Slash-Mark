# Credit Card Fraud Detection Pipeline

This project upgrades the reference notebook into a full end-to-end fraud detection pipeline.

## What it does
- Loads the Kaggle `creditcard.csv` dataset
- Engineers features from `Time` and `Amount`
- Handles class imbalance with SMOTE and undersampling
- Compares multiple models
- Uses cross-validation
- Reports both ROC-AUC and PR-AUC
- Tunes the decision threshold with a cost-sensitive metric
- Saves the final model and evaluation artifacts

## Files
- `train_fraud_model.py` — main training and evaluation script
- `requirements.txt` — dependencies
- `MODEL_CARD.md` — generated after training
- `monitoring_checklist.md` — deployment and monitoring notes

## Dataset
The Kaggle dataset is not included in this zip. Download it separately and point the script to the CSV:

Dataset: Credit Card Fraud Detection

Source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains anonymized credit card transactions and is used for fraud detection research.

## Note:
The dataset file (creditcard.csv) is not included in this repository due to its size. Download it from Kaggle and place it in the project directory before running the code.

```bash
python train_fraud_model.py --data /path/to/creditcard.csv --output artifacts
```

## Output
The script writes:
- `fraud_detection_model.joblib`
- `cv_results.csv`
- `validation_results.csv`
- `test_metrics.json`
- `classification_report.txt`
- `test_pr_curve.png`
- `test_roc_curve.png`
- `model_metadata.json`
- `MODEL_CARD.md`

## Notes
- The project is designed to work with the standard Kaggle dataset schema.
- The threshold is selected on validation data using a cost function where false negatives are more expensive than false positives.
