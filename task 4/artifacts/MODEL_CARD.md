# Credit Card Fraud Detection Model Card

## Overview
A fraud detection classifier trained on the Kaggle credit card fraud dataset with:
- feature engineering on `Time` and `Amount`
- imbalance handling through resampling
- threshold tuning using a cost-sensitive objective

## Selected model
- **Model:** balanced_rf
- **Validation PR-AUC:** 0.7479
- **Validation ROC-AUC:** 0.9729
- **Chosen threshold:** 0.89
- **Validation cost:** 388.00

## Test metrics
- **PR-AUC:** 0.7655
- **ROC-AUC:** 0.9783
- **Precision:** 0.8500
- **Recall:** 0.6939
- **F1:** 0.7640
- **Balanced accuracy:** 0.8468
- **Cost:** 762.00

## Intended use
Transaction risk scoring and research prototypes. This model is not a substitute for a production fraud platform.

## Monitoring checklist
- Track PR-AUC and recall over time
- Monitor data drift for `Amount`, `Time`, and PCA features
- Recalibrate threshold when fraud prevalence changes
- Review false negatives with priority
