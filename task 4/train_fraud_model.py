#!/usr/bin/env python3
"""
Credit Card Fraud Detection Pipeline
------------------------------------

This script trains and evaluates an imbalanced classification pipeline for the
Kaggle credit card fraud dataset.

Highlights
- Feature engineering for Amount and Time
- Imbalance handling with SMOTE and undersampling alternatives
- Multiple candidate models
- Cross-validation with ROC-AUC and PR-AUC
- Threshold tuning with a cost-sensitive objective
- Final test-set evaluation and model export

Usage
-----
python train_fraud_model.py --data /path/to/creditcard.csv --output artifacts
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
FP_COST = 1.0
FN_COST = 25.0  # fraud misses are usually much more expensive than false alarms


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create a few stable features from Amount and Time."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy()
        if not isinstance(X_df, pd.DataFrame):
            X_df = pd.DataFrame(X_df)

        if "Amount" in X_df.columns:
            amount = X_df["Amount"].astype(float).fillna(0.0)
            X_df["Amount_Log1p"] = np.log1p(np.abs(amount))
            X_df["Amount_Sqrt"] = np.sqrt(np.abs(amount))
            X_df["Amount_IsZero"] = (amount == 0).astype(int)

        if "Time" in X_df.columns:
            time_sec = X_df["Time"].astype(float).fillna(0.0)
            day_seconds = 24 * 60 * 60
            X_df["Time_Sin"] = np.sin(2 * np.pi * (time_sec % day_seconds) / day_seconds)
            X_df["Time_Cos"] = np.cos(2 * np.pi * (time_sec % day_seconds) / day_seconds)
            X_df["Time_Hour"] = ((time_sec // 3600) % 24).astype(float)

        return X_df


@dataclass
class ModelResult:
    name: str
    cv_pr_auc_mean: float
    cv_pr_auc_std: float
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    val_pr_auc: float
    val_roc_auc: float
    best_threshold: float
    val_cost: float
    val_f1: float
    val_recall: float
    val_precision: float


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    expected_cols = {"Class"}
    if not expected_cols.issubset(df.columns):
        raise ValueError("Dataset must contain a 'Class' target column.")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = FraudFeatureEngineer().fit_transform(df)
    return engineered


def make_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])
    X = add_engineered_features(X)
    return X, y


def build_candidates(random_state: int = RANDOM_STATE):
    common_scaler = StandardScaler()

    candidates = {
        "logreg_smote": ImbPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", common_scaler),
                ("sampler", SMOTE(random_state=random_state, k_neighbors=5)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "rf_undersample": ImbPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", common_scaler),
                ("sampler", RandomUnderSampler(random_state=random_state)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        max_depth=None,
                        min_samples_leaf=1,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "balanced_rf": ImbPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", common_scaler),
                (
                    "model",
                    BalancedRandomForestClassifier(
                        n_estimators=350,
                        max_depth=None,
                        sampling_strategy="auto",
                        replacement=False,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    return candidates


def cv_evaluate_model(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pr_aucs, roc_aucs = [], []

    for train_idx, valid_idx in skf.split(X, y):
        X_train_cv, X_valid_cv = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_cv, y_valid_cv = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train_cv, y_train_cv)
        probas = model.predict_proba(X_valid_cv)[:, 1]
        pr_aucs.append(average_precision_score(y_valid_cv, probas))
        roc_aucs.append(roc_auc_score(y_valid_cv, probas))

    return {
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs, ddof=1)) if len(pr_aucs) > 1 else 0.0,
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs, ddof=1)) if len(roc_aucs) > 1 else 0.0,
    }


def expected_cost(y_true, y_pred, fp_cost: float = FP_COST, fn_cost: float = FN_COST) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return float(fp * fp_cost + fn * fn_cost)


def tune_threshold(y_true, y_prob, fp_cost: float = FP_COST, fn_cost: float = FN_COST):
    thresholds = np.unique(np.clip(np.round(np.linspace(0.01, 0.99, 99), 4), 0.0, 1.0))
    best = {
        "threshold": 0.5,
        "cost": float("inf"),
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "balanced_accuracy": 0.0,
    }

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        cost = expected_cost(y_true, y_pred, fp_cost=fp_cost, fn_cost=fn_cost)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if (cost < best["cost"]) or (math.isclose(cost, best["cost"]) and f1 > best["f1"]):
            best.update(
                {
                    "threshold": float(thr),
                    "cost": float(cost),
                    "f1": float(f1),
                    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                }
            )
    return best


def plot_curves(y_true, y_prob, outdir: Path, prefix: str):
    outdir.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({prefix}) | AUC-PR = {pr_auc:.4f}")
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_pr_curve.png", dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({prefix}) | AUC-ROC = {roc_auc:.4f}")
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_roc_curve.png", dpi=160)
    plt.close()


def train_and_select_model(X_train, y_train, X_val, y_val, output_dir: Path) -> Tuple[str, object, ModelResult, dict]:
    candidates = build_candidates()

    cv_rows = []
    val_rows = []
    best_name = None
    best_model = None
    best_result = None
    best_score = (-1.0, -1.0)  # (val_pr_auc, cv_pr_auc_mean)

    for name, model in candidates.items():
        cv_metrics = cv_evaluate_model(model, X_train, y_train, n_splits=5)

        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        val_pr_auc = average_precision_score(y_val, val_prob)
        val_roc_auc = roc_auc_score(y_val, val_prob)
        tuned = tune_threshold(y_val, val_prob)

        result = ModelResult(
            name=name,
            cv_pr_auc_mean=cv_metrics["pr_auc_mean"],
            cv_pr_auc_std=cv_metrics["pr_auc_std"],
            cv_roc_auc_mean=cv_metrics["roc_auc_mean"],
            cv_roc_auc_std=cv_metrics["roc_auc_std"],
            val_pr_auc=float(val_pr_auc),
            val_roc_auc=float(val_roc_auc),
            best_threshold=tuned["threshold"],
            val_cost=tuned["cost"],
            val_f1=tuned["f1"],
            val_recall=tuned["recall"],
            val_precision=tuned["precision"],
        )

        cv_rows.append(
            {
                "model": name,
                "cv_pr_auc_mean": cv_metrics["pr_auc_mean"],
                "cv_pr_auc_std": cv_metrics["pr_auc_std"],
                "cv_roc_auc_mean": cv_metrics["roc_auc_mean"],
                "cv_roc_auc_std": cv_metrics["roc_auc_std"],
            }
        )
        val_rows.append(asdict(result))

        score = (result.val_pr_auc, result.cv_pr_auc_mean)
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model
            best_result = result

    cv_df = pd.DataFrame(cv_rows).sort_values(by="cv_pr_auc_mean", ascending=False)
    val_df = pd.DataFrame(val_rows).sort_values(by="val_pr_auc", ascending=False)

    cv_df.to_csv(output_dir / "cv_results.csv", index=False)
    val_df.to_csv(output_dir / "validation_results.csv", index=False)

    return best_name, best_model, best_result, {"cv": cv_df, "val": val_df}


def evaluate_on_test(best_model, threshold: float, X_test, y_test, output_dir: Path) -> dict:
    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    metrics = {
        "test_pr_auc": float(average_precision_score(y_test, test_prob)),
        "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
        "test_cost": float(expected_cost(y_test, test_pred)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist(),
    }

    report = classification_report(y_test, test_pred, digits=4, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_curves(y_test, test_prob, output_dir, prefix="test")
    return metrics


def save_model_artifacts(model, output_dir: Path, metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "fraud_detection_model.joblib")
    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def write_model_card(output_dir: Path, best_name: str, selected_result: ModelResult, metrics: dict):
    md = f"""# Credit Card Fraud Detection Model Card

## Overview
A fraud detection classifier trained on the Kaggle credit card fraud dataset with:
- feature engineering on `Time` and `Amount`
- imbalance handling through resampling
- threshold tuning using a cost-sensitive objective

## Selected model
- **Model:** {best_name}
- **Validation PR-AUC:** {selected_result.val_pr_auc:.4f}
- **Validation ROC-AUC:** {selected_result.val_roc_auc:.4f}
- **Chosen threshold:** {selected_result.best_threshold:.2f}
- **Validation cost:** {selected_result.val_cost:.2f}

## Test metrics
- **PR-AUC:** {metrics["test_pr_auc"]:.4f}
- **ROC-AUC:** {metrics["test_roc_auc"]:.4f}
- **Precision:** {metrics["test_precision"]:.4f}
- **Recall:** {metrics["test_recall"]:.4f}
- **F1:** {metrics["test_f1"]:.4f}
- **Balanced accuracy:** {metrics["test_balanced_accuracy"]:.4f}
- **Cost:** {metrics["test_cost"]:.2f}

## Intended use
Transaction risk scoring and research prototypes. This model is not a substitute for a production fraud platform.

## Monitoring checklist
- Track PR-AUC and recall over time
- Monitor data drift for `Amount`, `Time`, and PCA features
- Recalibrate threshold when fraud prevalence changes
- Review false negatives with priority
"""
    (output_dir / "MODEL_CARD.md").write_text(md, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train a credit card fraud detection model.")
    parser.add_argument("--data", type=Path, required=True, help="Path to creditcard.csv")
    parser.add_argument("--output", type=Path, default=Path("artifacts"), help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split fraction of the remaining train data")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    X, y = make_xy(df)

    # Stratified train / val / test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y_train_val,
    )

    best_name, best_model, selected_result, result_tables = train_and_select_model(
        X_train, y_train, X_val, y_val, output_dir
    )

    # Refit on train + val using the selected model type and selected threshold
    candidate_model = build_candidates(random_state=args.random_state)[best_name]
    X_fit = pd.concat([X_train, X_val], axis=0)
    y_fit = pd.concat([y_train, y_val], axis=0)
    candidate_model.fit(X_fit, y_fit)

    # Refresh threshold using validation predictions from the refit model
    val_prob_refit = candidate_model.predict_proba(X_val)[:, 1]
    tuned = tune_threshold(y_val, val_prob_refit)
    threshold = tuned["threshold"]

    test_metrics = evaluate_on_test(candidate_model, threshold, X_test, y_test, output_dir)

    metadata = {
        "selected_model": best_name,
        "selected_threshold": threshold,
        "feature_count": int(X.shape[1]),
        "train_size": int(len(X_train)),
        "validation_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "fp_cost": FP_COST,
        "fn_cost": FN_COST,
        "cv_results_file": "cv_results.csv",
        "validation_results_file": "validation_results.csv",
        "test_metrics_file": "test_metrics.json",
    }
    save_model_artifacts(candidate_model, output_dir, metadata)
    write_model_card(output_dir, best_name, selected_result, test_metrics)

    print("\n=== Training complete ===")
    print(f"Selected model: {best_name}")
    print(f"Chosen threshold: {threshold:.2f}")
    print("\nValidation summary:")
    print(selected_result)
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
