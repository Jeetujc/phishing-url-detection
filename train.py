"""
train.py
--------
Automated ML training pipeline for the phishing URL detection model.

Steps
-----
1. Load dataset (phishing.csv or a custom CSV)
2. Feature engineering / validation
3. Train multiple classifiers with cross-validation
4. Hyperparameter tuning for the best model
5. Evaluate on a held-out test set (Accuracy, Precision, Recall, F1, ROC-AUC)
6. Save the best model to pickle/model.pkl (compatible with app.py)
7. Save a JSON performance report

Usage
-----
    python train.py                          # uses phishing.csv
    python train.py --data my_dataset.csv   # custom dataset
    python train.py --data phishing.csv --no-tune   # skip hyperparameter search
"""

import argparse
import json
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.preprocessing import LabelEncoder

from model_utils import LabelDecodingWrapper

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional heavy dependencies – imported lazily
# ---------------------------------------------------------------------------
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "UsingIP", "LongURL", "ShortURL", "Symbol@", "Redirecting//",
    "PrefixSuffix-", "SubDomains", "HTTPS", "DomainRegLen", "Favicon",
    "NonStdPort", "HTTPSDomainURL", "RequestURL", "AnchorURL",
    "LinksInScriptTags", "ServerFormHandler", "InfoEmail", "AbnormalURL",
    "WebsiteForwarding", "StatusBarCust", "DisableRightClick",
    "UsingPopupWindow", "IframeRedirection", "AgeofDomain", "DNSRecording",
    "WebsiteTraffic", "PageRank", "GoogleIndex", "LinksPointingToPage",
    "StatsReport",
]
TARGET_COLUMN = "class"
PICKLE_DIR = Path("pickle")
REPORT_PATH = Path("training_report.json")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate the feature CSV."""
    df = pd.read_csv(path, index_col=0)

    # Drop any unnamed index columns that pandas sometimes adds
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    print(f"  Loaded {len(df):,} rows from '{path}'")
    print(f"  Class distribution:\n{df[TARGET_COLUMN].value_counts().to_string()}")
    return df


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_models():
    """Return a dict of {name: estimator}."""
    models = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=2,
            random_state=42, n_jobs=-1,
        ),
    }

    if _HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )

    if _HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=-1,
            random_state=42, n_jobs=-1, verbose=-1,
        )

    return models


# ---------------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------------

PARAM_GRIDS = {
    "GradientBoosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
    },
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6],
    },
    "LightGBM": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
    },
}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test, label_encoder=None) -> dict:
    """Compute a full set of metrics on the test split."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Convert encoded labels back if needed
    if label_encoder is not None:
        y_test_orig = label_encoder.inverse_transform(y_test)
        y_pred_orig = label_encoder.inverse_transform(y_pred)
    else:
        y_test_orig = y_test
        y_pred_orig = y_pred

    return {
        "accuracy":  round(float(accuracy_score(y_test_orig, y_pred_orig)), 4),
        "precision": round(float(precision_score(y_test_orig, y_pred_orig, average="weighted", zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test_orig, y_pred_orig, average="weighted", zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test_orig, y_pred_orig, average="weighted", zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test_orig, y_pred_orig).tolist(),
        "classification_report": classification_report(y_test_orig, y_pred_orig, zero_division=0),
    }


def cross_validate_model(model, X, y, n_folds: int = 5) -> dict:
    """Return mean ± std for accuracy and F1 via stratified k-fold CV."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    # n_jobs=1 avoids forking conflicts with LightGBM/XGBoost internal threading
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    f1_scores  = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=1)
    return {
        "cv_accuracy_mean": round(float(acc_scores.mean()), 4),
        "cv_accuracy_std":  round(float(acc_scores.std()),  4),
        "cv_f1_mean":       round(float(f1_scores.mean()),  4),
        "cv_f1_std":        round(float(f1_scores.std()),   4),
    }


def tune_model(model, param_grid: dict, X_train, y_train, n_folds: int = 3):
    """Run GridSearchCV and return the best estimator."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    grid = GridSearchCV(
        model, param_grid, cv=cv, scoring="accuracy",
        n_jobs=-1, verbose=0, refit=True,
    )
    grid.fit(X_train, y_train)
    print(f"    Best params : {grid.best_params_}")
    print(f"    Best CV acc : {grid.best_score_:.4f}")
    return grid.best_estimator_


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(data_path: str, do_tune: bool = True, n_cv_folds: int = 5):
    print("\n" + "=" * 60)
    print("  Phishing URL Detection – Model Training Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading dataset …")
    df = load_dataset(data_path)

    X = df[FEATURE_COLUMNS].values
    y_raw = df[TARGET_COLUMN].values

    # XGBoost / LightGBM require non-negative integer labels; encode {-1,1}->{0,1}
    le = LabelEncoder()
    y = le.fit_transform(y_raw)          # -1 -> 0, 1 -> 1  (alphabetical sort)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ------------------------------------------------------------------
    # 2. Train & cross-validate all models
    # ------------------------------------------------------------------
    print("\n[2/5] Training and cross-validating models …")
    models = build_models()
    results = {}

    for name, model in models.items():
        print(f"\n  ── {name}")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        cv_metrics = cross_validate_model(model, X_train, y_train, n_folds=n_cv_folds)
        test_metrics = evaluate(model, X_test, y_test, label_encoder=le)

        results[name] = {
            "train_time_s": round(elapsed, 2),
            **cv_metrics,
            **test_metrics,
        }
        print(f"    Train time  : {elapsed:.1f}s")
        print(f"    CV accuracy : {cv_metrics['cv_accuracy_mean']:.4f} ± {cv_metrics['cv_accuracy_std']:.4f}")
        print(f"    Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"    Test F1      : {test_metrics['f1']:.4f}")
        print(f"    ROC-AUC      : {test_metrics['roc_auc']:.4f}")

    # ------------------------------------------------------------------
    # 3. Select best model by test accuracy
    # ------------------------------------------------------------------
    print("\n[3/5] Selecting best model …")
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    print(f"  Best model: {best_name}  (accuracy={results[best_name]['accuracy']:.4f})")

    best_model = models[best_name]

    # ------------------------------------------------------------------
    # 4. Optional hyperparameter tuning
    # ------------------------------------------------------------------
    if do_tune and best_name in PARAM_GRIDS:
        print(f"\n[4/5] Hyperparameter tuning for {best_name} …")
        best_model = tune_model(
            best_model, PARAM_GRIDS[best_name], X_train, y_train,
        )
        # Re-evaluate after tuning
        tuned_metrics = evaluate(best_model, X_test, y_test, label_encoder=le)
        print(f"  Tuned test accuracy : {tuned_metrics['accuracy']:.4f}  "
              f"(was {results[best_name]['accuracy']:.4f})")
        results[best_name].update({f"tuned_{k}": v for k, v in tuned_metrics.items()})
        results[best_name]["tuned"] = True
    else:
        print("\n[4/5] Skipping hyperparameter tuning.")

    # ------------------------------------------------------------------
    # 5. Save model and report
    # ------------------------------------------------------------------
    print("\n[5/5] Saving model and report …")
    PICKLE_DIR.mkdir(exist_ok=True)

    wrapped_model = LabelDecodingWrapper(best_model, le)

    model_path = PICKLE_DIR / "model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(wrapped_model, fh)
    print(f"  Model saved to '{model_path}'")

    report = {
        "best_model": best_name,
        "dataset": data_path,
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "feature_columns": FEATURE_COLUMNS,
        "models": results,
    }
    with open(REPORT_PATH, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"  Report saved to '{REPORT_PATH}'")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    header = f"  {'Model':<20} {'CV Acc':>8} {'Test Acc':>10} {'F1':>8} {'ROC-AUC':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, m in results.items():
        acc_key = "tuned_accuracy" if m.get("tuned") else "accuracy"
        f1_key  = "tuned_f1"       if m.get("tuned") else "f1"
        auc_key = "tuned_roc_auc"  if m.get("tuned") else "roc_auc"
        print(f"  {name:<20} {m['cv_accuracy_mean']:>8.4f} {m[acc_key]:>10.4f} "
              f"{m[f1_key]:>8.4f} {m[auc_key]:>9.4f}"
              + (" ✓ (tuned)" if m.get("tuned") else ""))
    print("=" * 60)
    print(f"\nBest model → '{best_name}' saved to pickle/model.pkl\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train phishing URL detection models.")
    parser.add_argument(
        "--data", default="phishing.csv",
        help="Path to the feature CSV (default: phishing.csv)",
    )
    parser.add_argument(
        "--no-tune", action="store_true",
        help="Skip hyperparameter tuning (faster, but possibly lower accuracy)",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    args = parser.parse_args()

    main(data_path=args.data, do_tune=not args.no_tune, n_cv_folds=args.cv_folds)
