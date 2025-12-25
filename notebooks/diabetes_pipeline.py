
"""
Diabetes Prediction - End-to-End ML Pipeline
Optimized for Google Colab / Local Execution

Usage:
1) Place train.csv and test.csv in the same directory.
2) Run: python diabetes_pipeline.py
3) Output: submission.csv
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def main():
    # =========================
    # Load Data
    # =========================
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    # =========================
    # Split Features & Target
    # =========================
    X = train.drop(columns=["id", "diagnosed_diabetes"])
    y = train["diagnosed_diabetes"]
    X_test = test.drop(columns=["id"])

    # =========================
    # Categorical Features
    # =========================
    cat_features = [
        'gender', 'ethnicity', 'education_level',
        'income_level', 'smoking_status', 'employment_status'
    ]
    cat_features_idx = [X.columns.get_loc(col) for col in cat_features]

    # =========================
    # Cross Validation
    # =========================
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # =========================
    # CatBoost Model
    # =========================
    cat_model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        eval_metric="AUC",
        random_seed=42,
        verbose=100
    )

    cat_auc = []
    for tr, val in skf.split(X, y):
        X_tr, X_val = X.iloc[tr], X.iloc[val]
        y_tr, y_val = y.iloc[tr], y.iloc[val]

        cat_model.fit(X_tr, y_tr, cat_features=cat_features_idx)
        preds = cat_model.predict_proba(X_val)[:, 1]
        cat_auc.append(roc_auc_score(y_val, preds))

    print("CatBoost Mean AUC:", np.mean(cat_auc))

    # =========================
    # LightGBM Model
    # =========================
    lgb_model = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )

    lgb_auc = []
    for tr, val in skf.split(X, y):
        X_tr, X_val = X.iloc[tr], X.iloc[val]
        y_tr, y_val = y.iloc[tr], y.iloc[val]

        lgb_model.fit(X_tr, y_tr)
        preds = lgb_model.predict_proba(X_val)[:, 1]
        lgb_auc.append(roc_auc_score(y_val, preds))

    print("LightGBM Mean AUC:", np.mean(lgb_auc))

    # =========================
    # XGBoost Model
    # =========================
    xgb_model = XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="auc",
        n_jobs=-1,
        random_state=42
    )

    xgb_auc = []
    for tr, val in skf.split(X, y):
        X_tr, X_val = X.iloc[tr], X.iloc[val]
        y_tr, y_val = y.iloc[tr], y.iloc[val]

        xgb_model.fit(X_tr, y_tr)
        preds = xgb_model.predict_proba(X_val)[:, 1]
        xgb_auc.append(roc_auc_score(y_val, preds))

    print("XGBoost Mean AUC:", np.mean(xgb_auc))

    # =========================
    # Train Final Models
    # =========================
    cat_model.fit(X, y, cat_features=cat_features_idx)
    lgb_model.fit(X, y)
    xgb_model.fit(X, y)

    # =========================
    # Ensemble Prediction
    # =========================
    test_preds = (
        0.4 * cat_model.predict_proba(X_test)[:, 1] +
        0.3 * lgb_model.predict_proba(X_test)[:, 1] +
        0.3 * xgb_model.predict_proba(X_test)[:, 1]
    )

    # =========================
    # Save Submission
    # =========================
    submission = pd.DataFrame({
        "id": test["id"],
        "diagnosed_diabetes": test_preds
    })

    submission.to_csv("submission.csv", index=False)
    print("submission.csv created successfully!")


if __name__ == "__main__":
    main()
