"""
importance.py — Feature Importance
=====================================
featuremind v0.5.0

Uses RandomForest to compute feature importances.
SHAP importance (when available) is computed separately in evaluator.py.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def get_feature_importance(df: pd.DataFrame, target: str) -> dict:
    """
    Compute feature importances using RandomForest.

    Returns:
        dict sorted by importance descending: {feature: score}
        Returns {} on failure — never raises.
    """
    try:
        X = df.drop(columns=[target]).fillna(0)
        y = df[target]

        # Remove zero-variance columns
        X = X.dropna(axis=1, how="all")
        X = X.loc[:, X.std() > 0]

        valid = y.notna()
        X, y  = X[valid], y[valid]

        if len(X) < 10:
            raise ValueError(f"Only {len(X)} valid rows — cannot compute importance.")

        model = (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                 if y.nunique() <= 15 else
                 RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        model.fit(X, y)

        imp = dict(zip(X.columns, model.feature_importances_))
        return dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))

    except Exception as e:
        print(f"⚠️  Feature importance failed: {e}")
        return {}