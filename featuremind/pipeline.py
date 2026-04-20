"""
pipeline.py — Production Pipeline: Save, Load, Predict
=========================================================
featuremind v3.1.0

The core production feature missing from v1.1.
Companies need the same preprocessing applied at prediction time as at training time.
This module provides exactly that.

Usage:
    # TRAIN + SAVE
    import featuremind as fm
    pipeline = fm.train("data.csv", target="Churn")
    pipeline.save("churn_pipeline")

    # LOAD + PREDICT (in production / API)
    pipeline = fm.load_pipeline("churn_pipeline")
    predictions = pipeline.predict("new_data.csv")
    probabilities = pipeline.predict_proba("new_data.csv")

    # BATCH PREDICT
    import pandas as pd
    df_new = pd.read_csv("new_customers.csv")
    predictions = pipeline.predict_df(df_new)

What gets saved:
    churn_pipeline/
        model.pkl           ← trained best model
        scaler.pkl          ← fitted StandardScaler
        config.json         ← all metadata (features, target, task, thresholds)
        feature_names.json  ← exact feature columns used in training
        best_feature.json   ← best engineered feature formula
"""

import json
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class FeaturemindPipeline:
    """
    A production-ready, saveable/loadable ML pipeline.
    Captures everything needed to reproduce predictions on new data.
    """

    def __init__(self):
        self.model          = None
        self.scaler         = StandardScaler()
        self.target         = None
        self.task           = None
        self.feature_names  = []
        self.best_feature   = None   # dict: {name, formula, impact}
        self.opt_threshold  = 0.5
        self.model_name     = None
        self.base_score     = None
        self.version        = "3.1.0"
        self.trained_at     = None
        self.training_rows  = None
        self.col_types      = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, target: str, model, scaler: StandardScaler,
            feature_names: list, task: str, opt_threshold: float,
            model_name: str, base_score: float, best_feature: dict = None,
            col_types: dict = None):
        """Store all training artifacts. Called internally by fm.train()."""
        self.model          = model
        self.scaler         = scaler
        self.target         = target
        self.task           = task
        self.feature_names  = feature_names
        self.best_feature   = best_feature
        self.opt_threshold  = opt_threshold
        self.model_name     = model_name
        self.base_score     = base_score
        self.trained_at     = datetime.now().isoformat()
        self.training_rows  = len(df)
        self.col_types      = col_types or {}
        return self

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply the exact same preprocessing used during training.
        Handles: missing columns (filled with 0), extra columns (dropped),
                 best engineered feature, one-hot encoding alignment.
        """
        df = df.copy()

        # Apply best engineered feature if trained with one
        if self.best_feature and self.best_feature.get("formula"):
            try:
                df["best_feature"] = eval(
                    self.best_feature["formula"],
                    {"df": df, "np": np, "pd": pd}
                )
                df["best_feature"] = (
                    pd.to_numeric(df["best_feature"], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                )
            except Exception:
                df["best_feature"] = 0

        # Drop target if present
        if self.target in df.columns:
            df = df.drop(columns=[self.target])

        # One-hot encode categorical columns
        cat_cols = [c for c in df.columns
                    if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # Align to training features
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0                           # missing → zero

        df = df[self.feature_names].fillna(0)         # exact training column order
        return self.scaler.transform(df)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, input_data) -> np.ndarray:
        """
        Predict from a CSV file path or a DataFrame.

        Returns:
            np.ndarray of predicted labels (classification) or values (regression)
        """
        if self.model is None:
            raise RuntimeError("Pipeline not trained. Call fm.train() first.")

        df = pd.read_csv(input_data) if isinstance(input_data, str) else input_data.copy()
        X  = self._preprocess(df)

        if self.task == "classification" and self.opt_threshold != 0.5:
            if hasattr(self.model, "predict_proba"):
                try:
                    proba = self.model.predict_proba(X)[:, 1]
                    return (proba >= self.opt_threshold).astype(int)
                except Exception:
                    pass
        preds = self.model.predict(X)
        # v3.1: For regression, clip negative predictions to 0
        # (e.g. area, price, count cannot be negative)
        if self.task == "regression":
            import numpy as np
            preds = np.clip(preds, 0, None)
        return preds

    def predict_proba(self, input_data) -> np.ndarray:
        """
        Return class probabilities (classification only).
        Returns shape (n_samples, n_classes).
        """
        if self.task != "classification":
            raise RuntimeError("predict_proba only available for classification tasks.")
        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError(f"{self.model_name} does not support predict_proba.")

        df = pd.read_csv(input_data) if isinstance(input_data, str) else input_data.copy()
        X  = self._preprocess(df)
        return self.model.predict_proba(X)

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict from DataFrame and return results as a new DataFrame.
        Includes original data + predictions + confidence (if classification).
        """
        df_out = df.copy()
        preds  = self.predict(df)
        # v3.1: Round regression predictions to 2 decimal places for readability
        if self.task == "regression":
            import numpy as np
            preds = np.round(preds, 2)
        df_out["prediction"] = preds

        if self.task == "classification" and hasattr(self.model, "predict_proba"):
            try:
                X      = self._preprocess(df)
                proba  = self.model.predict_proba(X).max(axis=1)
                df_out["confidence"] = (proba * 100).round(1)
            except Exception:
                pass
        return df_out

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> str:
        """
        Save pipeline to a directory.

        Creates:
            {path}/model.pkl
            {path}/scaler.pkl
            {path}/config.json
            {path}/feature_names.json
        """
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)

        with open(os.path.join(path, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        config = {
            "version"       : self.version,
            "target"        : self.target,
            "task"          : self.task,
            "model_name"    : self.model_name,
            "base_score"    : self.base_score,
            "opt_threshold" : self.opt_threshold,
            "trained_at"    : self.trained_at,
            "training_rows" : self.training_rows,
            "best_feature"  : self.best_feature,
            "col_types"     : self.col_types,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        with open(os.path.join(path, "feature_names.json"), "w") as f:
            json.dump(self.feature_names, f, indent=2)

        print(f"\n💾 Pipeline saved to '{path}/'")
        print(f"   Files: model.pkl · scaler.pkl · config.json · feature_names.json")
        print(f"   Model : {self.model_name}  |  Score: {self.base_score:.4f}")
        print(f"   Target: {self.target}  |  Task: {self.task}")
        return path

    @classmethod
    def load(cls, path: str) -> "FeaturemindPipeline":
        """Load a saved pipeline from a directory."""
        p = cls()

        with open(os.path.join(path, "model.pkl"), "rb") as f:
            p.model = pickle.load(f)

        with open(os.path.join(path, "scaler.pkl"), "rb") as f:
            p.scaler = pickle.load(f)

        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)

        with open(os.path.join(path, "feature_names.json")) as f:
            p.feature_names = json.load(f)

        p.version       = config.get("version", "?")
        p.target        = config["target"]
        p.task          = config["task"]
        p.model_name    = config["model_name"]
        p.base_score    = config.get("base_score")
        p.opt_threshold = config.get("opt_threshold", 0.5)
        p.trained_at    = config.get("trained_at")
        p.training_rows = config.get("training_rows")
        p.best_feature  = config.get("best_feature")
        p.col_types     = config.get("col_types", {})

        print(f"\n📂 Pipeline loaded from '{path}/'")
        print(f"   Model : {p.model_name}  |  Score: {p.base_score:.4f}")
        print(f"   Target: {p.target}  |  Task: {p.task}")
        if p.trained_at:
            print(f"   Trained: {p.trained_at[:19]}")
        return p

    def summary(self):
        """Print a summary of the loaded/trained pipeline."""
        print(f"\n{'='*55}")
        print(f"  🧠 featuremind Pipeline v{self.version}")
        print(f"{'='*55}")
        print(f"  Model         : {self.model_name or 'Not trained'}")
        print(f"  Task          : {self.task or '?'}")
        print(f"  Target        : {self.target or '?'}")
        print(f"  CV Score      : {self.base_score:.4f}" if self.base_score else "  CV Score      : N/A")
        if self.base_score and hasattr(self, 'cv_std') and self.cv_std:
            stability = round(self.base_score / max(self.cv_std, 1e-6), 2)
            label = "🟢 Stable" if stability > 10 else "⚠️ Unstable" if stability < 5 else "🟡 Moderate"
            print(f"  Stability     : {stability:.1f}  ({label})")
        print(f"  Threshold     : {self.opt_threshold}")
        print(f"  Features      : {len(self.feature_names)} columns")
        print(f"  Training rows : {self.training_rows:,}" if self.training_rows else "  Training rows : N/A")
        print(f"  Trained at    : {self.trained_at[:19]}" if self.trained_at else "  Trained at    : N/A")
        if self.best_feature:
            print(f"  Best feature  : {self.best_feature.get('name','?')}"
                  f" (+{self.best_feature.get('impact',0):.4f})")
        print(f"{'='*55}")


def load_pipeline(path: str) -> FeaturemindPipeline:
    """Load a saved featuremind pipeline. Shortcut for FeaturemindPipeline.load()."""
    return FeaturemindPipeline.load(path)