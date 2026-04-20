"""
evaluator.py — Enterprise Model Selection & Evaluation
========================================================
featuremind v1.1.0

KEY UPGRADE — Class Imbalance Auto-Fix:
  When minority class < 30%, the system automatically:
  1. Applies SMOTE oversampling (if imbalanced-learn is installed)
  2. Falls back to class_weight='balanced' (always available in sklearn)
  3. Uses Stratified CV to preserve class ratios in every fold
  4. Reports both SMOTE and original class distribution in output

KEY UPGRADE — Optimal Threshold Tuning:
  Default decision boundary is 0.5 — wrong for imbalanced data.
  System searches 20 thresholds (0.1–0.9) and picks the one that
  maximises F1 on the validation set. Prints the optimal threshold.

KEY UPGRADE — Imbalance-aware scoring:
  Uses 'f1_weighted' as the CV metric when imbalance is detected,
  instead of 'accuracy' (which is misleading on skewed targets).

Models: LogisticRegression · RandomForest · GradientBoosting ·
        XGBoost · LightGBM · CatBoost (all optional, graceful fallback)
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    classification_report, f1_score, mean_absolute_error, r2_score,
)
from sklearn.model_selection import (
    KFold, RepeatedKFold, RepeatedStratifiedKFold, RandomizedSearchCV, StratifiedKFold,
    cross_val_score, train_test_split,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── v3.1: Leakage guard + reliability ────────────────────────────────────────
try:
    from .leakage_guard import run_full_leakage_guard, check_score_reliability
    _HAS_LEAKAGE_GUARD = True
except Exception:
    _HAS_LEAKAGE_GUARD = False

# ── Optional advanced libraries ───────────────────────────────────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False

try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except ImportError:
    _HAS_SMOTE = False

# ── Thresholds ────────────────────────────────────────────────────────────────
CV_MAX_FEATURES   = 200
CV_MIN_ROWS       = 200
TUNE_MAX_ROWS     = 3000
IMBALANCE_THRESH  = 0.30   # minority class < 30% → apply imbalance handling


# ══════════════════════════════════════════════════════════════════════════════
# Model registry
# ══════════════════════════════════════════════════════════════════════════════

def _get_models(task: str, use_balanced: bool = False) -> dict:
    """Return all models for the given task. use_balanced adds class_weight."""
    clf = task == "classification"
    cw  = "balanced" if (clf and use_balanced) else None
    models = {}

    if clf:
        models["LogisticRegression"] = LogisticRegression(
            max_iter=1000, random_state=42, class_weight=cw or "balanced")
        models["RandomForest"]       = RandomForestClassifier(
            n_estimators=50, random_state=42, class_weight=cw or "balanced")
        models["GradientBoosting"]   = GradientBoostingClassifier(
            n_estimators=50, random_state=42)
        if _HAS_XGB:
            models["XGBoost"] = XGBClassifier(
                n_estimators=50, random_state=42,
                eval_metric="logloss", verbosity=0)
        if _HAS_LGBM:
            models["LightGBM"] = LGBMClassifier(
                n_estimators=50, random_state=42,
                verbose=-1, class_weight=cw or "balanced")
        if _HAS_CAT:
            models["CatBoost"] = CatBoostClassifier(
                iterations=50, random_state=42, verbose=0,
                auto_class_weights="Balanced")
    else:
        models["Ridge"]            = Ridge()
        models["RandomForest"]     = RandomForestRegressor(n_estimators=50, random_state=42)
        models["GradientBoosting"] = GradientBoostingRegressor(n_estimators=50, random_state=42)
        if _HAS_XGB:
            models["XGBoost"] = XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
        if _HAS_LGBM:
            models["LightGBM"] = LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
        if _HAS_CAT:
            models["CatBoost"] = CatBoostRegressor(iterations=50, random_state=42, verbose=0)

    return models


# ══════════════════════════════════════════════════════════════════════════════
# Class imbalance handler  ← NEW in v1.1
# ══════════════════════════════════════════════════════════════════════════════

def _handle_imbalance(X: np.ndarray, y: pd.Series) -> tuple:
    """
    Detect and handle class imbalance.

    Strategy:
      1. If SMOTE available and minority ≥ 6 samples → apply SMOTE
      2. Otherwise → return original data (class_weight handles it in model)

    Returns (X_resampled, y_resampled, method_used, imbalance_info)
    """
    counts    = y.value_counts()
    minority  = counts.min()
    majority  = counts.max()
    min_ratio = minority / max(len(y), 1)
    is_imbalanced = min_ratio < IMBALANCE_THRESH

    info = {
        "is_imbalanced"   : is_imbalanced,
        "minority_ratio"  : round(min_ratio * 100, 1),
        "minority_count"  : int(minority),
        "majority_count"  : int(majority),
        "method"          : "none",
        "smote_available" : _HAS_SMOTE,
    }

    if not is_imbalanced:
        return X, y, info

    # Try SMOTE
    if _HAS_SMOTE and minority >= 6:
        try:
            k = min(5, minority - 1)
            sm = SMOTE(random_state=42, k_neighbors=k)
            X_res, y_res = sm.fit_resample(X, y)
            info["method"]           = "SMOTE"
            info["resampled_size"] = len(X_res)
            info["resampled_rows"] = len(X_res)
            print(f"⚖️   SMOTE applied: {len(y)} → {len(y_res)} rows "
                  f"(minority was {min_ratio*100:.1f}%)")
            return X_res, pd.Series(y_res), info
        except Exception as e:
            print(f"⚖️   SMOTE failed ({e}) — using class_weight='balanced' instead.")

    # Fallback: class_weight (handled in model registry)
    info["method"] = "class_weight"
    print(f"⚖️   Class imbalance detected ({min_ratio*100:.1f}% minority) "
          f"— using class_weight='balanced'")
    return X, y, info


# ══════════════════════════════════════════════════════════════════════════════
# Optimal threshold tuning  ← NEW in v1.1
# ══════════════════════════════════════════════════════════════════════════════

def _find_optimal_threshold(model, X_val: np.ndarray, y_val: pd.Series) -> tuple:
    """
    Search thresholds 0.10–0.90 for best F1 on validation set.
    Returns (optimal_threshold, f1_at_threshold, all_threshold_scores).
    Only applies to binary classification with predict_proba support.
    """
    if not hasattr(model, "predict_proba"):
        return 0.5, None, {}

    if len(y_val.unique()) != 2:
        return 0.5, None, {}

    try:
        proba = model.predict_proba(X_val)[:, 1]
    except Exception:
        return 0.5, None, {}

    thresholds = np.linspace(0.10, 0.90, 17)
    scores     = {}
    best_t, best_f1 = 0.5, -1

    for t in thresholds:
        preds = (proba >= t).astype(int)
        try:
            f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            scores[round(t, 2)] = round(f1, 4)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        except Exception:
            pass

    return round(best_t, 2), round(best_f1, 4), scores


# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameter spaces
# ══════════════════════════════════════════════════════════════════════════════

_PARAM_SPACES = {
    "RandomForest": {
        "n_estimators"     : [50, 100, 200],
        "max_depth"        : [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features"     : ["sqrt", "log2", 0.5],
    },
    "GradientBoosting": {
        "n_estimators" : [50, 100, 150],
        "max_depth"    : [3, 5, 7],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "subsample"    : [0.7, 0.8, 1.0],
    },
    "XGBoost": {
        "n_estimators"    : [50, 100, 150],
        "max_depth"       : [3, 5, 6, 8],
        "learning_rate"   : [0.03, 0.05, 0.1, 0.2],
        "subsample"       : [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators"     : [50, 100, 150],
        "max_depth"        : [3, 5, -1],
        "learning_rate"    : [0.03, 0.05, 0.1, 0.2],
        "num_leaves"       : [31, 50, 100],
        "min_child_samples": [10, 20, 50],
    },
    "CatBoost": {
        "iterations"   : [50, 100],
        "depth"        : [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "l2_leaf_reg"  : [1, 3, 5],
    },
}


def _tune(model_name, model, X_tr, y_tr, task, scoring):
    space = _PARAM_SPACES.get(model_name)
    if not space:
        return model, {}
    cv = (StratifiedKFold(3, shuffle=True, random_state=42)
          if task == "classification" else
          KFold(3, shuffle=True, random_state=42))
    try:
        search = RandomizedSearchCV(
            model, space, n_iter=10, cv=cv, scoring=scoring,
            random_state=42, n_jobs=-1, refit=True)
        search.fit(X_tr, y_tr)
        return search.best_estimator_, search.best_params_
    except Exception:
        return model, {}


# ══════════════════════════════════════════════════════════════════════════════
# Cross-validation
# ══════════════════════════════════════════════════════════════════════════════

def _cross_validate(X, y, task, scoring, use_balanced):
    models  = _get_models(task, use_balanced)
    # v3.1: Use Repeated CV for regression — averages out high variance
    n_samples = len(y)
    if task == "classification":
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
    elif n_samples < 500:
        # Small regression: use repeated 3-fold for stability
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=42)
    else:
        cv = KFold(5, shuffle=True, random_state=42)

    best_name, best_mean, best_std = "", -np.inf, 0.0
    all_scores = {}

    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            mean, std = float(scores.mean()), float(scores.std())
            if np.isnan(mean) or np.isnan(std):   # ← FIX: skip NaN scores
                continue
            all_scores[name] = round(mean, 4)
            if mean > best_mean:
                best_mean, best_std, best_name = mean, std, name
        except Exception:
            pass

    if not best_name:
        raise RuntimeError("All models failed during CV.")

    best_model = _get_models(task, use_balanced)[best_name]
    best_model.fit(X, y)
    return best_model, round(best_mean, 4), round(best_std, 4), best_name, all_scores


# ══════════════════════════════════════════════════════════════════════════════
# Detailed metrics
# ══════════════════════════════════════════════════════════════════════════════

def _detailed_metrics(model, X_te, y_te, task, opt_threshold=0.5) -> dict:
    m = {}
    try:
        y_pred = model.predict(X_te)
    except Exception:
        return m

    # Apply optimal threshold if different from default
    if task == "classification" and opt_threshold != 0.5 and hasattr(model, "predict_proba"):
        try:
            proba  = model.predict_proba(X_te)[:, 1]
            y_pred = (proba >= opt_threshold).astype(int)
        except Exception:
            pass

    if task == "classification":
        rep = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
        m["per_class"]    = {
            k: {mt: round(v, 3) for mt, v in v.items()
                if mt in ("precision", "recall", "f1-score")}
            for k, v in rep.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        }
        m["macro_f1"]    = round(rep["macro avg"]["f1-score"], 4)
        m["weighted_f1"] = round(rep["weighted avg"]["f1-score"], 4)
        m["optimal_threshold"] = opt_threshold

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_te).max(axis=1)
                m["avg_confidence"]     = round(float(proba.mean()), 4)
                m["low_confidence_pct"] = round(float((proba < 0.60).mean() * 100), 1)
            except Exception:
                pass
    else:
        yp = np.array(y_pred); yt = np.array(y_te)
        m["r2"]   = round(float(r2_score(yt, yp)), 4)
        m["mae"]  = round(float(mean_absolute_error(yt, yp)), 4)
        m["rmse"] = round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4)
        if hasattr(model, "estimators_"):
            try:
                preds = np.array([e.predict(X_te) for e in model.estimators_])
                m["pred_interval_95"] = round(float(np.percentile(np.std(preds, axis=0), 95)), 4)
            except Exception:
                pass
    return m


# ══════════════════════════════════════════════════════════════════════════════
# SHAP
# ══════════════════════════════════════════════════════════════════════════════

def _shap_importance(model, X_sample, feature_names):
    if not _HAS_SHAP:
        return {}
    try:
        sample      = X_sample[:min(200, len(X_sample))]
        explainer   = _shap.Explainer(model, sample)
        shap_values = explainer(sample)
        vals = np.abs(shap_values.values)
        if vals.ndim == 3:
            vals = vals.mean(axis=2)
        mean_abs = vals.mean(axis=0)
        imp = dict(zip(feature_names, mean_abs))
        return dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Bias check
# ══════════════════════════════════════════════════════════════════════════════

_SENSITIVE = {"gender", "sex", "race", "ethnicity", "age",
              "religion", "nationality", "disability", "marital"}


def _bias_check(df, target, features):
    warns = []
    sens  = [c for c in df.columns
             if any(kw in c.lower() for kw in _SENSITIVE) and c != target]
    for sc in sens:
        try:
            corr = abs(df[features].corrwith(df[sc]).max())
            if corr > 0.30:
                warns.append(
                    f"Bias risk: '{sc}' correlates with model features "
                    f"(max r={corr:.2f}). Review for demographic fairness.")
        except Exception:
            pass
    return warns


# ══════════════════════════════════════════════════════════════════════════════
# Confidence level
# ══════════════════════════════════════════════════════════════════════════════

def _confidence(cv_mean, cv_std, n, task, forced_confidence=None):
    # v3.1: forced_confidence from reliability check overrides everything
    if forced_confidence is not None:
        reason = f"CV score {cv_mean:.3f} ±{cv_std:.3f} — OVERRIDDEN by reliability check"
        return forced_confidence, reason

    reasons = []
    if n < 200:   reasons.append(f"small dataset ({n} rows)")
    elif n < 500: reasons.append(f"limited dataset ({n} rows)")
    if cv_std > 0.05: reasons.append(f"unstable CV (±{cv_std:.3f})")

    if task == "classification":
        level = ("High ✅"   if cv_mean >= 0.85 and cv_std <= 0.03 else
                 "Medium ⚠️" if cv_mean >= 0.65 or cv_std <= 0.05 else
                 "Low ❌")
    else:
        if cv_mean >= 0.70 and cv_std <= 0.05: level = "High ✅"
        elif cv_mean >= 0.40:                   level = "Medium ⚠️"
        else:
            level = "Low ❌"
            reasons.append("weak signal in data")

    reason = f"CV score {cv_mean:.3f} ±{cv_std:.3f}"
    if reasons:
        reason += f" — {', '.join(reasons)}"
    return level, reason


# ══════════════════════════════════════════════════════════════════════════════
# Feature impact
# ══════════════════════════════════════════════════════════════════════════════

def _eval_cv(df, target, f, scaler, y, base, task, scoring):
    dc = df.copy()
    try:
        dc["new_feat"] = (
            pd.Series(eval(f["formula"], {"df": dc, "np": np, "pd": pd}),
                      index=dc.index)
            .replace([np.inf, -np.inf], np.nan).fillna(0)
        )
    except Exception:
        return None
    X_new = dc.drop(columns=[target]).fillna(0)
    try:
        Xs = scaler.fit_transform(X_new)
    except Exception:
        return None
    cv = (StratifiedKFold(5, shuffle=True, random_state=42)
          if task == "classification" else
          KFold(5, shuffle=True, random_state=42))
    try:
        m = (RandomForestClassifier(50, random_state=42, class_weight="balanced")
             if task == "classification" else
             RandomForestRegressor(50, random_state=42))
        scores  = cross_val_score(m, Xs, y, cv=cv, scoring=scoring, n_jobs=-1)
        new     = round(float(scores.mean()), 4)
    except Exception:
        return None
    f["impact"]    = round(new - base, 4)
    f["new_score"] = new
    return f


def _eval_single(df, target, f, scaler, y, base, task):
    dc = df.copy()
    try:
        dc["new_feat"] = (
            pd.Series(eval(f["formula"], {"df": dc, "np": np, "pd": pd}),
                      index=dc.index)
            .replace([np.inf, -np.inf], np.nan).fillna(0)
        )
    except Exception:
        return None
    X_new = dc.drop(columns=[target]).fillna(0)
    try:
        Xs = scaler.fit_transform(X_new)
    except Exception:
        return None
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)
    try:
        m = (Ridge() if task == "regression" else
             LogisticRegression(max_iter=500, random_state=42, class_weight="balanced"))
        m.fit(Xtr, ytr)
        new = round(float(
            m.score(Xte, yte) if task == "classification"
            else r2_score(yte, m.predict(Xte))), 4)
    except Exception:
        return None
    f["impact"]    = round(new - base, 4)
    f["new_score"] = new
    return f


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_features(df: pd.DataFrame, target: str, features: list):
    """
    Full v1.1 evaluation pipeline with imbalance handling + threshold tuning.

    Returns:
        base_score   : CV score
        model_name   : Best model name
        final_feats  : Top-5 features sorted by impact
        eval_results : Full results dict
    """
    X = df.drop(columns=[target]).fillna(0)
    y = df[target]

    if len(X) < 10:
        return 0.0, None, [], {}

    task    = "classification" if y.nunique() <= 15 else "regression"
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    f_names = list(X.columns)
    n_feat  = X_sc.shape[1]
    use_cv  = n_feat <= CV_MAX_FEATURES and len(X) >= CV_MIN_ROWS

    # ── Imbalance detection & handling ────────────────────────────────────────
    imbalance_info = {"is_imbalanced": False, "method": "none"}
    X_train = X_sc
    y_train = y
    use_balanced = False

    if task == "classification":
        X_train, y_train, imbalance_info = _handle_imbalance(X_sc, y)
        use_balanced = imbalance_info["is_imbalanced"]

    # ── Choose scoring metric ─────────────────────────────────────────────────
    if task == "classification":
        scoring = "f1_weighted" if use_balanced else "accuracy"
    else:
        scoring = "r2"

    # ── Baseline ──────────────────────────────────────────────────────────────
    if use_cv:
        best_model, cv_mean, cv_std, model_name, all_scores = _cross_validate(
            X_train, y_train, task, scoring, use_balanced)
        base = cv_mean
        metric_label = "F1-weighted" if use_balanced else "Accuracy"
        print(f"\n📐 Baseline — {model_name}: {base:.4f} "
              f"(±{cv_std:.4f}, 5-fold CV, metric={metric_label})")
    else:
        Xtr, Xte, ytr, yte = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        best_model, cv_mean, cv_std, model_name = None, -np.inf, 0.0, ""
        all_scores = {}
        for name, m in _get_models(task, use_balanced).items():
            try:
                m.fit(Xtr, ytr)
                s = float(m.score(Xte, yte) if task == "classification"
                          else r2_score(yte, m.predict(Xte)))
                all_scores[name] = round(s, 4)
                if s > cv_mean:
                    cv_mean, model_name, best_model = s, name, m
            except Exception:
                pass
        cv_mean = round(cv_mean, 4)
        base    = cv_mean
        cv_std  = 0.0
        print(f"\n📐 Baseline — {model_name}: {base:.4f} (single split, {n_feat} features)")

    # ── Hyperparameter tuning ─────────────────────────────────────────────────
    best_params = {}
    if use_cv and len(X) <= TUNE_MAX_ROWS and best_model is not None:
        Xtr, Xte, ytr, yte = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        try:
            tuned, best_params = _tune(model_name, best_model, Xtr, ytr, task, scoring)
            if best_params:
                best_model = tuned
                print(f"🔧  Tuned {model_name}: {best_params}")
        except Exception:
            pass

    # ── Detailed metrics + optimal threshold ──────────────────────────────────
    Xtr, Xte, ytr, yte = train_test_split(X_sc, y, test_size=0.2, random_state=42)
    opt_threshold, opt_f1, threshold_scores = 0.5, None, {}
    det_metrics = {}
    if best_model is not None:
        try:
            best_model.fit(Xtr, ytr)
            # Find optimal threshold (classification only)
            if task == "classification":
                opt_threshold, opt_f1, threshold_scores = _find_optimal_threshold(
                    best_model, Xte, yte)
                if opt_threshold != 0.5:
                    print(f"🎯  Optimal threshold: {opt_threshold} "
                          f"(F1={opt_f1:.4f} vs default 0.5)")
            det_metrics = _detailed_metrics(best_model, Xte, yte, task, opt_threshold)
        except Exception:
            pass

    # ── SHAP ──────────────────────────────────────────────────────────────────
    shap_imp = {}
    if _HAS_SHAP and best_model is not None:
        try:
            shap_imp = _shap_importance(best_model, Xtr, f_names)
            if shap_imp:
                print(f"🔍  SHAP values computed ({len(shap_imp)} features).")
        except Exception:
            pass

    # ── Bias check ────────────────────────────────────────────────────────────
    bias_warns = _bias_check(df, target, f_names)
    for w in bias_warns:
        print(f"⚠️  {w}")

    # ── Confidence ────────────────────────────────────────────────────────────
    # ── v3.1: Score reliability check (fixes confidence contradiction) ─────────
    reliability       = {}
    forced_confidence = None
    if _HAS_LEAKAGE_GUARD:
        try:
            reliability = check_score_reliability(cv_mean, cv_std, model_name, task)
            forced_confidence = reliability.get("forced_confidence")
            if reliability.get("messages"):
                print(f"\n{'='*60}")
                print("  🔍 RELIABILITY CHECK")
                print(f"{'='*60}")
                for msg in reliability["messages"]:
                    print(f"  {msg}")
                print(f"{'='*60}")
        except Exception:
            pass

    conf_label, conf_reason = _confidence(cv_mean, cv_std, len(X), task,
                                          forced_confidence=forced_confidence)

    # ── v3.1: Leakage guard runs BEFORE feature evaluation ──────────────────
    leakage_report = {"blocked": [], "n_blocked": 0, "dataset_warnings": [], "n_passed": len(features)}
    if _HAS_LEAKAGE_GUARD:
        try:
            clean_features, blocked_features, dataset_warnings = run_full_leakage_guard(
                df, features, target)
            leakage_report = {
                "blocked"         : blocked_features,
                "n_blocked"       : len(blocked_features),
                "dataset_warnings": dataset_warnings,
                "n_passed"        : len(clean_features),
            }
        except Exception:
            clean_features = features
    else:
        clean_features = features

    # ── Feature impact (use top-20 by prescore for speed) ────────────────────
    results    = []
    candidates = clean_features[:20]   # leakage-free candidates first
    fn = _eval_cv if use_cv else _eval_single

    for f in candidates:
        kwargs = {"scoring": scoring} if use_cv else {}
        try:
            updated = fn(df, target, f, scaler, y, base, task, **kwargs)
        except TypeError:
            updated = _eval_single(df, target, f, scaler, y, base, task)
        if updated is not None:
            results.append(updated)

    results = sorted(results, key=lambda x: x["impact"], reverse=True)
    for i, f in enumerate(results):
        f["rank"] = f"#{i + 1}"

    # v3.1: Separate positive and negative impact features
    positive_features = [f for f in results if f.get("impact", 0) > 0]
    negative_features = [f for f in results if f.get("impact", 0) <= 0]
    if not positive_features and results:
        print(f"\n✅ Model already captures most signal — "
              f"engineered features had minimal impact "
              f"(best was {results[0].get('impact',0):+.4f}).")

    eval_results = {
        "task"              : task,
        "cv_mean"           : cv_mean,
        "cv_std"            : cv_std,
        # v3.1: correct metric label (R² for regression, not Accuracy)
        "metric_label"      : ("F1-weighted" if scoring == "f1_weighted"
                               else "Accuracy" if task == "classification"
                               else "R²"),
        "model_name"        : model_name,
        "all_model_scores"  : all_scores,
        "detailed_metrics"  : det_metrics,
        "shap_importance"   : shap_imp,
        "bias_warnings"     : bias_warns,
        "best_params"       : best_params,
        "confidence_label"  : conf_label,
        "confidence_reason" : conf_reason,
        "forced_confidence" : forced_confidence,
        "n_rows"            : len(X),
        "n_features"        : n_feat,
        "use_cv"            : use_cv,
        "imbalance_info"    : imbalance_info,
        "opt_threshold"     : opt_threshold,
        "opt_f1"            : opt_f1,
        "threshold_scores"  : threshold_scores,
        "scoring_metric"    : scoring,
        "reliability"       : reliability,
        "leakage_report"    : leakage_report,
        "all_features"      : results,
        "negative_features" : negative_features if 'negative_features' in dir() else [],
        # v3.1: stability score = mean/std (higher = more stable)
        "stability_score"   : round(cv_mean / max(cv_std, 1e-6), 2) if cv_std > 0 else 99.0,
        "libs": {
            "xgboost"  : _HAS_XGB,
            "lightgbm" : _HAS_LGBM,
            "catboost" : _HAS_CAT,
            "shap"     : _HAS_SHAP,
            "smote"    : _HAS_SMOTE,
        },
    }

    # v3.1: return only positive-impact features in final list
    final_features = positive_features[:5] if 'positive_features' in dir() else results[:5]
    return base, model_name, final_features, eval_results