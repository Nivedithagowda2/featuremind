"""
insights.py — Enterprise Insight Engine
=========================================
featuremind v0.5.0

New in v0.5:
  ✅ SHAP-driven top driver insights (when available)
  ✅ Bias & fairness warning integration
  ✅ Hyperparameter tuning result commentary
  ✅ Data drift advisory (based on skewness + outlier report)
  ✅ Multi-step prioritised recommendations
  ✅ Library availability reporting
"""

from __future__ import annotations
import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# Score interpretation
# ═════════════════════════════════════════════════════════════════════════════

def _interpret_score(score: float, task: str) -> str:
    if task == "classification":
        if score >= 0.92:   return f"Excellent ({score:.1%}) — very high predictive accuracy."
        elif score >= 0.82: return f"Good ({score:.1%}) — model performs well on most cases."
        elif score >= 0.70: return f"Moderate ({score:.1%}) — reasonable accuracy but misses patterns."
        elif score >= 0.60: return f"Weak ({score:.1%}) — model struggles; needs more features or data."
        else:               return f"Poor ({score:.1%}) — near random. Check target column quality."
    else:
        if score >= 0.80:   return f"Excellent (R²={score:.3f}) — model explains most variance."
        elif score >= 0.60: return f"Good (R²={score:.3f}) — captures main patterns."
        elif score >= 0.40: return f"Moderate (R²={score:.3f}) — partial signal captured."
        elif score >= 0.20: return f"Weak (R²={score:.3f}) — limited explanatory power."
        else:               return (f"Poor (R²={score:.3f}) — model cannot explain target. "
                                    "Consider a different target column or richer features.")


def _score_recommendations(score: float, task: str, n_rows: int) -> list[str]:
    recs = []
    if task == "classification":
        if score < 0.70:
            recs.append("Score < 70% — try adding domain-specific features or collect more labelled data.")
        if score < 0.60:
            recs.append("Very low accuracy — verify target column labels are correct.")
    else:
        if score < 0.40:
            recs.append("Low R² — add more relevant numeric features or choose a different target.")
        if score < 0.20:
            recs.append("R² near zero — data may be too noisy. Consider domain expert review.")
    if n_rows < 500:
        recs.append(f"Only {n_rows} rows — collect more data. Models generalise poorly below 1,000 rows.")
    elif n_rows < 2000:
        recs.append(f"Moderate dataset ({n_rows} rows) — results are reasonable but more data would help.")
    return recs


# ═════════════════════════════════════════════════════════════════════════════
# Data quality insights
# ═════════════════════════════════════════════════════════════════════════════

def _data_insights(data_quality: dict) -> list[str]:
    insights = []
    dropped   = data_quality.get("dropped_columns", [])
    converted = data_quality.get("converted_columns", [])
    imputed   = data_quality.get("imputed_columns", [])
    dt_ext    = data_quality.get("datetime_extracted", [])
    skew_fixed= data_quality.get("skew_fixed", [])
    outliers  = data_quality.get("outlier_report", {})
    missing   = data_quality.get("missing_pct", 0)
    dupes     = data_quality.get("duplicate_rows", 0)

    if dropped:
        insights.append(f"Removed {len(dropped)} problematic column(s): "
                        + ", ".join(d.split(" ")[0] for d in dropped[:4])
                        + ("..." if len(dropped) > 4 else "."))
    if converted:
        insights.append(f"Auto-parsed {len(converted)} text column(s) to numeric values.")
    if imputed:
        insights.append(f"Applied median/mode imputation to {len(imputed)} column(s) with missing values.")
    if dt_ext:
        insights.append(f"Extracted rich datetime features from: {', '.join(dt_ext)}.")
    if skew_fixed:
        insights.append(f"Auto-fixed skewness in {len(skew_fixed)} column(s) using log1p transform.")
    if missing > 20:
        insights.append(f"High missing data ({missing:.1f}%) — imputation may introduce bias.")
    elif missing > 5:
        insights.append(f"Moderate missing data ({missing:.1f}%) — handled with imputation.")
    if dupes > 0:
        insights.append(f"Removed {dupes} duplicate rows before analysis.")
    if outliers:
        worst = max(outliers.items(), key=lambda x: x[1]["pct"])
        insights.append(
            f"Outliers detected: '{worst[0]}' has {worst[1]['pct']}% extreme values. "
            "Consider capping or removing them for better model stability.")

    for w in data_quality.get("warnings", []):
        insights.append(f"⚠️ {w}")

    return insights


# ═════════════════════════════════════════════════════════════════════════════
# Feature driver insights
# ═════════════════════════════════════════════════════════════════════════════

def _top_driver_insights(importance: dict, shap_importance: dict, target: str) -> list[str]:
    insights = []
    use_shap = bool(shap_importance)
    imp = shap_importance if use_shap else importance
    source = "SHAP" if use_shap else "RandomForest"

    if not imp:
        return ["Feature importance could not be computed."]

    items = list(imp.items())[:3]
    top_name, top_val = items[0]

    if use_shap:
        insights.append(
            f"[SHAP] '{top_name}' has the highest average impact on '{target}' predictions.")
    else:
        insights.append(
            f"[{source}] '{top_name}' is the strongest predictor of '{target}' "
            f"({top_val*100:.1f}% importance).")

    if len(items) >= 2:
        others = " and ".join(f"'{n}'" for n, _ in items[1:3])
        insights.append(f"Other key drivers: {others}.")

    if items[0][1] > 0.40:
        insights.append(
            f"Warning: '{top_name}' dominates at >{items[0][1]*100:.0f}% importance. "
            "Check for data leakage — this feature may be derived from the target.")

    return insights


# ═════════════════════════════════════════════════════════════════════════════
# Feature verdict
# ═════════════════════════════════════════════════════════════════════════════

def _feature_verdict(features: list, base_score: float) -> tuple[str, list[str]]:
    if not features:
        return "No engineered features could be evaluated.", []

    positive = [f for f in features if f.get("impact", 0) > 0]
    if not positive:
        return ("✅ Model already captures most signal — engineered features had minimal impact. "
                "The existing features already capture most predictive signal. "
                "Focus on data collection or domain expertise for further gains.",
                ["Prioritise data collection over feature engineering at this stage."])

    best = positive[0]
    impact = best["impact"]
    recs = []

    if impact >= 0.02:
        verdict = (f"✅ Strong win — '{best['name']}' boosts score by +{impact:.4f} "
                   f"({impact*100:.2f}%). Add this feature to production pipeline.")
        recs.append(f"Add '{best['name']}' (formula: {best['formula']}) to your feature set immediately.")
        if len(positive) > 1:
            recs.append(f"Also consider '{positive[1]['name']}' (+{positive[1]['impact']:.4f}).")
    elif impact > 0.005:
        verdict = (f"🟡 Mild improvement — '{best['name']}' adds +{impact:.4f}. "
                   "Include if model complexity budget allows.")
        recs.append(f"'{best['name']}' is a safe optional addition to your pipeline.")
    else:
        verdict = ("⚪ Marginal gain — best feature improves by only "
                   f"+{impact:.4f}. Existing features explain most variance.")
        recs.append("Your current features are well-chosen. Focus on hyperparameter tuning instead.")

    return verdict, recs


# ═════════════════════════════════════════════════════════════════════════════
# Model comparison insight
# ═════════════════════════════════════════════════════════════════════════════

def _model_comparison_insight(all_scores: dict, best_name: str) -> str:
    if len(all_scores) < 2:
        return ""
    sorted_m = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    parts = [f"{n}: {s:.4f}" for n, s in sorted_m]
    return f"All models tried: {' | '.join(parts)}. '{best_name}' selected as best."


# ═════════════════════════════════════════════════════════════════════════════
# Tuning insights
# ═════════════════════════════════════════════════════════════════════════════

def _tuning_insight(best_params: dict, model_name: str) -> str:
    if not best_params:
        return ""
    param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    return (f"Hyperparameter tuning optimised {model_name}: [{param_str}]. "
            "These settings were selected via 3-fold RandomizedSearchCV.")


# ═════════════════════════════════════════════════════════════════════════════
# Library availability advisory
# ═════════════════════════════════════════════════════════════════════════════

def _lib_advisory(libs: dict) -> list[str]:
    advisories = []
    if not libs.get("xgboost"):
        advisories.append(
            "Install XGBoost for better model selection: pip install xgboost")
    if not libs.get("lightgbm"):
        advisories.append(
            "Install LightGBM for faster training on large datasets: pip install lightgbm")
    if not libs.get("shap"):
        advisories.append(
            "Install SHAP for model explainability: pip install shap")
    return advisories


# ═════════════════════════════════════════════════════════════════════════════
# Detailed metrics insights
# ═════════════════════════════════════════════════════════════════════════════

def _metrics_insights(detailed: dict, cv_mean: float, task: str,
                       data_quality: dict) -> list[str]:
    insights = []
    if task == "classification":
        f1 = detailed.get("macro_f1")
        if f1 is not None and f1 < cv_mean - 0.05:
            insights.append(
                f"Macro F1 ({f1:.3f}) is notably lower than accuracy ({cv_mean:.3f}) — "
                "model is biased towards the majority class. Consider SMOTE or class_weight='balanced'.")
        conf = detailed.get("avg_confidence")
        low_conf = detailed.get("low_confidence_pct")
        if conf is not None:
            insights.append(
                f"Average prediction confidence: {conf*100:.1f}%. "
                f"{low_conf:.1f}% of predictions have <60% confidence — review borderline cases.")
        cb = data_quality.get("class_balance", {})
        if cb:
            cb_str = " | ".join(f"class {k}: {v}%" for k, v in cb.items())
            insights.append(f"Class distribution: {cb_str}.")
    else:
        mae  = detailed.get("mae")
        rmse = detailed.get("rmse")
        r2   = detailed.get("r2")
        if mae is not None:
            insights.append(f"Error metrics — MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
        pi = detailed.get("pred_interval_95")
        if pi is not None:
            insights.append(
                f"95th percentile prediction uncertainty: ±{pi:.4f} — "
                "predictions within this range should be treated as approximate.")
    return insights


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════════

def generate_insights(summary: dict, data_quality: dict, importance: dict,
                       features: list, eval_results: dict) -> dict:
    """Generate full plain-English insight report."""
    task        = eval_results.get("task", "classification")
    cv_mean     = eval_results.get("cv_mean", 0.0)
    cv_std      = eval_results.get("cv_std", 0.0)
    model_name  = eval_results.get("model_name", "")
    all_scores  = eval_results.get("all_model_scores", {})
    confidence  = eval_results.get("confidence_label", "")
    conf_reason = eval_results.get("confidence_reason", "")
    n_rows      = eval_results.get("n_rows", 0)
    target      = summary.get("target", "target")
    detailed    = eval_results.get("detailed_metrics", {})
    shap_imp    = eval_results.get("shap_importance", {})
    bias_warns  = eval_results.get("bias_warnings", [])
    best_params = eval_results.get("best_params", {})
    libs        = eval_results.get("libs", {})

    perf_line   = _interpret_score(cv_mean, task)
    score_recs  = _score_recommendations(cv_mean, task, n_rows)
    data_ins    = _data_insights(data_quality)
    driver_ins  = _top_driver_insights(importance, shap_imp, target)
    feat_verdict, feat_recs = _feature_verdict(features, cv_mean)
    model_cmp   = _model_comparison_insight(all_scores, model_name)
    metrics_ins = _metrics_insights(detailed, cv_mean, task, data_quality)
    tuning_ins  = _tuning_insight(best_params, model_name)
    lib_adv     = _lib_advisory(libs)

    # Build prioritised recommendations
    all_recs = []
    all_recs.extend(score_recs)
    all_recs.extend(feat_recs)
    for w in bias_warns:
        all_recs.append(f"[FAIRNESS] {w}")
    all_recs.extend(lib_adv)
    if not all_recs:
        all_recs = ["Results look solid. Validate on a fresh held-out test set before production deployment."]

    return {
        "performance_summary": perf_line,
        "confidence"         : f"{confidence} — {conf_reason}",
        "model_comparison"   : model_cmp,
        "tuning_summary"     : tuning_ins,
        "metrics_detail"     : metrics_ins,
        "top_drivers"        : driver_ins,
        "data_insights"      : data_ins,
        "feature_verdict"    : feat_verdict,
        "recommendations"    : all_recs,
        "bias_warnings"      : bias_warns,
        "lib_advisory"       : lib_adv,
    }