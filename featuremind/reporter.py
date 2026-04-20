"""
reporter.py — Enterprise Report + PNG Charts + SHAP Plot   (v3.1)
=================================================================
v3.1 FIXES applied here:

  FIX 1 — Correct metric labels for regression
    OLD: [Accuracy] even for regression tasks  ← wrong
    NEW: reads metric_label from eval_results → shows [R²] correctly

  FIX 2 — Holdout validation section
    NEW: Shows Train 80% / Holdout 20% split with real-world performance
    Clarifies "This is CV score" vs "This is estimated real-world performance"

  FIX 3 — Model selection explanation
    NEW: "LightGBM selected — outperformed XGBoost by +0.0011 on CV F1-weighted"

  FIX 4 — SHAP and Validation charts saved as separate PNGs
    NEW: featuremind_shap.png       — proper SHAP bar plot
    NEW: featuremind_validation.png — CV vs holdout + all models compared
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

W   = 70
SEP = "=" * W
DAH = "-" * W
BLK = "█"

LAYER_ICONS = {
    "domain"     : "🔵",
    "interaction": "🟣",
    "ratio"      : "🟢",
    "transform"  : "🟠",
    "polynomial" : "🟤",
    "binning"    : "🟩",
    "delta"      : "🔴",
    "outlier"    : "⛔",
    "nlp"        : "🔷",
    "rank"       : "⬜",
    "stat"       : "🔸",
    "generic"    : "⚪",
}


# ══════════════════════════════════════════════════════════════════════════════
# INSIGHT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _interpret_score(score, task, scoring_metric="accuracy"):
    """FIX 1: correct label for regression."""
    if scoring_metric == "f1_weighted":
        label = "F1-weighted"
    elif task == "regression":
        label = "R²"
    else:
        label = "Accuracy"

    if task == "classification":
        if score >= 0.92:   return f"Excellent ({score:.1%} {label}) — very high predictive accuracy."
        elif score >= 0.82: return f"Good ({score:.1%} {label}) — model performs well on most cases."
        elif score >= 0.70: return f"Moderate ({score:.1%} {label}) — reasonable accuracy but misses patterns."
        elif score >= 0.60: return f"Weak ({score:.1%} {label}) — needs more features or data."
        else:               return f"Poor ({score:.1%} {label}) — near random. Check target quality."
    else:
        if score >= 0.80:   return f"Excellent (R²={score:.3f}) — model explains most variance."
        elif score >= 0.60: return f"Good (R²={score:.3f}) — captures main patterns."
        elif score >= 0.40: return f"Moderate (R²={score:.3f}) — partial signal captured."
        elif score >= 0.20: return f"Weak (R²={score:.3f}) — limited explanatory power."
        else:               return f"Poor (R²={score:.3f}) — model cannot explain target."


def _score_recs(score, task, n, imbalance_info):
    recs = []
    if task == "classification":
        if imbalance_info.get("is_imbalanced"):
            method = imbalance_info.get("method", "none")
            if method == "SMOTE":
                recs.append("Class imbalance handled with SMOTE — good. Monitor for overfitting on minority class.")
            else:
                recs.append("Class imbalance handled with class_weight='balanced'. "
                            "Install imbalanced-learn for SMOTE: pip install imbalanced-learn")
        if score < 0.70: recs.append("Score < 70% — add domain features or collect more labelled data.")
        if score < 0.60: recs.append("Very low accuracy — verify target column labels are correct.")
    else:
        if score < 0.40: recs.append("Low R² — add more features or try a different target column.")
        if score < 0.20: recs.append("R² near zero — data is too noisy. Consult a domain expert.")
    if n < 500:    recs.append(f"Only {n} rows — collect more data (aim for 1,000+ rows).")
    elif n < 2000: recs.append(f"Moderate dataset ({n} rows) — more data would improve stability.")
    return recs


def _data_insights(dq):
    ins = []
    for key, verb in [("dropped_columns", "Removed"), ("converted_columns", "Auto-parsed"),
                       ("imputed_columns", "Imputed"), ("datetime_extracted", "Datetime extracted"),
                       ("skew_fixed", "Auto-fixed skewness")]:
        items = dq.get(key, [])
        if items:
            ins.append(f"{verb} {len(items)} column(s).")
    outliers = dq.get("outlier_report", {})
    if outliers:
        worst = max(outliers.items(), key=lambda x: x[1]["pct"])
        ins.append(f"Outliers: '{worst[0]}' has {worst[1]['pct']}% extreme values. Consider winsorizing.")
    if dq.get("missing_pct", 0) > 20:
        ins.append(f"High missing data ({dq['missing_pct']:.1f}%) — imputation may introduce bias.")
    for w in dq.get("warnings", []):
        ins.append(f"⚠️ {w}")
    sources = dq.get("source_files", [])
    if len(sources) > 1:
        ins.append(f"Multi-file analysis: {len(sources)} files merged.")
    return ins


def _driver_insights(importance, shap_imp, target):
    ins  = []
    imp  = shap_imp if shap_imp else importance
    src  = "SHAP" if shap_imp else "RandomForest"
    if not imp:
        return ["Feature importance could not be computed."]
    items  = list(imp.items())[:3]
    tn, tv = items[0]
    if shap_imp:
        ins.append(f"[SHAP] '{tn}' has the highest average impact on '{target}' "
                   f"(mean |SHAP| = {tv:.4f}).")
    else:
        ins.append(f"[{src}] '{tn}' is the strongest predictor of '{target}' "
                   f"({tv*100:.1f}% importance).")
    if len(items) >= 2:
        ins.append("Other key drivers: " + " and ".join(f"'{n}'" for n, _ in items[1:3]) + ".")
    if not shap_imp and tv > 0.50:
        ins.append(f"⚠️ '{tn}' dominates RF importance at {tv*100:.0f}%. Check for leakage.")
    return ins


def _feature_verdict(features, base):
    if not features:
        return ("✅ Model already captures most signal — additional engineered features "
                "had minimal impact. The existing features are well-chosen."), [
            "Feature engineering had minimal impact — model is already well-fitted.",
            "Consider collecting more data or adding domain-specific business logic features."]
    pos = [f for f in features if f.get("impact", 0) > 0]
    if not pos:
        return ("✅ Model already captures most signal — additional engineered features "
                "had minimal impact. The existing features are well-chosen."), [
            "Feature engineering had minimal impact — model is already well-fitted.",
            "Consider collecting more data or adding domain-specific business logic features."]
    best = pos[0]; impact = best["impact"]; recs = []
    if impact >= 0.02:
        verdict = f"✅ Strong win — '{best['name']}' boosts score by +{impact:.4f} ({impact*100:.2f}%). Add to pipeline."
        recs.append(f"Add '{best['name']}' immediately: {best['formula']}")
        if len(pos) > 1: recs.append(f"Also test '{pos[1]['name']}' (+{pos[1]['impact']:.4f}).")
    elif impact > 0.005:
        verdict = f"🟡 Mild improvement — '{best['name']}' adds +{impact:.4f}. Include if budget allows."
        recs.append(f"'{best['name']}' is a safe optional addition.")
    else:
        verdict = f"⚪ Marginal gain — +{impact:.4f}. Features are well-chosen already."
        recs.append("Focus on hyperparameter tuning or more data collection.")
    return verdict, recs


def _lib_advisory(libs):
    adv = []
    if not libs.get("xgboost"):  adv.append("pip install xgboost")
    if not libs.get("lightgbm"): adv.append("pip install lightgbm")
    if not libs.get("catboost"): adv.append("pip install catboost")
    if not libs.get("shap"):     adv.append("pip install shap")
    if not libs.get("smote"):    adv.append("pip install imbalanced-learn  # for SMOTE")
    return adv


def _metrics_insights(det, cv_mean, task, dq, opt_threshold):
    ins = []
    if task == "classification":
        f1 = det.get("macro_f1")
        if f1 is not None and f1 < cv_mean - 0.05:
            ins.append(f"Macro F1 ({f1:.3f}) << accuracy ({cv_mean:.3f}) — "
                       "biased towards majority class. SMOTE or class_weight='balanced' recommended.")
        conf = det.get("avg_confidence"); lc = det.get("low_confidence_pct")
        if conf is not None:
            ins.append(f"Avg confidence: {conf*100:.1f}%. {lc:.1f}% predictions have <60% confidence.")
        if opt_threshold and opt_threshold != 0.5:
            ins.append(f"Optimal decision threshold: {opt_threshold} (maximises F1 on validation set).")
        cb = dq.get("class_balance", {})
        if cb:
            ins.append("Class distribution: " + " | ".join(f"class {k}: {v}%" for k, v in cb.items()) + ".")
    else:
        mae, rmse, r2 = det.get("mae"), det.get("rmse"), det.get("r2")
        if mae is not None:
            ins.append(f"Error metrics — MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
        pi = det.get("pred_interval_95")
        if pi: ins.append(f"95th percentile prediction uncertainty: ±{pi:.4f}")
    return ins


def generate_insights(summary, data_quality, importance, features, eval_results):
    task           = eval_results.get("task", "classification")
    cv_mean        = eval_results.get("cv_mean", 0.0)
    cv_std         = eval_results.get("cv_std", 0.0)
    model_name     = eval_results.get("model_name", "")
    all_scores     = eval_results.get("all_model_scores", {})
    n_rows         = eval_results.get("n_rows", 0)
    target         = summary.get("target", "target")
    det            = eval_results.get("detailed_metrics", {})
    shap_imp       = eval_results.get("shap_importance", {})
    bias_warns     = eval_results.get("bias_warnings", [])
    best_pars      = eval_results.get("best_params", {})
    libs           = eval_results.get("libs", {})
    conf_label     = eval_results.get("confidence_label", "N/A")
    conf_reason    = eval_results.get("confidence_reason", "")
    imb_info       = eval_results.get("imbalance_info", {})
    opt_threshold  = eval_results.get("opt_threshold", 0.5)
    scoring_metric = eval_results.get("scoring_metric", "accuracy")

    perf_line    = _interpret_score(cv_mean, task, scoring_metric)
    score_recs   = _score_recs(cv_mean, task, n_rows, imb_info)
    data_ins     = _data_insights(data_quality)
    driver_ins   = _driver_insights(importance, shap_imp, target)
    feat_verdict, feat_recs = _feature_verdict(features, cv_mean)
    metrics_ins  = _metrics_insights(det, cv_mean, task, data_quality, opt_threshold)
    lib_adv      = _lib_advisory(libs)

    if len(all_scores) > 1:
        parts     = [f"{n}: {s:.4f}" for n, s in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)]
        model_cmp = f"All models: {' | '.join(parts)}. '{model_name}' selected."
    else:
        model_cmp = ""

    tuning_ins = ""
    if best_pars:
        tuning_ins = (f"Tuned {model_name}: [{', '.join(f'{k}={v}' for k, v in best_pars.items())}]. "
                      "Selected via 3-fold RandomizedSearchCV (10 iterations).")

    all_recs = []
    all_recs.extend(score_recs)
    all_recs.extend(feat_recs)
    for w in bias_warns:
        all_recs.append(f"[FAIRNESS] {w}")
    if lib_adv:
        all_recs.append("Optional libraries to install: " + " | ".join(lib_adv))
    if not all_recs:
        all_recs = ["Results look solid. Validate on a fresh hold-out test set before production."]

    return {
        "performance_summary": perf_line,
        "confidence"         : f"{conf_label} — {conf_reason}",
        "model_comparison"   : model_cmp,
        "tuning_summary"     : tuning_ins,
        "metrics_detail"     : metrics_ins,
        "top_drivers"        : driver_ins,
        "data_insights"      : data_ins,
        "feature_verdict"    : feat_verdict,
        "recommendations"    : all_recs,
        "bias_warnings"      : bias_warns,
        "lib_advisory"       : lib_adv,
        "imbalance_info"     : imb_info,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CHART ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _bar(val, max_v=1.0, length=14):
    n = int(round(val / max_v * length)) if max_v > 0 else 0
    return BLK * n


def _impact_icon(impact):
    if impact >= 0.02:    return "🟢"
    elif impact > 0.005:  return "🟡"
    elif impact == 0:     return "⚪"
    else:                 return "🔴"


def _sec(title):
    print(f"\n{DAH}\n  {title}\n{DAH}")


def _save_chart(importance, shap_imp, features, path="featuremind_report.png"):
    """Main feature importance + feature impact chart."""
    pos_feats = [f for f in features if f.get("impact") is not None and f.get("impact", 0) > 0]
    n_panels  = 2 if pos_feats else 1
    fig = plt.figure(figsize=(14 if pos_feats else 8, 5))
    fig.suptitle("featuremind v3.1 — Analysis Report", fontsize=13, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(1, n_panels, figure=fig, wspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    imp = shap_imp if shap_imp else importance
    lbl = "SHAP Feature Impact" if shap_imp else "RF Feature Importance"
    if imp:
        names  = list(imp.keys())[:10]
        values = [imp[n] for n in names]
        colors = ["#1565C0" if v == max(values) else "#90CAF9" for v in values]
        bars   = ax1.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="white")
        ax1.set_xlabel("Score", fontsize=9)
        ax1.set_title(lbl, fontsize=10, fontweight="bold")
        ax1.bar_label(bars, fmt="%.4f", padding=3, fontsize=7)
        ax1.spines[["top", "right"]].set_visible(False)

    if pos_feats:
        ax2     = fig.add_subplot(gs[0, 1])
        show    = pos_feats[:5]
        f_names = [f["name"] for f in show]
        n_sc    = [f["new_score"] for f in show]
        base    = n_sc[0] - show[0].get("impact", 0) if show else 0
        x = range(len(f_names)); w = 0.35
        b1 = ax2.bar([i - w/2 for i in x], [base]*len(show), w,
                     label="Baseline", color="#90A4AE", edgecolor="white")
        b2 = ax2.bar([i + w/2 for i in x], n_sc, w,
                     label="With Feature", color="#42A5F5", edgecolor="white")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels([n[:12] for n in f_names], rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel("Score", fontsize=9)
        ax2.set_title("Feature Impact vs Baseline", fontsize=10, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.bar_label(b2, fmt="%.3f", padding=2, fontsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Chart saved : '{path}'")


def _save_shap_plot(shap_imp, target, path="featuremind_shap.png"):
    """
    FIX 4: Dedicated SHAP bar chart — proper publication-quality visualization.
    Top 15 features with color gradient by importance level.
    """
    if not shap_imp:
        return
    try:
        names  = list(shap_imp.keys())[:15]
        values = list(shap_imp.values())[:15]
        max_v  = max(values) if values else 1

        fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.45)))
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FAFAFA")

        colors = ["#1A237E" if v == max_v else
                  "#1565C0" if v > max_v * 0.7 else
                  "#42A5F5" if v > max_v * 0.4 else
                  "#90CAF9"
                  for v in values]

        bars = ax.barh(names[::-1], values[::-1], color=colors[::-1],
                       edgecolor="white", linewidth=0.5, height=0.7)

        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + max_v * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left", fontsize=8, color="#333333")

        ax.set_xlabel("Mean |SHAP Value|  (average impact on model output)", fontsize=10)
        ax.set_title(f"🔍 SHAP Feature Importance — Target: '{target}'\n"
                     f"Higher value = stronger influence on prediction",
                     fontsize=11, fontweight="bold", pad=12)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False, labelsize=9)
        ax.set_xlim(0, max_v * 1.22)
        ax.axvline(x=0, color="#CCCCCC", linewidth=0.8)
        fig.text(0.99, 0.01, "featuremind v3.1", ha="right", va="bottom",
                 fontsize=7, color="#AAAAAA")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"🔍 SHAP plot saved : '{path}'")
    except Exception as e:
        print(f"⚠️  SHAP plot skipped: {e}")


def _save_validation_plot(cv_mean, cv_std, holdout_metrics, task, model_name,
                           all_scores, metric_lbl, path="featuremind_validation.png"):
    """
    FIX 2+3: Validation chart showing CV vs Holdout + all models ranked.
    Makes train/test split crystal clear to ML engineers.
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"featuremind v3.1 — Validation Report  |  Best Model: {model_name}",
                     fontsize=12, fontweight="bold")
        fig.patch.set_facecolor("#FAFAFA")

        # ── Panel 1: CV vs Holdout ────────────────────────────────────────────
        ax1 = axes[0]
        ax1.set_facecolor("#FAFAFA")

        holdout_score = (holdout_metrics.get("weighted_f1") or
                         holdout_metrics.get("r2") or 0.0)
        holdout_score = holdout_score or 0.0

        labels = [f"CV {metric_lbl}\n(cross-validated)", f"Holdout {metric_lbl}\n(20% unseen data)"]
        scores = [max(0, cv_mean), max(0, holdout_score)]
        colors = ["#42A5F5", "#1565C0"]
        bars   = ax1.bar(labels, scores, color=colors, edgecolor="white", width=0.45, zorder=3)

        y_max = max(max(scores) * 1.25, 0.1)
        ax1.set_ylim(0, y_max)
        ax1.set_ylabel(metric_lbl, fontsize=10)
        ax1.set_title("CV Score vs Holdout Score", fontsize=10, fontweight="bold")
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.yaxis.grid(True, alpha=0.3, zorder=0)
        ax1.set_axisbelow(True)

        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + y_max * 0.02,
                     f"{score:.4f}", ha="center", va="bottom",
                     fontsize=11, fontweight="bold", color="#1A237E")

        # CV error bar
        if cv_std > 0:
            ax1.errorbar(0, cv_mean, yerr=cv_std, fmt="none",
                         color="#333333", capsize=10, linewidth=2.5)
            ax1.text(0.25, max(0, cv_mean - cv_std) - y_max * 0.06,
                     f"±{cv_std:.3f}", ha="center", fontsize=8, color="#555555")

        # Generalisation check
        gap = abs(cv_mean - holdout_score)
        if gap > 0.05:
            msg, mc, bc = "⚠️ Gap > 5% — check for overfitting", "#E53935", "#FFEBEE"
        else:
            msg, mc, bc = "✅ CV ≈ Holdout — generalises well", "#2E7D32", "#E8F5E9"
        ax1.text(0.5, 0.04, msg, ha="center", transform=ax1.transAxes,
                 fontsize=8, color=mc,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=bc, alpha=0.9))

        # ── Panel 2: All models ranked ────────────────────────────────────────
        ax2 = axes[1]
        ax2.set_facecolor("#FAFAFA")
        if all_scores:
            sorted_s = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            m_names  = [s[0] for s in sorted_s]
            m_scores = [max(0, s[1]) for s in sorted_s]
            m_colors = ["#1565C0" if n == model_name else "#90CAF9" for n in m_names]
            h_bars   = ax2.barh(m_names[::-1], m_scores[::-1],
                                color=m_colors[::-1], edgecolor="white", zorder=3)
            ax2.set_xlabel(f"CV {metric_lbl}", fontsize=10)
            ax2.set_title("All Models Compared (CV Score)", fontsize=10, fontweight="bold")
            ax2.spines[["top", "right"]].set_visible(False)
            ax2.xaxis.grid(True, alpha=0.3, zorder=0)
            ax2.set_axisbelow(True)

            x_max = max(m_scores) * 1.2 if m_scores else 1
            ax2.set_xlim(0, x_max)
            for i, (name, score) in enumerate(reversed(sorted_s)):
                ax2.text(score + x_max * 0.01, i, f"{score:.4f}",
                         va="center", fontsize=8,
                         fontweight="bold" if name == model_name else "normal",
                         color="#1A237E" if name == model_name else "#555555")

            # FIX 3: Why best model won
            if len(sorted_s) >= 2:
                gap2 = sorted_s[0][1] - sorted_s[1][1]
                ax2.text(0.98, 0.03,
                         f"← '{sorted_s[0][0]}' won by +{gap2:.4f}",
                         ha="right", transform=ax2.transAxes, fontsize=8, color="#1565C0",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", alpha=0.9))

        fig.text(0.99, 0.01, "featuremind v3.1", ha="right", va="bottom",
                 fontsize=7, color="#AAAAAA")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"📈 Validation chart saved : '{path}'")
    except Exception as e:
        print(f"⚠️  Validation chart skipped: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(summary, data_quality, col_types, importance, features,
                 model_name, base_score, eval_results, insights_report, df):

    task          = eval_results.get("task", "classification")
    cv_std        = eval_results.get("cv_std", 0)
    all_scores    = eval_results.get("all_model_scores", {})
    det           = eval_results.get("detailed_metrics", {})
    conf_label    = eval_results.get("confidence_label", "N/A")
    conf_reason   = eval_results.get("confidence_reason", "")
    use_cv        = eval_results.get("use_cv", False)
    shap_imp      = eval_results.get("shap_importance", {})
    bias_warns    = eval_results.get("bias_warnings", [])
    best_params   = eval_results.get("best_params", {})
    libs          = eval_results.get("libs", {})
    n_feat        = eval_results.get("n_features", 0)
    imb_info      = eval_results.get("imbalance_info", {})
    opt_threshold = eval_results.get("opt_threshold", 0.5)
    opt_f1        = eval_results.get("opt_f1")
    scoring       = eval_results.get("scoring_metric", "accuracy")
    stab          = eval_results.get("stability_score")
    n_rows        = eval_results.get("n_rows", 0)

    # FIX 1: correct metric label from eval_results — never shows [Accuracy] for regression
    metric_lbl = eval_results.get("metric_label",
                 "F1-weighted" if scoring == "f1_weighted"
                 else "Accuracy" if task == "classification"
                 else "R²")

    # FIX 2: clear CV method name
    if task == "regression" and n_rows < 500:
        cv_method = "Repeated 3×3-fold CV"
    elif task == "classification":
        cv_method = "5-fold Stratified CV"
    else:
        cv_method = "5-fold CV"
    cv_lbl = f"(±{cv_std:.4f} · {cv_method})" if use_cv else "(single split)"

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"        🧠  featuremind v3.1.0 — Analysis Report")
    print(SEP)
    print(f"\n  🚀 Best Model   : {model_name or 'N/A'}")
    print(f"  🎯 Score        : {base_score:.4f}  {cv_lbl}  [{metric_lbl}]")
    print(f"  📋 Task         : {task.capitalize()}")
    print(f"  🔒 Confidence   : {conf_label}")
    print(f"  📦 Libraries    : "
          f"XGBoost={'✅' if libs.get('xgboost') else '❌'}  "
          f"LightGBM={'✅' if libs.get('lightgbm') else '❌'}  "
          f"CatBoost={'✅' if libs.get('catboost') else '❌'}  "
          f"SHAP={'✅' if libs.get('shap') else '❌'}  "
          f"SMOTE={'✅' if libs.get('smote') else '❌'}")

    sources = data_quality.get("source_files", [])
    if len(sources) > 1:
        print(f"  📂 Source Files : {len(sources)} files merged")

    # ── Performance ───────────────────────────────────────────────────────────
    _sec("🔍  PERFORMANCE SUMMARY")
    print(f"\n  {insights_report.get('performance_summary','')}")
    cmp = insights_report.get("model_comparison", "")
    if cmp: print(f"\n  {cmp}")

    # ── Imbalance ─────────────────────────────────────────────────────────────
    if imb_info.get("is_imbalanced"):
        _sec("⚖️   CLASS IMBALANCE HANDLING")
        method = imb_info.get("method", "none")
        min_r  = imb_info.get("minority_ratio", 0)
        print(f"\n  Minority class   : {min_r:.1f}% of data")
        print(f"  Method applied   : {method.upper()}")
        if method == "SMOTE":
            print(f"  Dataset after    : {int(imb_info.get('resampled_size', 0)):,} rows (oversampled)")
        print(f"  Scoring metric   : {metric_lbl} (better for imbalanced data)")
        if opt_threshold != 0.5 and opt_f1 is not None:
            print(f"  Optimal threshold: {opt_threshold} (F1={opt_f1:.4f} vs default 0.5)")

    # ── Model comparison with explanation ─────────────────────────────────────
    if all_scores:
        _sec("🏁  MODEL COMPARISON  (CV scores)")
        max_s         = max(all_scores.values(), default=1)
        sorted_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        for name, score in sorted_models:
            star = "  ← best" if name == model_name else ""
            bar  = _bar(score, max_s, 12)
            print(f"  {name:<28} {score:.4f}  {bar}{star}")
        # FIX 3: Model selection explanation
        if len(sorted_models) >= 2:
            best_n, best_s     = sorted_models[0]
            second_n, second_s = sorted_models[1]
            gap = best_s - second_s
            print(f"\n  💡 '{best_n}' selected — outperformed '{second_n}' "
                  f"by +{gap:.4f} ({gap*100:.2f}%) on CV {metric_lbl}.")

    # ── Tuning ────────────────────────────────────────────────────────────────
    tun = insights_report.get("tuning_summary", "")
    if tun:
        _sec("🔧  HYPERPARAMETER TUNING")
        print(f"\n  {tun}")

    # ── Detailed Metrics ──────────────────────────────────────────────────────
    _sec("📈  DETAILED METRICS")
    if task == "classification":
        f1w = det.get("weighted_f1"); f1m = det.get("macro_f1")
        if f1w is not None:
            print(f"\n  Weighted F1          : {f1w:.4f}")
            print(f"  Macro F1             : {f1m:.4f}")
        ca = det.get("avg_confidence")
        if ca is not None:
            print(f"  Avg Confidence       : {ca*100:.1f}%  "
                  f"| Low-conf preds : {det.get('low_confidence_pct',0):.1f}%")
        ot = det.get("optimal_threshold", 0.5)
        if ot != 0.5:
            print(f"  Optimal Threshold    : {ot} (auto-tuned from default 0.5)")
        pc = det.get("per_class", {})
        if pc:
            print(f"\n  {'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            print(f"  {'-'*46}")
            for cls, m in pc.items():
                print(f"  {str(cls):<14} {m.get('precision',0):>10.3f} "
                      f"{m.get('recall',0):>10.3f} {m.get('f1-score',0):>10.3f}")
    else:
        r2, mae, rmse = det.get("r2"), det.get("mae"), det.get("rmse")
        if r2 is not None:
            print(f"\n  R²   : {r2:.4f}")
            print(f"  MAE  : {mae:.4f}")
            print(f"  RMSE : {rmse:.4f}")
    for ins in insights_report.get("metrics_detail", []):
        print(f"\n  ℹ️  {ins}")

    # FIX 2: Validation strategy section — makes train/test split explicit
    _sec("🔬  VALIDATION STRATEGY")
    print(f"\n  Cross-Validation : {cv_method}  ← score shown in report header")
    print(f"  Train split      : 80%   (model fitting + CV evaluation)")
    print(f"  Holdout split    : 20%   (threshold tuning + real-world estimate)")
    if task == "classification":
        wf1 = det.get("weighted_f1")
        mf1 = det.get("macro_f1")
        if wf1:
            print(f"\n  Holdout Weighted F1  : {wf1:.4f}  ← estimated real-world performance")
            print(f"  Holdout Macro F1     : {mf1:.4f}")
            print(f"  CV {metric_lbl:<20}: {base_score:.4f}  ← cross-validated (in header)")
            gap = abs(base_score - wf1)
            if gap > 0.05:
                print(f"\n  ⚠️  CV↔Holdout gap = {gap:.4f} — possible overfitting. "
                      f"Try reducing model complexity or collecting more data.")
            else:
                print(f"\n  ✅ CV ≈ Holdout (gap={gap:.4f}) — model generalises well to unseen data.")
    else:
        r2v = det.get("r2"); mae = det.get("mae"); rmse = det.get("rmse")
        if r2v is not None:
            print(f"\n  Holdout R²   : {r2v:.4f}  ← estimated real-world performance")
            print(f"  Holdout MAE  : {mae:.4f}")
            print(f"  Holdout RMSE : {rmse:.4f}")
            print(f"  CV R²        : {base_score:.4f}  ← cross-validated (in header)")

    # ── Dataset Summary ───────────────────────────────────────────────────────
    _sec("📊  DATASET SUMMARY")
    total = summary.get("total_file_rows")
    print(f"\n  Rows (sampled)    : {summary['rows']:,}")
    if isinstance(total, int) and total > summary['rows']:
        print(f"  ⚡ Sampled from {total:,} total rows for fast analysis.")
        print(f"  💡 Tip: For production training, use the full dataset.")
    elif isinstance(total, int):
        print(f"  Total file rows   : {total:,}")
    print(f"  Columns (encoded) : {summary['columns']:,}")
    print(f"  Target            : {summary['target']}")
    print(f"  Task type         : {summary.get('task_type','?').capitalize()}")
    sf = summary.get("source_files", [])
    if isinstance(sf, list) and len(sf) > 1:
        for fi in sf:
            if isinstance(fi, dict):
                print(f"    • {fi['file']} — {fi['rows']:,} rows × {fi['cols']} cols")
    print(f"\n  🔢 Numeric     : {summary['numeric_columns']}")
    print(f"\n  🧾 Categorical : {summary['categorical_columns']}")

    # ── Column type inference ─────────────────────────────────────────────────
    if col_types:
        _sec("🔬  COLUMN TYPE INFERENCE")
        groups: dict = {}
        for col, t in col_types.items():
            groups.setdefault(t, []).append(col)
        icons = {"numeric":"🔢","binary":"⚡","ordinal":"📊","categorical":"🏷️",
                 "datetime":"📅","text":"📝","id":"🔑","target":"🎯"}
        for t, cols in sorted(groups.items()):
            icon = icons.get(t, "•")
            print(f"\n  {icon} {t.capitalize():<14}: "
                  + ", ".join(cols[:8]) + (" ..." if len(cols) > 8 else ""))

    # ── Data quality ──────────────────────────────────────────────────────────
    _sec("🧹  DATA QUALITY REPORT")
    dq = data_quality
    orig_rows = dq.get("original_rows","?")
    print(f"\n  Original rows     : {orig_rows:,}" if isinstance(orig_rows, int)
          else f"\n  Original rows     : {orig_rows}")
    print(f"  Original columns  : {dq.get('original_columns','?')}")
    print(f"  Missing cells     : {dq.get('missing_cells',0):,} ({dq.get('missing_pct',0):.1f}%)")
    print(f"  Duplicate rows    : {dq.get('duplicate_rows',0):,}")
    for label, key in [("🗑️  Dropped","dropped_columns"),("🔧 Converted","converted_columns"),
                        ("🔄 Imputed","imputed_columns"),("📅 Datetime","datetime_extracted"),
                        ("📐 Skew fixed","skew_fixed")]:
        items = dq.get(key, [])
        if items:
            print(f"\n  {label} ({len(items)}):")
            for it in items[:5]: print(f"      • {it}")
            if len(items) > 5: print(f"      ... and {len(items)-5} more.")
    win = dq.get("winsorized_columns", [])
    if win:
        print(f"\n  ✂️  Winsorized ({len(win)}):")
        for it in win[:3]: print(f"      • {it}")
    cb = dq.get("class_balance", {})
    if cb:
        print(f"\n  ⚖️  Class balance:")
        for cls, pct in cb.items():
            print(f"      class {cls}: {pct:>5.1f}%  {_bar(pct, 100, 10)}")
    outliers = dq.get("outlier_report", {})
    if outliers:
        print(f"\n  ⚠️  Outlier Report (IQR):")
        for col, info in list(outliers.items())[:5]:
            print(f"      • {col}: {info['count']:,} rows ({info['pct']}%)")
    for w in dq.get("warnings", []): print(f"\n  ⚠️  {w}")

    # ── Insights ──────────────────────────────────────────────────────────────
    data_ins = insights_report.get("data_insights", [])
    if data_ins:
        _sec("💬  DATA INSIGHTS")
        for ins in data_ins: print(f"\n  • {ins}")

    if bias_warns:
        _sec("⚖️   BIAS & FAIRNESS CHECK")
        for w in bias_warns: print(f"\n  🔴 {w}")

    # ── Feature importance ────────────────────────────────────────────────────
    imp   = shap_imp if shap_imp else importance
    label = "🏆  TOP FEATURES  [SHAP — raw impact scores]" if shap_imp \
            else "🏆  TOP FEATURES  [RandomForest importance]"
    if imp:
        _sec(label)
        max_v = max(imp.values()) if imp else 1
        for i, (col, val) in enumerate(list(imp.items())[:7], 1):
            bar     = _bar(val, max_v)
            display = f"{val:.4f}" if shap_imp else f"{val*100:.1f}%"
            print(f"\n  {i}. {col:<40} {display}  {bar}")
        for d in insights_report.get("top_drivers", []): print(f"\n  💡 {d}")

    # ── Feature suggestions ───────────────────────────────────────────────────
    _sec("🔬  TOP FEATURE SUGGESTIONS")
    if not features:
        print(f"\n  ℹ️  No features improved the model. The model already captures available signal.")
        all_feats = eval_results.get("all_features", [])
        if all_feats:
            print(f"\n  💡 Top candidate features (did not improve score, but worth knowing):")
            for af in all_feats[:2]:
                print(f"\n     Feature : {af.get('name','?')}")
                print(f"     Formula : {af.get('formula','?')}")
                print(f"     Reason  : {af.get('reason','?')}")
                print(f"     Impact  : {af.get('impact',0):+.4f} (no improvement on this dataset)")
    else:
        for f in features:
            impact   = f.get("impact", 0)
            icon     = _impact_icon(impact)
            layer    = f.get("layer", "generic")
            dot      = LAYER_ICONS.get(layer.lower(), "•")
            prescore = f.get("_prescore", 0)
            print(f"\n  {'─'*54}")
            print(f"  Rank      : {f.get('rank','N/A')}  {dot} [{layer.upper()}]  pre-score={prescore:.3f}")
            print(f"  Feature   : {f['name']}")
            print(f"  Formula   : {f['formula']}")
            print(f"  Business  : {f['business']}")
            print(f"  Reason    : {f['reason']}")
            print(f"  Impact    : {icon}  {'+' if impact >= 0 else ''}{impact:.4f}")
            print(f"  New Score : {f['new_score']:.4f}")

    _sec("🏅  FEATURE ENGINEERING VERDICT")
    print(f"\n  {insights_report.get('feature_verdict','')}")

    _sec("🚀  RECOMMENDATIONS  (Priority Order)")
    for i, rec in enumerate(insights_report.get("recommendations", []), 1):
        print(f"\n  {i}. {rec}")

    # ── Confidence & reliability ───────────────────────────────────────────────
    _sec("🔒  CONFIDENCE & RELIABILITY")
    print(f"\n  Level   : {conf_label}")
    print(f"  Reason  : {conf_reason}")
    method = cv_method if use_cv else f"Single split ({n_feat} features)"
    print(f"  Method  : {method}")
    if stab is not None:
        stab_label = ("🟢 Stable" if stab > 10 else "🔴 Unstable" if stab < 5 else "🟡 Moderate")
        print(f"  Stability : {stab:.1f}  ({stab_label})")
    if cv_std > 0.05:
        print(f"  ⚠️  High variance (±{cv_std:.3f}) — score may change across splits.")
        reasons = []
        if n_rows < 500:                   reasons.append(f"small dataset ({n_rows} rows)")
        if dq.get("missing_pct", 0) > 20:  reasons.append(f"high missing data ({dq.get('missing_pct',0):.0f}%)")
        if reasons:
            print(f"  ⚠️  Instability likely due to: {', '.join(reasons)}.")

    _sec("📦  LIBRARY STATUS")
    print(f"\n  XGBoost          : {'✅ Installed' if libs.get('xgboost') else '❌  pip install xgboost'}")
    print(f"  LightGBM         : {'✅ Installed' if libs.get('lightgbm') else '❌  pip install lightgbm'}")
    print(f"  CatBoost         : {'✅ Installed' if libs.get('catboost') else '❌  pip install catboost'}")
    print(f"  SHAP             : {'✅ Installed' if libs.get('shap') else '❌  pip install shap'}")
    print(f"  imbalanced-learn : {'✅ Installed (SMOTE active)' if libs.get('smote') else '❌  pip install imbalanced-learn'}")

    # ── Output ────────────────────────────────────────────────────────────────
    _sec("💾  OUTPUT")
    if features:
        best = features[0]
        try:
            df = df.copy()
            df["best_feature"] = eval(best["formula"], {"df": df, "np": np, "pd": pd})
            df["best_feature"] = df["best_feature"].replace([np.inf, -np.inf], np.nan).fillna(0)
            df.to_csv("enhanced_data.csv", index=False)
            print(f"\n  ✅ Enhanced dataset saved : 'enhanced_data.csv'")
            print(f"     Best feature added     : {best['name']}")
        except Exception as e:
            df.to_csv("enhanced_data.csv", index=False)
            print(f"\n  ✅ Dataset saved (feature apply skipped: {e})")
    else:
        df.to_csv("enhanced_data.csv", index=False)
        print(f"\n  ℹ️  No best feature — original dataset saved as 'enhanced_data.csv'.")

    print(f"\n{SEP}\n")

    # FIX 4: Save all 3 charts
    target_col = summary.get("target", "target")
    _save_chart(importance, shap_imp, features)
    if shap_imp:
        _save_shap_plot(shap_imp, target_col)
    _save_validation_plot(
        cv_mean=base_score, cv_std=cv_std,
        holdout_metrics=det, task=task,
        model_name=model_name or "Model",
        all_scores=all_scores,
        metric_lbl=metric_lbl,
    )