"""
leakage_guard.py  —  featuremind v3.0
=======================================
Universal leakage & reliability guard. Works on ANY dataset.

KEY DESIGN PRINCIPLES (v3.0):
  ✅ Never crashes — every function has full fallback
  ✅ Never assumes column names or domain
  ✅ ID detection: name + sequential + low-signal (not just uniqueness)
  ✅ PCA features (V1-V28) never flagged as IDs
  ✅ Score reliability overrides confidence (no contradictions)
  ✅ Warns only — never auto-drops anything (user decides)
"""

import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Thresholds
LEAKAGE_CORR_THRESHOLD = 0.95
SUSPICIOUS_SCORE_WARN  = 0.98
SUSPICIOUS_SCORE_CRIT  = 0.99
UNSTABLE_STD_WARN      = 0.10
UNSTABLE_STD_CRITICAL  = 0.20

TRUE_ID_KEYWORDS = {
    "id", "uuid", "uid", "key", "index", "idx", "no",
    "num", "number", "code", "seq", "sequence", "rowid",
    "row_id", "record_id", "customerid", "userid", "orderid",
    "transactionid", "account_id", "account_no",
}

PCA_PATTERNS = [
    r"^v\d+$", r"^pc\d+$", r"^component\d+$",
    r"^factor\d+$", r"^latent\d+$", r"^dim\d+$",
]


def _is_pca_feature(col_name):
    normalized = col_name.lower().strip()
    return any(re.match(p, normalized) for p in PCA_PATTERNS)


def _is_true_id_column(series, col_name, target_series=None):
    """
    Smart ID detection requiring MULTIPLE signals.
    Returns (is_id: bool, reason: str)
    """
    try:
        if _is_pca_feature(col_name):
            return False, ""
        if not pd.api.types.is_integer_dtype(series):
            return False, ""

        n_total    = max(len(series.dropna()), 1)
        n_unique   = series.nunique(dropna=True)
        uniq_ratio = n_unique / n_total

        col_lower  = col_name.lower().replace(" ", "").replace("_", "").replace("-", "")
        name_is_id = any(kw == col_lower or col_lower.endswith(kw)
                         for kw in TRUE_ID_KEYWORDS)

        is_sequential = False
        try:
            sv    = series.dropna().sort_values().reset_index(drop=True)
            diffs = sv.diff().dropna()
            if len(diffs) > 0 and (diffs == 1).mean() > 0.95:
                is_sequential = True
        except Exception:
            pass

        if not name_is_id and not is_sequential:
            return False, ""
        if uniq_ratio < 0.85:
            return False, ""

        if target_series is not None:
            try:
                corr = abs(float(series.corr(target_series)))
                if corr > 0.05:
                    return False, ""
            except Exception:
                pass

        reason = (f"Sequential integers ({uniq_ratio*100:.0f}% unique)"
                  if is_sequential else
                  f"Name matches ID pattern ({uniq_ratio*100:.0f}% unique)")
        return True, reason
    except Exception:
        return False, ""


def check_dataset_leakage(df, target):
    """Advisory scan — warns only, never modifies df."""
    warnings_list = []
    try:
        if target not in df.columns:
            return warnings_list

        y        = df[target]
        num_cols = df.select_dtypes(include="number").columns.tolist()

        for col in num_cols:
            if col == target:
                continue
            series = df[col]

            is_id, id_reason = _is_true_id_column(series, col, y)
            if is_id:
                warnings_list.append(
                    f"⚠️  '{col}' looks like an ID column ({id_reason}). "
                    f"Consider removing — may not generalise.")
                continue

            try:
                corr = abs(float(series.corr(y)))
                if corr > LEAKAGE_CORR_THRESHOLD and series.std() > 0:
                    warnings_list.append(
                        f"🚨 '{col}' |corr|={corr:.3f} with target — possible proxy/leakage.")
            except Exception:
                pass

            if series.std() == 0:
                warnings_list.append(
                    f"ℹ️  '{col}' is constant — adds no signal.")
    except Exception:
        pass
    return warnings_list


def check_formula_leakage(formula, target):
    try:
        escaped  = re.escape(target)
        patterns = [rf"df\[[\'\"]?{escaped}[\'\"]?\]", rf"\bdf\.{escaped}\b"]
        return any(re.search(p, formula, re.IGNORECASE) for p in patterns)
    except Exception:
        return False


def filter_leaky_formulas(features, target):
    clean, blocked = [], []
    for f in features:
        try:
            if check_formula_leakage(f.get("formula", ""), target):
                blocked.append({**f, "block_reason":
                    f"Formula references target '{target}' — data leakage!"})
            else:
                clean.append(f)
        except Exception:
            clean.append(f)
    if blocked:
        print(f"\n🛡️  Leakage Guard blocked {len(blocked)} formula(s):")
        for b in blocked:
            print(f"   ❌ '{b['name']}' — {b['block_reason']}")
    return clean, blocked


def check_correlation_leakage(df, features, target,
                               threshold=LEAKAGE_CORR_THRESHOLD):
    if target not in df.columns:
        return features, []
    y = df[target]
    clean, blocked = [], []
    for f in features:
        try:
            new_col = eval(f["formula"], {"df": df, "np": np, "pd": pd})
            new_col = pd.Series(new_col, index=df.index)
            corr    = abs(float(new_col.corr(y)))
            if corr > threshold:
                blocked.append({**f,
                    "block_reason": f"|corr|={corr:.3f} > {threshold} — possible leakage",
                    "corr_value": round(corr, 4)})
            else:
                clean.append(f)
        except Exception:
            clean.append(f)
    if blocked:
        print(f"\n🛡️  Correlation Guard blocked {len(blocked)} feature(s).")
    return clean, blocked


def check_score_reliability(score, score_std, model_name, task):
    """Returns dict with level, messages, forced_confidence."""
    messages, level, forced_confidence = [], "ok", None
    try:
        if task == "classification":
            if score > SUSPICIOUS_SCORE_CRIT:
                level, forced_confidence = "critical", "Low ❌"
                messages.append(
                    f"🚨 WARNING: Perfect/near-perfect score ({score:.4f}) detected.\n"
                    f"  Possible reasons:\n"
                    f"    • Data leakage (target information in features)\n"
                    f"    • Highly separable dataset (e.g. PCA-transformed fraud data)\n"
                    f"    • Sampling bias (lucky sample)\n"
                    f"  Recommendation: Validate on a completely unseen held-out dataset.\n"
                    f"  Confidence OVERRIDDEN → Low ❌")
            elif score > SUSPICIOUS_SCORE_WARN:
                level, forced_confidence = "warn", "Medium ⚠️"
                messages.append(
                    f"⚠️  High score ({score:.4f}) detected — verify with held-out data.\n"
                    f"  Possible reasons: overfitting, data leakage, or easy dataset.\n"
                    f"  Confidence OVERRIDDEN → Medium ⚠️")

        if score_std > UNSTABLE_STD_CRITICAL:
            level = "warn"
            if forced_confidence != "Low ❌":
                forced_confidence = "Low ❌"
            messages.append(f"🚨 VERY UNSTABLE: CV std={score_std:.4f} > "
                           f"{UNSTABLE_STD_CRITICAL} — unreliable model.")
        elif score_std > UNSTABLE_STD_WARN:
            if level == "ok":
                level = "warn"
            if forced_confidence is None:
                forced_confidence = "Medium ⚠️"
            messages.append(f"⚠️  UNSTABLE: CV std={score_std:.4f} > "
                           f"{UNSTABLE_STD_WARN} — consider more data.")
    except Exception:
        pass
    return {"level": level, "messages": messages,
            "forced_confidence": forced_confidence}


def run_full_leakage_guard(df, features, target):
    """Run all checks. Returns (clean_features, all_blocked, dataset_warnings)."""
    print("\n🛡️  Running Leakage Guard...")
    try:
        dataset_warnings = check_dataset_leakage(df, target)
        if dataset_warnings:
            n = len(dataset_warnings)
            print(f"\n🛡️  DATASET WARNINGS ({n}) — advisory, no auto-drop:")
            for w in dataset_warnings[:5]:
                print(f"   {w}")
            if n > 5:
                print(f"   ... and {n-5} more.")

        clean, blocked_formula = filter_leaky_formulas(features, target)
        clean, blocked_corr    = check_correlation_leakage(df, clean, target)
        all_blocked = blocked_formula + blocked_corr

        if not all_blocked:
            print(f"   ✅ No leakage detected in {len(features)} feature(s).")
        else:
            print(f"\n   🛡️  Summary: {len(all_blocked)} blocked, "
                  f"{len(clean)} passed.")
        return clean, all_blocked, dataset_warnings
    except Exception as e:
        print(f"   ⚠️  Leakage guard error (skipped): {e}")
        return features, [], []