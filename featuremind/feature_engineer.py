"""
feature_engineer.py — Enterprise Feature Engineering Engine
=============================================================
featuremind v1.1.0

KEY UPGRADE — Smart Domain Auto-Detection:
  Previously: domain templates only fired when exact column names matched.
  Now: system scans column names with fuzzy keyword matching, so ANY dataset
       with columns like "monthly_charge", "tenure_months", "creatinine" etc.
       automatically gets domain-specific features.

KEY UPGRADE — Correlation-Based Feature Pre-Scoring:
  Candidate features are pre-scored by their Pearson correlation with the
  target BEFORE running the expensive CV evaluation. Only the top-scoring
  candidates are evaluated — faster and higher-quality suggestions.

KEY UPGRADE — Fraud / Anomaly Detection domain:
  For datasets like creditcard.csv with PCA components (V1-V28),
  adds statistical anomaly scores and deviation features.

12 Layers:
  Layer 1  — Auto-detected Domain (Telecom · Medical · Real Estate · Finance · HR · Fraud)
  Layer 2  — Interactions (top correlated pairs)
  Layer 3  — Ratios
  Layer 4  — Log transforms
  Layer 5  — Polynomial (x², x³, √x)
  Layer 6  — Quantile binning (quartile + decile)
  Layer 7  — Delta / difference
  Layer 8  — Outlier flags (z-score)
  Layer 9  — NLP / Text features
  Layer 10 — Percentile rank
  Layer 11 — Target-guided combinations (high-corr pairs × low-corr pairs)
  Layer 12 — Statistical aggregations (abs, sign, clip)
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _safe_eval(df: pd.DataFrame, formula: str):
    """Evaluate formula safely. Returns Series or None if invalid/constant."""
    try:
        result = eval(formula, {"df": df, "np": np, "pd": pd})
        result = pd.Series(result, index=df.index)
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
        if result.std() == 0:
            return None
        return result
    except Exception:
        return None


def _f(name, formula, business, reason, layer="generic", score=0.0):
    return {"name": name, "formula": formula,
            "business": business, "reason": reason,
            "layer": layer, "_prescore": score}


def _col_match(cols: set, *keywords) -> str | None:
    """Find first column whose name contains any of the given keywords (case-insensitive)."""
    for col in cols:
        cl = col.lower().replace("_", "").replace(" ", "")
        for kw in keywords:
            if kw.lower() in cl:
                return col
    return None


def _prescore(candidate_series: pd.Series, target_series: pd.Series) -> float:
    """Compute |Pearson correlation| between a candidate feature and target."""
    try:
        corr = float(abs(candidate_series.corr(target_series)))
        return corr if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Domain detector
# ══════════════════════════════════════════════════════════════════════════════

def _detect_domain(cols: set) -> str:
    """
    Auto-detect dataset domain from column names.
    Returns one of: 'telecom', 'medical', 'realestate', 'finance', 'hr',
                    'fraud', 'generic'
    """
    col_str = " ".join(c.lower() for c in cols)

    # Fraud / anomaly (PCA features like V1..V28 + Amount + Class)
    pca_cols = sum(1 for c in cols if c.upper().startswith("V") and c[1:].isdigit())
    if pca_cols >= 5 and _col_match(cols, "amount", "class", "fraud"):
        return "fraud"

    # Medical
    if any(kw in col_str for kw in ["creatinine", "ejection", "serum", "platelet",
                                      "ejection_fraction", "death_event", "anaemia"]):
        return "medical"

    # Telecom / Subscription
    if any(kw in col_str for kw in ["churn", "monthlycharges", "tenure", "totalcharges",
                                      "monthly_charges", "contract"]):
        return "telecom"

    # HR / Attrition
    if any(kw in col_str for kw in ["attrition", "monthlyincome", "yearsatcompany",
                                      "jobsatisfaction", "overtime", "worklife"]):
        return "hr"

    # Finance / Credit
    if any(kw in col_str for kw in ["loan", "credit", "interest", "annual_inc",
                                      "default", "int_rate", "installment"]):
        return "finance"

    # Real estate
    if any(kw in col_str for kw in ["grlivarea", "overallqual", "totalbsmtsf",
                                      "yearbuilt", "bathroom", "carpet", "furnish"]):
        return "realestate"

    return "generic"


# ══════════════════════════════════════════════════════════════════════════════
# Domain feature generators
# ══════════════════════════════════════════════════════════════════════════════

def _domain_telecom(cols, df, add):
    """Features for Telecom / Subscription / Churn datasets."""
    tc = _col_match(cols, "totalcharges", "total_charges", "totalcharge")
    te = _col_match(cols, "tenure", "months", "duration")
    mc = _col_match(cols, "monthlycharges", "monthly_charges", "monthlyfee", "monthlycost")

    if tc and te:
        add(_f("avg_monthly_spend", f"df['{tc}'] / (df['{te}'] + 1e-5)",
               "Average monthly spend", "Higher monthly spend → higher churn risk", "domain"))
    if mc:
        add(_f("high_charge_flag",
               f"(df['{mc}'] > df['{mc}'].mean()).astype(int)",
               "Above-average charge flag", "Above-average payers more likely to churn", "domain"))
    if mc and tc and te:
        add(_f("charge_growth_rate",
               f"df['{mc}'] / (df['{tc}'] / (df['{te}'] + 1e-5) + 1e-5)",
               "Charge growth rate", "Rising charges signal dissatisfaction", "domain"))
    if mc and te:
        add(_f("monthly_charge_x_tenure",
               f"df['{mc}'] * df['{te}']",
               "Revenue lifetime value", "Total revenue proxy — low tenure + high charge = at-risk", "domain"))
    if tc and mc:
        add(_f("charge_ratio",
               f"df['{mc}'] / (df['{tc}'] + 1e-5)",
               "Current vs historical charge ratio",
               "Sudden increase in charge rate predicts churn", "domain"))


def _domain_medical(cols, df, add):
    """Features for medical / healthcare datasets."""
    ef  = _col_match(cols, "ejectionfraction", "ejection_fraction", "ef")
    sc  = _col_match(cols, "serumcreatinine", "serum_creatinine", "creatinine")
    ss  = _col_match(cols, "serumsodium", "serum_sodium", "sodium")
    plt = _col_match(cols, "platelet", "platelets")
    cpk = _col_match(cols, "creatinephospho", "creatinine_phospho", "cpk")
    age = _col_match(cols, "age")
    tm  = _col_match(cols, "time", "followup", "follow_up")

    if ef and sc:
        add(_f("heart_risk_score", f"df['{sc}'] / (df['{ef}'] + 1e-5)",
               "Heart failure risk ratio", "High creatinine + low ejection fraction → death risk", "domain"))
    if ss and sc:
        add(_f("electrolyte_balance", f"df['{ss}'] / (df['{sc}'] + 1e-5)",
               "Electrolyte balance", "Sodium/creatinine imbalance is a cardiac indicator", "domain"))
    if plt and cpk:
        add(_f("platelet_cpk_ratio", f"df['{plt}'] / (df['{cpk}'] + 1e-5)",
               "Platelet-CPK ratio", "Combined cardiac blood marker ratio", "domain"))
    if age and tm:
        add(_f("age_time_ratio", f"df['{age}'] / (df['{tm}'] + 1e-5)",
               "Age vs follow-up time", "Older patients with shorter follow-up → higher risk", "domain"))
    if ef and sc and ss:
        add(_f("cardiac_composite",
               f"df['{sc}'] / (df['{ef}'] + 1e-5) + (100 - df['{ss}'])",
               "Cardiac composite risk", "Combined cardiac failure indicator", "domain"))


def _domain_realestate(cols, df, add):
    """Features for real estate / property datasets."""
    price = _col_match(cols, "price", "salesprice", "sale_price", "amount")
    area  = _col_match(cols, "grlivarea", "carpet", "superarea", "builtup", "area")
    qual  = _col_match(cols, "overallqual", "quality", "grade")
    bath  = _col_match(cols, "bathroom", "bath")
    yr    = _col_match(cols, "yearbuilt", "year_built", "builtyear")
    sold  = _col_match(cols, "yrsold", "year_sold", "soldon")
    bsmt  = _col_match(cols, "totalbsmtsf", "basement")

    if price and area:
        add(_f("price_per_area", f"df['{price}'] / (df['{area}'] + 1e-5)",
               "Price per unit area", "Key real-estate efficiency metric", "domain"))
    if area and qual:
        add(_f("quality_area_score", f"df['{area}'] * df['{qual}']",
               "Quality × area score", "Strongest house price predictor", "domain"))
    if area and bath:
        add(_f("area_per_bathroom", f"df['{area}'] / (df['{bath}'] + 1e-5)",
               "Area per bathroom", "Efficiency of space use", "domain"))
    if yr and sold:
        add(_f("house_age_at_sale", f"df['{sold}'] - df['{yr}']",
               "House age at sale", "Newer houses command higher prices", "domain"))
    if bsmt and area:
        add(_f("total_living_area", f"df['{bsmt}'] + df['{area}']",
               "Total living area", "Combined floor area drives value", "domain"))


def _domain_finance(cols, df, add):
    """Features for finance / credit / banking datasets."""
    loan  = _col_match(cols, "loan_amnt", "loanamount", "loan_amount", "principal")
    inc   = _col_match(cols, "annual_inc", "income", "annual_income", "salary")
    rate  = _col_match(cols, "int_rate", "interest_rate", "interest", "rate")
    cred  = _col_match(cols, "credit_score", "fico", "credit", "creditscore")
    inst  = _col_match(cols, "installment", "emi", "monthly_payment")

    if loan and inc:
        add(_f("debt_to_income", f"df['{loan}'] / (df['{inc}'] + 1e-5)",
               "Debt-to-income ratio", "High DTI strongly predicts default", "domain"))
    if rate and loan:
        add(_f("total_interest", f"df['{rate}'] * df['{loan}'] / 100",
               "Total interest payable", "Higher interest → higher default probability", "domain"))
    if cred and loan:
        add(_f("credit_per_loan", f"df['{cred}'] / (df['{loan}'] + 1e-5)",
               "Credit quality per loan", "Higher ratio → lower default risk", "domain"))
    if inst and inc:
        add(_f("payment_burden", f"df['{inst}'] / (df['{inc}'] / 12 + 1e-5)",
               "Monthly payment burden", "EMI as fraction of monthly income", "domain"))
    if loan and cred:
        add(_f("loan_per_credit", f"df['{loan}'] / (df['{cred}'] + 1e-5)",
               "Loan-to-credit ratio", "Higher ratio indicates over-leveraging", "domain"))


def _domain_hr(cols, df, add):
    """Features for HR / employee attrition datasets."""
    inc   = _col_match(cols, "monthlyincome", "monthly_income", "salary", "income")
    age   = _col_match(cols, "age")
    yac   = _col_match(cols, "yearsatcompany", "years_at_company", "tenure")
    twy   = _col_match(cols, "totalworkingyears", "total_working_years", "experience")
    dist  = _col_match(cols, "distancefromhome", "distance_from_home", "commute")
    jsat  = _col_match(cols, "jobsatisfaction", "job_satisfaction", "satisfaction")
    ot    = _col_match(cols, "overtime", "over_time")
    wlb   = _col_match(cols, "worklifebalance", "work_life_balance")

    if inc and age:
        add(_f("income_per_age", f"df['{inc}'] / (df['{age}'] + 1e-5)",
               "Income relative to age", "Underpaid employees tend to leave", "domain"))
    if yac and twy:
        add(_f("company_tenure_ratio", f"df['{yac}'] / (df['{twy}'] + 1e-5)",
               "Company loyalty ratio", "Low ratio → job-hopping tendency", "domain"))
    if dist and jsat:
        add(_f("commute_satisfaction_burden", f"df['{dist}'] / (df['{jsat}'] + 1e-5)",
               "Commute vs satisfaction", "High commute + low satisfaction → attrition", "domain"))
    if ot and wlb:
        add(_f("overtime_stress", f"df['{ot}'] / (df['{wlb}'] + 1e-5)",
               "Overtime stress index", "High overtime vs work-life balance → burnout", "domain"))
    if inc and yac:
        add(_f("income_growth_rate", f"df['{inc}'] / (df['{yac}'] + 1e-5)",
               "Income per year at company", "Slow income growth → higher attrition", "domain"))


def _domain_fraud(cols, df, add, top):
    """Features for fraud/anomaly detection (PCA-based like creditcard.csv)."""
    # Statistical anomaly scores from PCA components
    pca_cols = sorted([c for c in cols if c.upper().startswith("V")
                       and c[1:].isdigit()], key=lambda x: int(x[1:]))[:10]

    if pca_cols:
        # L2 norm of top PCA features (distance from origin in PCA space)
        pca_formula = " + ".join(f"df['{c}']**2" for c in pca_cols[:5])
        add(_f("pca_l2_norm", f"np.sqrt({pca_formula})",
               "L2 norm of PCA components", "Distance from normal in PCA space → anomaly score", "domain"))

        # Max absolute deviation across PCA features
        max_formula = "np.abs(pd.concat([" + ", ".join(f"df['{c}']" for c in pca_cols[:5]) + "], axis=1)).max(axis=1)"
        add(_f("pca_max_deviation", max_formula,
               "Max PCA deviation", "Largest single-component anomaly signal", "domain"))

    amt = _col_match(cols, "amount", "amt")
    if amt:
        add(_f("log_amount_std", f"(np.log1p(df['{amt}']) - np.log1p(df['{amt}']).mean()) / (np.log1p(df['{amt}']).std() + 1e-5)",
               "Standardised log amount", "Normalised transaction amount z-score", "domain"))
        add(_f("amount_quartile", f"pd.qcut(df['{amt}'], q=4, labels=False, duplicates='drop')",
               "Amount quartile", "Which quartile is this transaction in?", "domain"))

    tm = _col_match(cols, "time")
    if tm:
        add(_f("time_of_day_sin", f"np.sin(2 * np.pi * df['{tm}'] / 86400)",
               "Cyclical time encoding (sin)", "Time-of-day pattern for fraud detection", "domain"))
        add(_f("time_of_day_cos", f"np.cos(2 * np.pi * df['{tm}'] / 86400)",
               "Cyclical time encoding (cos)", "Complementary time-of-day signal", "domain"))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def generate_feature_suggestions(df: pd.DataFrame, target: str,
                                  col_types: dict = None) -> list:
    """
    Generate and validate feature suggestions across 12 layers.
    Uses correlation pre-scoring to prioritise the best candidates.

    Returns list of validated suggestion dicts sorted by pre-score.
    """
    suggestions = []
    seen        = set()

    def add(s):
        key = s["formula"].replace(" ", "")
        if key not in seen:
            seen.add(key)
            suggestions.append(s)

    # ── Find top correlated columns with target ───────────────────────────────
    num_df = df.select_dtypes(include="number")
    if target not in num_df.columns or num_df.shape[1] < 2:
        return []

    corr = (
        num_df.corr()[target]
        .abs()
        .drop(labels=[target], errors="ignore")
        .dropna()
        .sort_values(ascending=False)
    )
    top  = corr.index[:6].tolist()
    cols = set(df.columns)
    y    = df[target]

    if not top:
        return []

    # ── Detect domain ─────────────────────────────────────────────────────────
    domain = _detect_domain(cols)
    print(f"🔍  Detected domain: {domain.upper()}")

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1 — Auto-detected Domain Features
    # ══════════════════════════════════════════════════════════════════════════
    if domain == "telecom":
        _domain_telecom(cols, df, add)
    elif domain == "medical":
        _domain_medical(cols, df, add)
    elif domain == "realestate":
        _domain_realestate(cols, df, add)
    elif domain == "finance":
        _domain_finance(cols, df, add)
    elif domain == "hr":
        _domain_hr(cols, df, add)
    elif domain == "fraud":
        _domain_fraud(cols, df, add, top)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 2 — Interactions (top-6 pairs)
    # ══════════════════════════════════════════════════════════════════════════
    for i in range(min(5, len(top))):
        for j in range(i + 1, min(6, len(top))):
            c1, c2 = top[i], top[j]
            add(_f(f"{c1}_x_{c2}", f"df['{c1}'] * df['{c2}']",
                   f"Interaction: {c1} × {c2}",
                   f"Combined effect of {c1} and {c2} on {target}", "interaction"))

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 3 — Ratios
    # ══════════════════════════════════════════════════════════════════════════
    for i in range(min(4, len(top))):
        for j in range(i + 1, min(5, len(top))):
            c1, c2 = top[i], top[j]
            add(_f(f"{c1}_div_{c2}", f"df['{c1}'] / (df['{c2}'] + 1e-5)",
                   f"Ratio: {c1} ÷ {c2}",
                   f"Relative magnitude of {c1} vs {c2}", "ratio"))

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 4 — Log transforms
    # ══════════════════════════════════════════════════════════════════════════
    for col in top[:3]:
        if col in cols and df[col].min() >= 0:
            add(_f(f"log_{col}", f"np.log1p(df['{col}'])",
                   f"Log-transform of {col}",
                   f"Reduces right skewness — helps linear models", "transform"))

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 5 — Polynomial & Root
    # ══════════════════════════════════════════════════════════════════════════
    for col in top[:2]:
        add(_f(f"{col}_sq", f"df['{col}'] ** 2", f"{col} squared",
               f"Non-linear signal of {col}", "polynomial"))
        add(_f(f"{col}_cb", f"df['{col}'] ** 3", f"{col} cubed",
               f"Strong non-linear signal of {col}", "polynomial"))
        if col in cols and df[col].min() >= 0:
            add(_f(f"sqrt_{col}", f"np.sqrt(df['{col}'])", f"√{col}",
                   f"Soft compression of {col}", "transform"))

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 6 — Quantile Binning
    # ══════════════════════════════════════════════════════════════════════════
    for col in top[:3]:
        if col in cols and pd.api.types.is_numeric_dtype(df[col]):
            add(_f(f"{col}_quartile",
                   f"pd.qcut(df['{col}'], q=4, labels=False, duplicates='drop')",
                   f"Quartile bucket of {col}", f"Threshold effects in {col}", "binning"))
            add(_f(f"{col}_decile",
                   f"pd.qcut(df['{col}'], q=10, labels=False, duplicates='drop')",
                   f"Decile bucket of {col}", f"Fine-grained bucket of {col}", "binning"))

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 7 — Delta / Difference
    # ══════════════════════════════════════════════════════════════════════════
    for i in range(min(3, len(top))):
        for j in range(i + 1, min(4, len(top))):
            c1, c2 = top[i], top[j]
            add(_f(f"{c1}_minus_{c2}", f"df['{c1}'] - df['{c2}']",
                   f"Difference: {c1} − {c2}",
                   f"Net change between {c1} and {c2}", "delta"))

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 8 — Outlier Flags
    # ══════════════════════════════════════════════════════════════════════════
    for col in top[:3]:
        if col in cols and pd.api.types.is_numeric_dtype(df[col]):
            add(_f(f"{col}_is_outlier",
                   (f"(((df['{col}'] - df['{col}'].mean()) / "
                    f"(df['{col}'].std() + 1e-5)).abs() > 3).astype(int)"),
                   f"Outlier flag for {col}", f"z-score > 3 flag", "outlier"))

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 9 — NLP / Text features
    # ══════════════════════════════════════════════════════════════════════════
    text_cols = []
    if col_types:
        text_cols = [c for c, t in col_types.items()
                     if t == "text" and c in cols and c != target]
    for col in text_cols[:2]:
        safe = col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        add(_f(f"{safe}_word_count", f"df['{col}'].astype(str).str.split().str.len()",
               f"Word count of {col}", "Content richness indicator", "nlp"))
        add(_f(f"{safe}_char_count", f"df['{col}'].astype(str).str.len()",
               f"Char count of {col}", "Text length proxy", "nlp"))

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 10 — Percentile Rank
    # ══════════════════════════════════════════════════════════════════════════
    for col in top[:2]:
        if col in cols and pd.api.types.is_numeric_dtype(df[col]):
            add(_f(f"{col}_pct_rank", f"df['{col}'].rank(pct=True)",
                   f"Percentile rank of {col}",
                   f"Relative position in {col} distribution", "rank"))

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 11 — Target-Guided Combinations
    # (High-corr feature × low-corr feature — cross interactions)
    # ══════════════════════════════════════════════════════════════════════════
    if len(top) >= 4:
        high_corr = top[:2]   # strongest predictors
        low_corr  = top[3:5]  # weaker but complementary predictors
        for hc in high_corr:
            for lc in low_corr:
                add(_f(f"{hc}_x_{lc}_guided",
                       f"df['{hc}'] * df['{lc}']",
                       f"Target-guided: {hc} × {lc}",
                       f"Cross-signal: strong predictor × weaker signal of {target}", "interaction"))

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 12 — Statistical Aggregations
    # ══════════════════════════════════════════════════════════════════════════
    for col in top[:2]:
        if col in cols and pd.api.types.is_numeric_dtype(df[col]):
            add(_f(f"{col}_abs", f"df['{col}'].abs()",
                   f"Absolute value of {col}", f"Captures magnitude regardless of sign", "stat"))
            add(_f(f"{col}_sign",
                   f"np.sign(df['{col}'])",
                   f"Sign of {col}", f"Direction of {col}", "stat"))

    # ── Validate + pre-score ──────────────────────────────────────────────────
    valid = []
    for s in suggestions:
        series = _safe_eval(df, s["formula"])
        if series is not None:
            s["_prescore"] = _prescore(series, y)
            valid.append(s)

    # Sort by pre-score (correlation with target) — best candidates first
    valid.sort(key=lambda x: x["_prescore"], reverse=True)
    return valid