"""
analyzer.py — Enterprise Data Loading, Cleaning & Quality Reporting
====================================================================
featuremind v1.0.0

KEY NEW FEATURE — Multi-file support:
  ✅ analyze_multiple([file1, file2, ...]) — join multiple CSV files automatically
     Works by: detecting common join key → merging → treating as one dataset
  ✅ Single file: fm.analyze("data.csv")
  ✅ Multiple files: fm.analyze(["orders.csv", "customers.csv", "products.csv"])

All v0.5/v0.6 features retained:
  ✅ Chunked streaming for files of any size (millions of rows)
  ✅ Smart type inference: numeric/binary/ordinal/categorical/datetime/text/id
  ✅ Auto skewness fix (log1p when |skew| > 2)
  ✅ Advanced imputation: median (numeric), mode (categorical)
  ✅ Datetime feature extraction
  ✅ IQR outlier detection & reporting
  ✅ Full data quality audit dict

Bug fixes from v0.5 → v0.6:
  ✅ _load_csv: sample uses min(N, len(df)) — fixes "larger than population" crash
  ✅ pandas 3.x StringDtype handled via _is_str()
  ✅ infer_datetime_format removed (deprecated in pandas 2+)
"""

import re
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

VERSION          = "3.1.1"
MAX_SAMPLE_SIZE  = 5000
CHUNK_SIZE       = 100_000
MIN_VIABLE_ROWS  = 10
HIGH_CARD_RATIO  = 0.10
NULL_COL_THRESH  = 0.80  # v3.1: raised from 0.60 to save more columns
SKEW_THRESHOLD   = 2.0
TEXT_WORD_THRESH = 5

TARGET_KEYWORDS = {
    "target", "label", "output", "churn", "y",
    "price", "salesprice", "sale_price",
    "death_event", "survived", "outcome",
    "class", "result", "fraud", "default",
    "readmitted", "is_fraud", "loan_status",
    "attrition", "diagnosis", "sentiment",
    "conversion", "clicked", "purchased", "defaulted",
    "revenue", "sales", "profit", "loss",
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _is_str(s: pd.Series) -> bool:
    return s.dtype == "object" or pd.api.types.is_string_dtype(s)


def _to_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"[₹$€£,\s]", "", regex=True)

    def _expand(val):
        val = str(val).strip()
        m = re.match(r"^([\d.]+)\s*[Cc][Rr]?$", val)
        if m: return float(m.group(1)) * 1e7
        m = re.match(r"^([\d.]+)\s*[Ll][Aa][Cc]?[Hh]?$", val)
        if m: return float(m.group(1)) * 1e5
        return val

    s = s.apply(_expand)
    s = pd.Series(s).astype(str).str.replace(
        r"[/\s]*(sq\.?ft|sq\.?m|sqft|BHK|bhk|[-a-zA-Z]+)$", "", regex=True
    ).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _try_dt(val: str) -> bool:
    try:
        pd.to_datetime(val)
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Multi-file loader  ← NEW in v1.0
# ══════════════════════════════════════════════════════════════════════════════

def _find_join_key(dfs: list) -> str | None:
    """
    Find the best common column to join multiple DataFrames on.
    Prefers: 'id', 'customer_id', 'user_id', 'order_id', etc.
    Falls back to any common string/int column.
    """
    id_keywords = {"id", "key", "uid", "uuid", "code", "no", "num", "number"}
    all_cols = [set(df.columns.str.lower()) for df in dfs]
    common   = set.intersection(*all_cols) if all_cols else set()

    # Priority 1: column containing "id" keyword
    for col_lower in common:
        if any(kw in col_lower for kw in id_keywords):
            # Find the actual column name (original case)
            for df in dfs:
                for col in df.columns:
                    if col.lower() == col_lower:
                        return col
    # Priority 2: any common column
    if common:
        col_lower = next(iter(common))
        for df in dfs:
            for col in df.columns:
                if col.lower() == col_lower:
                    return col
    return None


def load_and_merge(files: list, target: str = None) -> tuple:
    """
    Load multiple CSV files and merge them into one DataFrame.
    Automatically detects the join key column.

    Returns (merged_df, total_rows, file_info_list)
    """
    print(f"📂 Loading {len(files)} file(s)...")
    dfs        = []
    file_infos = []
    total_rows = 0

    for i, file in enumerate(files):
        try:
            df_i, rows_i = _load_single_csv(file)
            dfs.append(df_i)
            total_rows += rows_i
            file_infos.append({"file": file, "rows": rows_i, "cols": df_i.shape[1]})
            print(f"   [{i+1}] {file}: {rows_i:,} rows × {df_i.shape[1]} cols")
        except Exception as e:
            print(f"   ⚠️  Failed to load '{file}': {e}")

    if not dfs:
        raise ValueError("No files could be loaded.")

    if len(dfs) == 1:
        return dfs[0], total_rows, file_infos

    # Find join key
    join_key = _find_join_key(dfs)
    if join_key:
        print(f"🔗 Joining on common key: '{join_key}'")
        merged = dfs[0]
        for df_i in dfs[1:]:
            try:
                if join_key in df_i.columns:
                    merged = pd.merge(merged, df_i, on=join_key, how="left",
                                      suffixes=("", f"_{df_i.columns[0]}"))
                else:
                    # No join key in this file — concatenate columns if same length
                    if len(df_i) == len(merged):
                        merged = pd.concat(
                            [merged.reset_index(drop=True),
                             df_i.reset_index(drop=True)], axis=1)
            except Exception as e:
                print(f"   ⚠️  Merge failed: {e} — concatenating instead")
                merged = pd.concat([merged.reset_index(drop=True),
                                    df_i.reset_index(drop=True)], axis=1)
    else:
        # No common key: concatenate column-wise if same length, else row-wise
        if all(len(d) == len(dfs[0]) for d in dfs):
            print(f"🔗 No join key found — merging columns side by side")
            merged = pd.concat([d.reset_index(drop=True) for d in dfs], axis=1)
        else:
            print(f"🔗 No join key found — stacking rows")
            merged = pd.concat(dfs, ignore_index=True)

    print(f"✅ Merged dataset: {merged.shape[0]:,} rows × {merged.shape[1]} cols")
    return merged, total_rows, file_infos


# ══════════════════════════════════════════════════════════════════════════════
# Single-file CSV loader  ← BUG FIXED: sample uses min(N, len)
# ══════════════════════════════════════════════════════════════════════════════

def _load_single_csv(file: str) -> tuple:
    """Load one CSV. Returns (df, total_row_count)."""
    total = 0
    try:
        for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE, usecols=[0]):
            total += len(chunk)
    except Exception:
        total = -1

    # Small enough — load fully
    if 0 < total <= MAX_SAMPLE_SIZE * 2:
        return pd.read_csv(file), total

    # Large file — stream-sample
    if total > MAX_SAMPLE_SIZE * 2:
        print(f"⚡ Large file ({total:,} rows) — using {MAX_SAMPLE_SIZE:,}-row sample for fast analysis.")
        print(f"   💡 Tip: Analysis uses {MAX_SAMPLE_SIZE:,} rows. Pipeline training uses full data.")
        chunks    = []
        collected = 0
        for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):
            n_take = min(500, len(chunk))
            chunks.append(chunk.sample(n_take, random_state=42))
            collected += n_take
            if collected >= MAX_SAMPLE_SIZE:
                break
        combined = pd.concat(chunks, ignore_index=True)
        # FIX: never ask for more than we have
        n_sample = min(MAX_SAMPLE_SIZE, len(combined))
        return combined.sample(n_sample, random_state=42).reset_index(drop=True), total

    return pd.read_csv(file), total


# ══════════════════════════════════════════════════════════════════════════════
# Type inference
# ══════════════════════════════════════════════════════════════════════════════

def _infer_types(df: pd.DataFrame, target: str) -> dict:
    tmap = {}
    for col in df.columns:
        if col == target:
            tmap[col] = "target"; continue
        s        = df[col]
        n_unique = s.nunique(dropna=True)
        n_total  = max(len(s.dropna()), 1)

        if s.dtype in (np.int64, np.int32) and n_unique / n_total > 0.90:
            tmap[col] = "id"; continue
        if pd.api.types.is_numeric_dtype(s):
            tmap[col] = ("binary" if n_unique == 2 else
                         "ordinal" if n_unique <= 10 else "numeric"); continue
        if pd.api.types.is_datetime64_any_dtype(s):
            tmap[col] = "datetime"; continue
        if _is_str(s):
            sample = s.dropna().head(20).astype(str).tolist()
            if sample and sum(1 for v in sample if _try_dt(v)) / len(sample) > 0.80:
                tmap[col] = "datetime"; continue
            try:
                avg_w = s.dropna().astype(str).str.split().str.len().mean()
                if avg_w and avg_w > TEXT_WORD_THRESH:
                    tmap[col] = "text"; continue
            except Exception:
                pass
            if n_unique / n_total > HIGH_CARD_RATIO: tmap[col] = "text"
            elif n_unique == 2:                       tmap[col] = "binary"
            else:                                     tmap[col] = "categorical"
            continue
        tmap[col] = "unknown"
    return tmap


# ══════════════════════════════════════════════════════════════════════════════
# Column cleaning
# ══════════════════════════════════════════════════════════════════════════════

def _drop_useless(df, target, q):
    drop = []
    for col in df.columns:
        if col == target: continue
        nf = df[col].isna().mean()
        if nf > NULL_COL_THRESH:
            drop.append((col, f"{nf*100:.0f}% null"))
            q["dropped_columns"].append(f"{col} ({nf*100:.0f}% null)"); continue
        nu = df[col].nunique(dropna=True)
        if nu <= 1:
            drop.append((col, "constant"))
            q["dropped_columns"].append(f"{col} (constant)"); continue
        if df[col].dtype in (np.int64, np.int32) and nu / max(len(df), 1) > 0.90:
            drop.append((col, "ID-like"))
            q["dropped_columns"].append(f"{col} (ID-like)")
    if drop:
        print(f"🗑️  Dropped: {', '.join(f'{c}({r})' for c, r in drop[:5])}"
              + (f" +{len(drop)-5} more" if len(drop) > 5 else ""))
        df = df.drop(columns=[c for c, _ in drop])
    return df


def _convert_mixed(df, target, q):
    for col in df.columns:
        if col == target or not _is_str(df[col]): continue
        conv = _to_numeric(df[col])
        rate = conv.notna().mean()
        if rate >= 0.30:
            df[col] = conv
            q["converted_columns"].append(f"{col} ({rate*100:.0f}% parsed)")
            print(f"🔧  Converted '{col}' → numeric ({rate*100:.0f}% parsed)")
    return df


def _drop_high_card(df, target, q):
    drop = [c for c in df.columns if c != target and _is_str(df[c])
            and df[c].nunique() / max(len(df), 1) > HIGH_CARD_RATIO]
    if drop:
        print(f"🗑️  Dropped high-cardinality text: {drop[:5]}"
              + (f" +{len(drop)-5} more" if len(drop) > 5 else ""))
        for c in drop:
            q["dropped_columns"].append(f"{c} (high-cardinality text)")
        df = df.drop(columns=drop)
    return df


def _extract_datetimes(df, col_types, q):
    dt_cols = [c for c, t in col_types.items() if t == "datetime" and c in df.columns]
    done    = []
    for col in dt_cols:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.isna().mean() > 0.50: continue
            p = col.replace(" ", "_").replace("(", "").replace(")", "")
            df[f"{p}_year"]      = parsed.dt.year.fillna(0).astype(int)
            df[f"{p}_month"]     = parsed.dt.month.fillna(0).astype(int)
            df[f"{p}_day"]       = parsed.dt.day.fillna(0).astype(int)
            df[f"{p}_dayofweek"] = parsed.dt.dayofweek.fillna(0).astype(int)
            df[f"{p}_is_weekend"]= parsed.dt.dayofweek.isin([5, 6]).astype(int)
            df[f"{p}_quarter"]   = parsed.dt.quarter.fillna(0).astype(int)
            if parsed.dt.hour.max() > 0:
                df[f"{p}_hour"] = parsed.dt.hour.fillna(0).astype(int)
            df.drop(columns=[col], inplace=True)
            done.append(col)
        except Exception:
            pass
    if done:
        q["datetime_extracted"] = done
        print(f"📅  Datetime features extracted: {done}")
    return df


def _impute(df, target, q):
    done = []
    for col in df.columns:
        if col == target: continue
        n_null = df[col].isna().sum()
        if n_null == 0: continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill    = df[col].median()
            df[col] = df[col].fillna(fill)
            done.append(f"{col} (median={fill:.2f}, {n_null} cells)")
        else:
            modes = df[col].mode()
            if len(modes):
                df[col] = df[col].fillna(modes[0])
                done.append(f"{col} (mode='{modes[0]}', {n_null} cells)")
    q["imputed_columns"] = done
    if done:
        print(f"🔄  Imputed {len(done)} column(s) with median/mode.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Target handling
# ══════════════════════════════════════════════════════════════════════════════

def detect_target(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower().strip().replace(" ", "_") in TARGET_KEYWORDS:
            if df[col].notna().sum() > 0:
                return col
    for col in reversed(df.columns.tolist()):
        if df[col].notna().sum() > MIN_VIABLE_ROWS:
            return col
    return df.columns[-1]


def _encode_target(df, target):
    col = df[target]
    if _is_str(col):
        mapping = {"yes": 1, "no": 0, "true": 1, "false": 0}
        mapped  = col.astype(str).str.strip().str.lower().map(mapping)
        if mapped.notna().all():
            df = df.copy(); df[target] = mapped
        else:
            num = _to_numeric(col)
            if num.notna().mean() >= 0.30:
                df = df.copy(); df[target] = num
            else:
                df = df.copy()
                df[target] = pd.Categorical(col).codes.astype(float)
                df.loc[df[target] < 0, target] = np.nan

    df[target] = pd.to_numeric(df[target], errors="coerce")
    before  = len(df)
    df      = df.dropna(subset=[target]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"⚠️  Dropped {dropped} rows with missing/invalid target values.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Skew fix + outlier detection
# ══════════════════════════════════════════════════════════════════════════════

def _fix_skew(df, target, q):
    fixed = []
    for col in df.select_dtypes(include="number").columns:
        if col == target: continue
        try:
            sk = float(df[col].skew())
            if abs(sk) > SKEW_THRESHOLD and df[col].min() >= 0:
                df[col] = np.log1p(df[col])
                fixed.append(f"{col} (skew={sk:.1f})")
        except Exception:
            pass
    q["skew_fixed"] = fixed
    if fixed:
        cols_str = ", ".join(f.split(" ")[0] for f in fixed[:4])
        extra    = f" +{len(fixed)-4} more" if len(fixed) > 4 else ""
        print(f"📐  Auto-fixed skewness: {cols_str}{extra}")
    return df


def _detect_outliers(df, target, q):
    """Detect and winsorize extreme outliers (>20% of column)."""
    report    = {}
    winsorized = []
    for col in df.select_dtypes(include="number").columns:
        if col == target: continue
        try:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR    = Q3 - Q1
            lower  = Q1 - 1.5 * IQR
            upper  = Q3 + 1.5 * IQR
            n_out  = int(((df[col] < lower) | (df[col] > upper)).sum())
            pct    = n_out / max(len(df), 1)
            if pct > 0.01:
                report[col] = {"count": n_out, "pct": round(pct*100, 1)}
            # v3.1: Auto-winsorize columns with >20% outliers (skip if IQR=0)
            if pct >= 0.20 and IQR > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                winsorized.append(f"{col} ({pct*100:.0f}% outliers)")
        except Exception:
            pass
    q["outlier_report"] = report
    if winsorized:
        q.setdefault("winsorized_columns", []).extend(winsorized)
        print(f"✂️   Winsorized {len(winsorized)} column(s) with extreme outliers: "
              + ", ".join(w.split(" ")[0] for w in winsorized[:3]))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Quality report
# ══════════════════════════════════════════════════════════════════════════════

def _build_quality(df_raw, df, target, q):
    q["original_rows"]    = len(df_raw)
    q["original_columns"] = df_raw.shape[1]
    q["missing_cells"]    = int(df_raw.isnull().sum().sum())
    q["missing_pct"]      = round(df_raw.isnull().mean().mean() * 100, 2)
    q["duplicate_rows"]   = int(df_raw.duplicated().sum())

    y = df[target]
    if y.nunique() <= 15:
        counts = y.value_counts(normalize=True)
        q["class_balance"] = {str(k): round(v*100, 1) for k, v in counts.items()}
        mn = counts.min()
        if mn < 0.15:
            q["warnings"].append(
                f"Severe class imbalance: minority class = {mn*100:.1f}%. "
                "Use SMOTE or class_weight='balanced'.")
        elif mn < 0.30:
            q["warnings"].append(f"Mild class imbalance: minority = {mn*100:.1f}%.")

    skewed = []
    for col in df.select_dtypes(include="number").drop(columns=[target], errors="ignore").columns:
        try:
            sk = float(df[col].skew())
            if abs(sk) > SKEW_THRESHOLD:
                skewed.append(f"{col} (skew={sk:.1f})")
        except Exception:
            pass
    q["skewed_columns"] = skewed

    if df.shape[0] < 500:
        q["warnings"].append(
            f"Small dataset ({df.shape[0]} rows) — results may not generalise well.")
    return q


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(df_raw, file_label, target, q):
    """Common cleaning pipeline for both single and multi-file paths."""
    if len(df_raw) > MAX_SAMPLE_SIZE:
        df = df_raw.sample(MAX_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    else:
        df = df_raw.copy()

    df = df.drop_duplicates().reset_index(drop=True)

    if target is None:
        target = detect_target(df)
        print(f"🎯 Auto-detected target: '{target}'")
    elif target not in df.columns:
        raise ValueError(f"Target '{target}' not found.\nAvailable: {list(df.columns)}")

    df = _drop_useless(df, target, q)
    df = _convert_mixed(df, target, q)
    df = _drop_high_card(df, target, q)

    col_types = _infer_types(df, target)
    df        = _extract_datetimes(df, col_types, q)
    col_types = _infer_types(df, target)

    df = _impute(df, target, q)
    df = _encode_target(df, target)

    if len(df) < MIN_VIABLE_ROWS:
        raise ValueError(
            f"Only {len(df)} valid rows remain after cleaning '{target}'.\n"
            f"Try: fm.analyze(file, target='ColumnName').\n"
            f"Available: {list(df.columns)}")

    df = _fix_skew(df, target, q)
    df = _detect_outliers(df, target, q)
    q  = _build_quality(df_raw, df, target, q)

    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
    cat_cols     = [c for c in df.columns if c != target and _is_str(df[c])]

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.fillna(0)

    task_type = "classification" if df[target].nunique() <= 15 else "regression"
    total_rows = q.get("original_rows", len(df_raw))

    summary = {
        "rows"               : df.shape[0],
        "columns"            : df.shape[1],
        "target"             : target,
        "missing_values"     : int(df.isnull().sum().sum()),
        "numeric_columns"    : numeric_cols,
        "categorical_columns": cat_cols,
        "task_type"          : task_type,
        "total_file_rows"    : total_rows,
        "source_files"       : file_label,
    }

    return df, summary, q, col_types, target


def analyze_data(file, target: str = None):
    """
    Main entry point. Accepts a single file path OR a list of file paths.

    Single file:   analyze_data("data.csv")
    Multi-file:    analyze_data(["orders.csv", "customers.csv"])
    """
    print(f"\n🧠 featuremind v{VERSION} — Starting Analysis")

    q = {
        "original_rows": 0, "original_columns": 0,
        "missing_cells": 0, "missing_pct": 0.0, "duplicate_rows": 0,
        "dropped_columns": [], "converted_columns": [], "imputed_columns": [],
        "datetime_extracted": [], "skew_fixed": [],
        "class_balance": {}, "skewed_columns": [], "outlier_report": {},
        "warnings": [], "source_files": [],
    }

    if isinstance(file, (list, tuple)):
        # ── Multi-file path ───────────────────────────────────────────────────
        print(f"📁 Files : {file}")
        df_raw, total_rows, file_infos = load_and_merge(file, target)
        q["source_files"]   = [fi["file"] for fi in file_infos]
        q["original_rows"]  = total_rows
        file_label          = file_infos
        return _run_pipeline(df_raw, file_label, target, q)
    else:
        # ── Single-file path ──────────────────────────────────────────────────
        print(f"📁 File  : {file}")
        df_raw, total_rows = _load_single_csv(file)
        q["source_files"] = [file]
        file_label        = [{"file": file, "rows": total_rows, "cols": df_raw.shape[1]}]
        return _run_pipeline(df_raw, file_label, target, q)