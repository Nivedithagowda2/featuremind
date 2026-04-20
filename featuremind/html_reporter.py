"""
html_reporter.py — Self-Contained HTML Report
===============================================
featuremind v1.0.0

BUG FIXES vs v0.6:
  ✅ "too many values to unpack (expected 3)" — library loop now uses
     4 variables: for lib, key, install_cmd, benefit in [...]
  ✅ SHAP values shown as raw scores (not percentages) — fixes "21910.6%"
  ✅ Multi-file source information shown in dataset summary
"""

import base64
import datetime
import os


_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;color:#222}
.hdr{background:linear-gradient(135deg,#1a237e 0%,#283593 55%,#3949ab 100%);
     color:#fff;padding:32px 44px}
.hdr h1{font-size:24px;font-weight:700}
.hdr p{font-size:13px;opacity:.8;margin-top:5px}
.wrap{max-width:1080px;margin:0 auto;padding:24px 20px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-bottom:18px}
.card{background:#fff;border-radius:10px;padding:20px 24px;box-shadow:0 2px 8px rgba(0,0,0,.07)}
.card h2{font-size:13px;font-weight:700;color:#1a237e;margin-bottom:12px;
         border-bottom:2px solid #e8eaf6;padding-bottom:7px}
.sr{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f0f0f0;font-size:13px}
.sr:last-child{border-bottom:none}
.sl{color:#666}.sv{font-weight:600;color:#1a237e}
.big{font-size:44px;font-weight:800;letter-spacing:-2px}
.sub{font-size:12px;color:#777;margin-top:3px}
.sec{background:#fff;border-radius:10px;padding:20px 24px;
     box-shadow:0 2px 8px rgba(0,0,0,.07);margin-bottom:18px}
.sec h2{font-size:13px;font-weight:700;color:#1a237e;margin-bottom:12px;
        border-bottom:2px solid #e8eaf6;padding-bottom:7px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#e8eaf6;color:#1a237e;padding:8px 10px;text-align:left;font-weight:700}
td{padding:7px 10px;border-bottom:1px solid #f0f0f0}
tr:hover td{background:#fafafa}
.cb{display:inline-block;padding:5px 14px;border-radius:16px;font-size:13px;font-weight:700;margin-bottom:6px}
.hi{background:#e8f5e9;color:#2e7d32}.me{background:#fff8e1;color:#f57f17}.lo{background:#ffebee;color:#c62828}
.fc{background:#fff;border:1px solid #e8eaf6;border-radius:8px;padding:14px 18px;margin-bottom:10px}
.fc h3{font-size:13px;font-weight:700;color:#283593;margin-bottom:6px}
.fml{font-family:monospace;font-size:11px;background:#f3f4f8;padding:5px 8px;border-radius:4px;color:#333;margin:5px 0}
.fr{display:flex;justify-content:space-between;font-size:12px;color:#555;margin-top:3px}
.lb{display:inline-block;padding:2px 7px;border-radius:3px;font-size:11px;font-weight:600;color:#fff}
.bg{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:700}
.bg-g{background:#e8f5e9;color:#2e7d32}.bg-a{background:#fff8e1;color:#f57f17}
.bg-r{background:#ffebee;color:#c62828}.bg-z{background:#eceff1;color:#546e7a}
.ri{display:flex;gap:10px;padding:9px 0;border-bottom:1px solid #f0f0f0;font-size:13px}
.ri:last-child{border-bottom:none}
.rn{background:#1a237e;color:#fff;border-radius:50%;width:22px;height:22px;
    display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0}
.warn{background:#fff8e1;border-left:4px solid #ffb300;padding:9px 12px;
      border-radius:0 5px 5px 0;margin-bottom:7px;font-size:13px}
.tag{background:#e8eaf6;color:#283593;padding:3px 8px;border-radius:9px;font-size:11px;margin:2px}
.tgw{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px}
img.ch{width:100%;border-radius:7px;margin-top:6px}
"""


def _b64(path):
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        pass
    return ""


def _sc(score, task):
    if task == "classification":
        return "#2e7d32" if score >= 0.85 else "#f57f17" if score >= 0.70 else "#c62828"
    return "#2e7d32" if score >= 0.60 else "#f57f17" if score >= 0.30 else "#c62828"


def _ibadge(impact):
    if impact >= 0.02:   return f'<span class="bg bg-g">▲ +{impact:.4f}</span>'
    elif impact > 0.005: return f'<span class="bg bg-a">▲ +{impact:.4f}</span>'
    elif impact == 0:    return f'<span class="bg bg-z">= 0</span>'
    else:                return f'<span class="bg bg-r">▼ {impact:.4f}</span>'


def _lbadge(layer):
    colors = {"domain":"#1565c0","interaction":"#6a1b9a","ratio":"#00695c",
              "transform":"#e65100","polynomial":"#4e342e","binning":"#1b5e20",
              "delta":"#880e4f","outlier":"#b71c1c","nlp":"#0d47a1","rank":"#37474f"}
    c = colors.get(layer.lower(), "#555")
    return f'<span class="lb" style="background:{c};">{layer.upper()}</span>'


def _pbar(pct, color="#1976d2", w=140):
    fw = max(3, int(pct / 100 * w))
    return (f'<div style="display:inline-block;background:#e3f2fd;border-radius:3px;'
            f'width:{w}px;height:12px;vertical-align:middle;">'
            f'<div style="background:{color};width:{fw}px;height:12px;border-radius:3px;"></div>'
            f'</div> <span style="font-size:11px;color:#666;">{pct:.1f}%</span>')


def generate_html_report(
    summary, data_quality, col_types, importance, features,
    model_name, base_score, eval_results, insights_report,
    save_path="featuremind_report.html",
):
    """Generate self-contained HTML report. BUG FIX: 4-tuple lib loop."""

    task        = eval_results.get("task", "classification")
    cv_std      = eval_results.get("cv_std", 0)
    all_scores  = eval_results.get("all_model_scores", {})
    det         = eval_results.get("detailed_metrics", {})
    confidence  = eval_results.get("confidence_label", "N/A")
    shap_imp    = eval_results.get("shap_importance", {})
    bias_warns  = eval_results.get("bias_warnings", [])
    best_params = eval_results.get("best_params", {})
    libs        = eval_results.get("libs", {})
    use_cv      = eval_results.get("use_cv", True)
    n_rows      = eval_results.get("n_rows", 0)
    target      = summary.get("target", "")
    ts          = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    sc        = _sc(base_score, task)
    chart_b64 = _b64("featuremind_report.png")
    imp       = shap_imp if shap_imp else importance
    imp_lbl   = "SHAP Feature Impact (raw scores)" if shap_imp else "Feature Importance (RandomForest)"
    conf_cls  = ("hi" if "High" in confidence else "me" if "Medium" in confidence else "lo")

    sources = data_quality.get("source_files", [])
    src_str = f"{len(sources)} files merged" if len(sources) > 1 else (sources[0] if sources else "")

    H = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>featuremind v1.0 — {target}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="hdr">
  <h1>🧠 featuremind v1.0.0 — Enterprise Analysis Report</h1>
  <p>Target: <b>{target}</b> &nbsp;|&nbsp; Task: <b>{task.capitalize()}</b>
     &nbsp;|&nbsp; Source: <b>{src_str}</b> &nbsp;|&nbsp; {ts}</p>
</div>
<div class="wrap">
"""
    # ── Row 1: Score · Models · Confidence ────────────────────────────────────
    H += '<div class="g3">'
    H += f"""<div class="card" style="text-align:center">
  <h2>🎯 Model Score</h2>
  <div class="big" style="color:{sc}">{base_score:.3f}</div>
  <div class="sub">{'Accuracy · 5-fold CV' if use_cv else 'Score · single split'} ±{cv_std:.4f}</div>
  <div style="font-size:12px;margin-top:8px">{insights_report.get('performance_summary','')}</div>
</div>"""

    H += '<div class="card"><h2>🏁 Model Comparison</h2><table>'
    H += '<tr><th>Model</th><th>CV Score</th></tr>'
    for mn, ms in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
        star = " ⭐" if mn == model_name else ""
        rs   = ' style="background:#e8f5e9"' if mn == model_name else ""
        H += f'<tr{rs}><td>{mn}{star}</td><td><b>{ms:.4f}</b></td></tr>'
    H += '</table></div>'

    H += f"""<div class="card" style="text-align:center">
  <h2>🔒 Confidence</h2>
  <div class="cb {conf_cls}">{confidence}</div>
  <div style="font-size:12px;color:#666;margin-top:7px">{eval_results.get('confidence_reason','')}</div>
  <div style="font-size:11px;color:#888;margin-top:9px">{'5-fold CV' if use_cv else 'Single split'} · {n_rows:,} rows</div>
</div>"""
    H += '</div>\n'

    # ── Detailed Metrics ──────────────────────────────────────────────────────
    H += '<div class="sec"><h2>📈 Detailed Metrics</h2>'
    if task == "classification":
        f1w = det.get("weighted_f1","—"); f1m = det.get("macro_f1","—")
        ca  = det.get("avg_confidence"); lc  = det.get("low_confidence_pct")
        H += '<div class="g2"><div><table><tr><th colspan="2">Overall</th></tr>'
        H += f'<tr><td>Weighted F1</td><td><b>{f1w}</b></td></tr>'
        H += f'<tr><td>Macro F1</td><td><b>{f1m}</b></td></tr>'
        if ca is not None:
            H += f'<tr><td>Avg Confidence</td><td><b>{ca*100:.1f}%</b></td></tr>'
            H += f'<tr><td>Low-conf preds</td><td><b>{lc:.1f}%</b></td></tr>'
        H += '</table></div><div><table>'
        H += '<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr>'
        for cls, m in det.get("per_class", {}).items():
            H += (f'<tr><td>{cls}</td><td>{m.get("precision",0):.3f}</td>'
                  f'<td>{m.get("recall",0):.3f}</td><td>{m.get("f1-score",0):.3f}</td></tr>')
        H += '</table></div></div>'
    else:
        r2, mae, rmse = det.get("r2","—"), det.get("mae","—"), det.get("rmse","—")
        H += f'<table style="max-width:340px"><tr><th>Metric</th><th>Value</th></tr>'
        H += f'<tr><td>R²</td><td><b>{r2}</b></td></tr>'
        H += f'<tr><td>MAE</td><td><b>{mae}</b></td></tr>'
        H += f'<tr><td>RMSE</td><td><b>{rmse}</b></td></tr>'
        pi = det.get("pred_interval_95")
        if pi: H += f'<tr><td>95th Uncertainty</td><td><b>±{pi:.4f}</b></td></tr>'
        H += '</table>'
    for ins in insights_report.get("metrics_detail", []):
        H += f'<p style="font-size:13px;margin-top:8px;color:#555">ℹ️ {ins}</p>'
    H += '</div>\n'

    # ── Dataset + Data Quality ────────────────────────────────────────────────
    dq = data_quality
    H += '<div class="g2">'
    H += '<div class="card"><h2>📊 Dataset Summary</h2>'
    for lbl, val in [
        ("Rows (sampled)", f"{summary['rows']:,}"),
        ("Total file rows", f"{summary.get('total_file_rows','?'):,}"
                            if isinstance(summary.get('total_file_rows'), int) else "—"),
        ("Columns (encoded)", f"{summary['columns']:,}"),
        ("Target", summary['target']),
        ("Task", summary.get('task_type','?').capitalize()),
        ("Source", src_str),
    ]:
        H += f'<div class="sr"><span class="sl">{lbl}</span><span class="sv">{val}</span></div>'
    H += '</div>'

    H += '<div class="card"><h2>🧹 Data Quality</h2>'
    for lbl, val in [
        ("Original rows",   f"{dq.get('original_rows','?'):,}" if isinstance(dq.get('original_rows'), int) else "—"),
        ("Original columns", str(dq.get('original_columns','?'))),
        ("Missing cells",   f"{dq.get('missing_cells',0):,} ({dq.get('missing_pct',0):.1f}%)"),
        ("Duplicate rows",  str(dq.get('duplicate_rows',0))),
        ("Dropped",         str(len(dq.get('dropped_columns',[])))),
        ("Converted",       str(len(dq.get('converted_columns',[])))),
        ("Imputed",         str(len(dq.get('imputed_columns',[])))),
        ("Skew fixed",      str(len(dq.get('skew_fixed',[])))),
    ]:
        H += f'<div class="sr"><span class="sl">{lbl}</span><span class="sv">{val}</span></div>'
    cb = dq.get("class_balance", {})
    if cb:
        H += '<div style="margin-top:9px;font-size:12px;font-weight:700;color:#1a237e">Class Balance</div>'
        for cls, pct in cb.items():
            H += f'<div class="sr"><span class="sl">Class {cls}</span><span>{_pbar(pct)}</span></div>'
    H += '</div></div>\n'

    # ── Warnings ──────────────────────────────────────────────────────────────
    all_w = dq.get("warnings", []) + bias_warns
    if all_w:
        H += '<div class="sec"><h2>⚠️ Warnings & Alerts</h2>'
        for w in all_w:
            icon = "⚖️" if "bias" in w.lower() else "⚠️"
            H += f'<div class="warn">{icon} {w}</div>'
        H += '</div>\n'

    # ── Column types ──────────────────────────────────────────────────────────
    if col_types:
        H += '<div class="sec"><h2>🔬 Column Type Inference</h2>'
        groups: dict = {}
        for col, t in col_types.items():
            groups.setdefault(t, []).append(col)
        icons = {"numeric":"🔢","binary":"⚡","ordinal":"📊","categorical":"🏷️",
                 "datetime":"📅","text":"📝","id":"🔑","target":"🎯"}
        for t, cols in sorted(groups.items()):
            icon = icons.get(t, "•")
            H += f'<div style="margin-bottom:8px"><b>{icon} {t.capitalize()}</b>'
            H += '<div class="tgw">'
            for col in cols: H += f'<span class="tag">{col}</span>'
            H += '</div></div>'
        H += '</div>\n'

    # ── Outliers ──────────────────────────────────────────────────────────────
    outliers = dq.get("outlier_report", {})
    if outliers:
        H += '<div class="sec"><h2>🔍 Outlier Report (IQR)</h2>'
        H += '<table><tr><th>Column</th><th>Count</th><th>% of Rows</th></tr>'
        for col, info in outliers.items():
            H += f'<tr><td>{col}</td><td>{info["count"]:,}</td><td>{info["pct"]}%</td></tr>'
        H += '</table></div>\n'

    # ── Tuning ────────────────────────────────────────────────────────────────
    if best_params:
        H += '<div class="sec"><h2>🔧 Hyperparameter Tuning</h2>'
        H += f'<p style="font-size:13px;margin-bottom:9px">{insights_report.get("tuning_summary","")}</p>'
        H += '<table style="max-width:360px"><tr><th>Parameter</th><th>Best Value</th></tr>'
        for k, v in best_params.items():
            H += f'<tr><td>{k}</td><td><b>{v}</b></td></tr>'
        H += '</table></div>\n'

    # ── Chart ─────────────────────────────────────────────────────────────────
    if chart_b64:
        H += f'<div class="sec"><h2>📊 Analysis Chart</h2><img class="ch" src="{chart_b64}" alt="chart"></div>\n'

    # ── Feature Importance  ← FIX: SHAP shown as raw scores ──────────────────
    if imp:
        H += f'<div class="sec"><h2>🏆 {imp_lbl}</h2>'
        H += '<table><tr><th>#</th><th>Feature</th><th>Score</th><th>Bar</th></tr>'
        mv = max(imp.values()) if imp else 1
        for i, (col, val) in enumerate(list(imp.items())[:10], 1):
            display = f"{val:.4f}" if shap_imp else f"{val*100:.1f}%"
            H += (f'<tr><td>{i}</td><td>{col}</td>'
                  f'<td>{display}</td>'
                  f'<td>{_pbar(val/mv*100,"#1565c0",120)}</td></tr>')
        H += '</table>'
        for d in insights_report.get("top_drivers", []):
            H += f'<p style="font-size:13px;margin-top:8px;color:#555">💡 {d}</p>'
        H += '</div>\n'

    # ── Feature Suggestions ───────────────────────────────────────────────────
    H += '<div class="sec"><h2>🔬 Feature Suggestions</h2>'
    if not features:
        H += '<p style="color:#888;font-size:13px">No valid feature suggestions evaluated.</p>'
    else:
        for f in features:
            impact = f.get("impact", 0)
            layer  = f.get("layer", "generic")
            H += (f'<div class="fc"><h3>{f.get("rank","?")} — {f["name"]} &nbsp;'
                  f'{_lbadge(layer)} &nbsp;{_ibadge(impact)}</h3>'
                  f'<div class="fml">{f["formula"]}</div>'
                  f'<div class="fr"><span>📌 {f["business"]}</span>'
                  f'<span>New Score: <b>{f["new_score"]:.4f}</b></span></div>'
                  f'<div style="font-size:12px;color:#666;margin-top:3px">💡 {f["reason"]}</div></div>')
    H += '</div>\n'

    # ── Verdict ───────────────────────────────────────────────────────────────
    H += (f'<div class="sec"><h2>🏅 Feature Engineering Verdict</h2>'
          f'<p style="font-size:14px">{insights_report.get("feature_verdict","")}</p></div>\n')

    # ── Data Insights ─────────────────────────────────────────────────────────
    data_ins = insights_report.get("data_insights", [])
    if data_ins:
        H += '<div class="sec"><h2>💬 Data Insights</h2>'
        for ins in data_ins:
            H += f'<div style="padding:6px 0;border-bottom:1px solid #f0f0f0;font-size:13px">• {ins}</div>'
        H += '</div>\n'

    # ── Recommendations ───────────────────────────────────────────────────────
    H += '<div class="sec"><h2>🚀 Recommendations (Priority Order)</h2>'
    for i, rec in enumerate(insights_report.get("recommendations", []), 1):
        H += (f'<div class="ri"><div class="rn">{i}</div>'
              f'<div style="padding-top:1px">{rec}</div></div>')
    H += '</div>\n'

    # ── Library Status  ← BUG FIX: 4-variable tuple unpacking ────────────────
    H += '<div class="sec"><h2>📦 Library Status</h2>'
    H += '<table style="max-width:600px"><tr><th>Library</th><th>Status</th><th>Install</th><th>Benefit</th></tr>'
    for lib, key, install_cmd, benefit in [
        ("XGBoost",  "xgboost",  "pip install xgboost",  "Best accuracy · industry standard"),
        ("LightGBM", "lightgbm", "pip install lightgbm", "Fastest on large datasets"),
        ("CatBoost", "catboost", "pip install catboost", "Best on categorical data"),
        ("SHAP",     "shap",     "pip install shap",     "Explains WHY each prediction was made"),
    ]:
        ok = libs.get(key, False)
        st = (f'<span style="color:#2e7d32;font-weight:600">✅ Installed</span>' if ok else
              f'<span style="color:#c62828;font-weight:600">❌ Missing</span>')
        cmd = ("—" if ok else
               f'<code style="font-size:11px;background:#f3f4f8;padding:2px 5px;border-radius:3px">'
               f'{install_cmd}</code>')
        H += (f'<tr><td><b>{lib}</b></td><td>{st}</td>'
              f'<td style="font-size:12px">{cmd}</td>'
              f'<td style="font-size:12px;color:#666">{benefit}</td></tr>')
    H += '</table></div>\n'

    H += (f'<div style="text-align:center;padding:20px;color:#aaa;font-size:11px">'
          f'Generated by <b>featuremind v1.0.0</b> &nbsp;|&nbsp; {ts}</div>\n'
          f'</div></body></html>')

    with open(save_path, "w", encoding="utf-8") as fout:
        fout.write(H)
    print(f"📄 HTML report saved  : '{save_path}'")
    return save_path