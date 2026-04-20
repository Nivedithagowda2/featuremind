"""
test.py — featuremind v3.1.1 Demo
================================
Works on ANY dataset. Change filenames below.
"""
import featuremind as fm

print("\n" + "="*65)
print(f"  featuremind v{fm.__version__} — Full Demo")
print("="*65)

# Change these to your actual CSV files
DATA_FILE = "data2.csv"      # your main dataset

# ── DEMO 1: Standard Analysis ─────────────────────────────────────────────────
print("\n" + "─"*65)
print("  DEMO 1: Standard Analysis")
print("─"*65)
fm.analyze(DATA_FILE)

# ── DEMO 2: Standalone Leakage Check ─────────────────────────────────────────
print("\n" + "─"*65)
print("  DEMO 2: Standalone Leakage Check")
print("─"*65)
fm.check_leakage(DATA_FILE)

# ── DEMO 3: Train & Save Production Pipeline ──────────────────────────────────
print("\n" + "─"*65)
print("  DEMO 3: Train & Save Production Pipeline")
print("─"*65)
pipeline = fm.train(DATA_FILE)
pipeline.save("my_pipeline")
pipeline.summary()

# ── DEMO 4: Load Pipeline + Predict ───────────────────────────────────────────
print("\n" + "─"*65)
print("  DEMO 4: Load Pipeline + Predict on New Data")
print("─"*65)
import pandas as pd
pipeline = fm.load_pipeline("my_pipeline")
pipeline.summary()
df_new   = pd.read_csv(DATA_FILE).head(5)
results  = pipeline.predict_df(df_new)
print("\n  Predictions:")
cols = ["prediction", "confidence"] if "confidence" in results.columns else ["prediction"]
print(results[cols].to_string(index=False))

# ── DEMO 5: Experiment Tracker ────────────────────────────────────────────────
print("\n" + "─"*65)
print("  DEMO 5: Experiment Tracker")
print("─"*65)
tracker = fm.get_tracker()
tracker.leaderboard()
tracker.export_csv("featuremind_experiments.csv")

print("\n" + "="*65)
print("  ✅ featuremind v3.1 demo complete!")
print("  Files generated:")
print("    featuremind_report.png")
print("    featuremind_report.html")
print("    enhanced_data.csv")
print("    featuremind_experiments.csv")
print("    my_pipeline/  (model.pkl + scaler.pkl + config.json)")
print("="*65)