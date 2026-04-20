"""
tracker.py — Experiment Tracker
=================================
featuremind v1.2.0

Tracks every training run automatically.
No MLflow needed — this is a lightweight built-in tracker.

Auto-tracks:
  - Model name, CV score, std, task type
  - Best hyperparameters
  - Best feature engineered
  - Dataset info (rows, columns, target)
  - All model scores compared
  - Timestamp and run ID
  - Optional notes/tags

Usage:
    # Auto-tracked during fm.analyze() and fm.train()
    # Manual usage:
    from featuremind.tracker import ExperimentTracker
    tracker = ExperimentTracker()
    tracker.log_run({...})
    tracker.leaderboard()
    tracker.export_csv("experiments.csv")
    tracker.best_run()
"""

import json
import os
import uuid
from datetime import datetime


TRACKER_FILE = "featuremind_experiments.json"


class ExperimentTracker:
    """
    Lightweight experiment tracker.
    Saves every run to a JSON file in the current directory.
    """

    def __init__(self, path: str = TRACKER_FILE):
        self.path = path
        self._runs: list = self._load()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> list:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._runs, f, indent=2, default=str)

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_run(self, eval_results: dict, summary: dict,
                data_quality: dict, features: list,
                notes: str = "", tags: list = None) -> str:
        """
        Log a complete experiment run.
        Returns the run_id string.
        """
        run_id = str(uuid.uuid4())[:8]
        ts     = datetime.now().isoformat()

        imb    = eval_results.get("imbalance_info", {})
        det    = eval_results.get("detailed_metrics", {})
        libs   = eval_results.get("libs", {})

        # Best feature info
        best_feat = None
        if features:
            bf = features[0]
            best_feat = {
                "name"    : bf.get("name"),
                "formula" : bf.get("formula"),
                "impact"  : bf.get("impact", 0),
                "layer"   : bf.get("layer"),
            }

        run = {
            "run_id"         : run_id,
            "timestamp"      : ts,
            "notes"          : notes,
            "tags"           : tags or [],

            # Dataset
            "file"           : summary.get("source_files", ["?"])[0]
                               if isinstance(summary.get("source_files"), list) else "?",
            "n_files"        : len(summary.get("source_files", [])),
            "rows"           : summary.get("rows", 0),
            "columns"        : summary.get("columns", 0),
            "target"         : summary.get("target", "?"),
            "task"           : eval_results.get("task", "?"),
            "missing_pct"    : data_quality.get("missing_pct", 0),

            # Model
            "model_name"     : eval_results.get("model_name", "?"),
            "cv_score"       : eval_results.get("cv_mean", 0),
            "cv_std"         : eval_results.get("cv_std", 0),
            "scoring_metric" : eval_results.get("scoring_metric", "accuracy"),
            "all_scores"     : eval_results.get("all_model_scores", {}),
            "best_params"    : eval_results.get("best_params", {}),

            # Metrics
            "weighted_f1"    : det.get("weighted_f1"),
            "macro_f1"       : det.get("macro_f1"),
            "r2"             : det.get("r2"),
            "mae"            : det.get("mae"),
            "rmse"           : det.get("rmse"),
            "opt_threshold"  : eval_results.get("opt_threshold", 0.5),

            # Imbalance
            "imbalance"      : imb.get("is_imbalanced", False),
            "imbalance_method": imb.get("method", "none"),
            "minority_ratio" : imb.get("minority_ratio", None),

            # Features
            "n_features"     : eval_results.get("n_features", 0),
            "best_feature"   : best_feat,

            # Libraries
            "libs"           : {k: v for k, v in libs.items()},

            # Confidence
            "confidence"     : eval_results.get("confidence_label", "?"),
        }

        self._runs.append(run)
        self._save()
        print(f"📝 Experiment logged  : run_id={run_id}  "
              f"model={run['model_name']}  score={run['cv_score']:.4f}")
        return run_id

    # ── Querying ──────────────────────────────────────────────────────────────

    def leaderboard(self, n: int = 10, task: str = None):
        """Print a ranked leaderboard of all runs."""
        runs = self._runs
        if task:
            runs = [r for r in runs if r.get("task") == task]
        if not runs:
            print("No experiments logged yet.")
            return

        runs_sorted = sorted(runs, key=lambda r: r.get("cv_score", 0), reverse=True)

        print(f"\n{'='*85}")
        print(f"  🏆  featuremind Experiment Leaderboard  ({len(runs_sorted)} runs)")
        print(f"{'='*85}")
        print(f"  {'#':<3} {'Run ID':<10} {'Model':<22} {'Score':>7} {'±Std':>6} "
              f"{'Task':<8} {'Target':<18} {'Rows':>6}")
        print(f"  {'-'*80}")

        for i, r in enumerate(runs_sorted[:n], 1):
            score  = r.get("cv_score", 0)
            std    = r.get("cv_std", 0)
            target = r.get("target", "?")[:17]
            model  = r.get("model_name", "?")[:21]
            rows   = r.get("rows", 0)
            task_  = r.get("task", "?")[:7]
            rid    = r.get("run_id", "?")
            star   = " ← best" if i == 1 else ""
            print(f"  {i:<3} {rid:<10} {model:<22} {score:>7.4f} {std:>6.4f} "
                  f"{task_:<8} {target:<18} {rows:>6}{star}")

        print(f"{'='*85}")

    def best_run(self) -> dict:
        """Return the dict for the best-scoring run."""
        if not self._runs:
            return {}
        return max(self._runs, key=lambda r: r.get("cv_score", 0))

    def run_details(self, run_id: str) -> dict:
        """Return full details for a specific run_id."""
        for r in self._runs:
            if r.get("run_id") == run_id:
                return r
        return {}

    def compare(self, run_id_a: str, run_id_b: str):
        """Side-by-side comparison of two runs."""
        a = self.run_details(run_id_a)
        b = self.run_details(run_id_b)
        if not a or not b:
            print("One or both run IDs not found.")
            return

        fields = [
            ("Model",      "model_name"),
            ("CV Score",   "cv_score"),
            ("CV Std",     "cv_std"),
            ("Task",       "task"),
            ("Target",     "target"),
            ("Rows",       "rows"),
            ("Weighted F1","weighted_f1"),
            ("Macro F1",   "macro_f1"),
            ("R²",         "r2"),
            ("MAE",        "mae"),
            ("Imbalance",  "imbalance"),
            ("Threshold",  "opt_threshold"),
        ]
        print(f"\n{'='*65}")
        print(f"  Run Comparison: {run_id_a}  vs  {run_id_b}")
        print(f"{'='*65}")
        print(f"  {'Field':<20} {'Run A':>18} {'Run B':>18}")
        print(f"  {'-'*60}")
        for label, key in fields:
            va = a.get(key, "—")
            vb = b.get(key, "—")
            if isinstance(va, float): va = f"{va:.4f}"
            if isinstance(vb, float): vb = f"{vb:.4f}"
            winner = ""
            try:
                if float(a.get(key, 0)) > float(b.get(key, 0)):
                    winner = "← A"
                elif float(b.get(key, 0)) > float(a.get(key, 0)):
                    winner = "← B"
            except Exception:
                pass
            print(f"  {label:<20} {str(va):>18} {str(vb):>18}  {winner}")
        print(f"{'='*65}")

    def export_csv(self, path: str = "featuremind_experiments.csv"):
        """Export all runs to a CSV file for Excel / analysis."""
        if not self._runs:
            print("No experiments to export.")
            return

        import csv
        if not self._runs:
            return

        # Flatten nested fields
        flat_runs = []
        for r in self._runs:
            row = {k: v for k, v in r.items()
                   if not isinstance(v, (dict, list))}
            # Add best feature info
            bf = r.get("best_feature") or {}
            row["best_feat_name"]   = bf.get("name", "")
            row["best_feat_impact"] = bf.get("impact", "")
            row["best_feat_layer"]  = bf.get("layer", "")
            # Add top model scores
            for mn, ms in (r.get("all_scores") or {}).items():
                row[f"score_{mn}"] = ms
            flat_runs.append(row)

        all_keys = list(dict.fromkeys(k for r in flat_runs for k in r))
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flat_runs)

        print(f"📊 Experiments exported : '{path}'  ({len(flat_runs)} runs)")

    def clear(self):
        """Clear all experiment history."""
        self._runs = []
        self._save()
        print("🗑️  Experiment history cleared.")

    def __len__(self):
        return len(self._runs)

    def __repr__(self):
        return f"ExperimentTracker({len(self._runs)} runs, file='{self.path}')"