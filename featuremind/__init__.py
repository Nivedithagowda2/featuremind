"""
featuremind v3.1.0
===================
v1.2 stability + v3.0 safety + all bugs fixed.

QUICK START:
    import featuremind as fm
    fm.analyze("data.csv")
    fm.analyze("data.csv", target="Churn")
    fm.check_leakage("data.csv")
    pipeline = fm.train("data.csv", target="Churn")
    pipeline.save("my_pipeline")
    pipeline = fm.load_pipeline("my_pipeline")
    predictions = pipeline.predict_df(new_df)
    tracker = fm.get_tracker()
    tracker.leaderboard()
"""

from .analyzer         import analyze_data
from .importance       import get_feature_importance
from .feature_engineer import generate_feature_suggestions
from .evaluator        import evaluate_features
from .reporter         import generate_insights, print_report
from .html_reporter    import generate_html_report
from .pipeline         import FeaturemindPipeline, load_pipeline
from .tracker          import ExperimentTracker
from .api              import serve
from .leakage_guard    import (
    run_full_leakage_guard,
    check_score_reliability,
    check_dataset_leakage,
)

__version__ = "3.1.0"
__author__  = "featuremind"

_tracker = ExperimentTracker()


def get_tracker() -> ExperimentTracker:
    return _tracker


def check_leakage(file, target=None):
    """Standalone leakage scan — no model training needed."""
    import pandas as pd
    try:
        df = pd.read_csv(file, nrows=5000)
        if target is None:
            from .analyzer import detect_target
            target = detect_target(df)
            print(f"🎯 Auto-detected target: '{target}'")
        print(f"\n🛡️  Leakage scan: '{file}' (target='{target}')")
        warnings_list = check_dataset_leakage(df, target)
        if warnings_list:
            print(f"\n  ⚠️  Found {len(warnings_list)} potential leakage risk(s):")
            for w in warnings_list:
                print(f"     {w}")
        else:
            print(f"  ✅ No column-level leakage detected.")
        return {"dataset_warnings": warnings_list, "target": target}
    except Exception as e:
        print(f"⚠️  Leakage check error: {e}")
        return {"dataset_warnings": [], "target": target}


def analyze(file, target: str = None, html: bool = True,
            notes: str = "", tags: list = None) -> dict:
    """
    Run the full featuremind v3.1 analysis pipeline.

    Args:
        file   : str or list — single CSV path or list of CSV paths
        target : target column name. Auto-detected if not provided.
        html   : save HTML report (default True)
        notes  : optional experiment note
        tags   : optional list of tags

    Returns:
        dict with eval_results, summary, data_quality, features, importance, df
    """
    # Step 1: Load + clean
    try:
        df, summary, data_quality, col_types, target = analyze_data(file, target)
    except Exception as e:
        print(f"\n❌ Data loading failed: {e}")
        return {}

    # Step 2: Feature importance
    try:
        importance = get_feature_importance(df, target)
    except Exception as e:
        print(f"⚠️  Feature importance failed: {e}")
        importance = {}

    # Step 3: Feature suggestions (12 layers, pre-scored)
    try:
        features = generate_feature_suggestions(df, target, col_types)
    except Exception as e:
        print(f"⚠️  Feature suggestion failed: {e}")
        features = []

    # Step 4: Evaluate (SMOTE + threshold + SHAP + bias + leakage guard)
    try:
        base_score, model_name, final_features, eval_results = evaluate_features(
            df, target, features)
    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")
        base_score, model_name, final_features, eval_results = 0.0, None, [], {}

    # Step 5: Insights
    try:
        insights_report = generate_insights(
            summary, data_quality, importance, final_features, eval_results)
    except Exception as e:
        print(f"⚠️  Insight generation failed: {e}")
        insights_report = {}

    # Step 6: Terminal report + PNG chart
    try:
        print_report(
            summary, data_quality, col_types, importance, final_features,
            model_name, base_score, eval_results, insights_report, df)
    except Exception as e:
        print(f"⚠️  Report generation failed: {e}")

    # Step 7: HTML report
    if html:
        try:
            generate_html_report(
                summary, data_quality, col_types, importance, final_features,
                model_name, base_score, eval_results, insights_report)
        except Exception as e:
            print(f"⚠️  HTML report failed: {e}")

    # Step 8: Auto-log to tracker
    try:
        _tracker.log_run(eval_results, summary, data_quality,
                         final_features, notes=notes, tags=tags or [])
    except Exception:
        pass

    return {
        "eval_results" : eval_results,
        "summary"      : summary,
        "data_quality" : data_quality,
        "features"     : final_features,
        "importance"   : importance,
        "df"           : df,
    }


def train(file, target: str = None, html: bool = True,
          notes: str = "", tags: list = None) -> FeaturemindPipeline:
    """
    Train featuremind and return a production-ready pipeline.

    Example:
        pipeline = fm.train("data.csv", target="Churn")
        pipeline.save("churn_v3")
        pipeline.summary()
    """
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    results = analyze(file, target=target, html=html, notes=notes, tags=tags)
    if not results:
        raise RuntimeError("Analysis failed — cannot build pipeline.")

    df           = results["df"]
    summary      = results["summary"]
    eval_res     = results["eval_results"]
    features     = results["features"]
    target       = summary["target"]

    print(f"\n🔧 Building production pipeline on full dataset...")

    X = df.drop(columns=[target]).fillna(0)
    y = df[target]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    task       = eval_res.get("task", "classification")
    model_name = eval_res.get("model_name", "RandomForest")
    best_params= eval_res.get("best_params", {})
    imb_info   = eval_res.get("imbalance_info", {})
    use_balanced = imb_info.get("is_imbalanced", False)

    from .evaluator import _get_models
    model_registry = _get_models(task, use_balanced)
    best_model     = model_registry.get(model_name)

    if best_model is not None and best_params:
        try:
            best_model.set_params(**{k: v for k, v in best_params.items()
                                     if hasattr(best_model, k)})
        except Exception:
            pass

    if best_model is None:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        best_model = (RandomForestClassifier(random_state=42)
                      if task == "classification" else
                      RandomForestRegressor(random_state=42))

    X_train, y_train = X_scaled, y
    if task == "classification" and imb_info.get("method") == "SMOTE":
        try:
            from imblearn.over_sampling import SMOTE
            minority = y.value_counts().min()
            k = min(5, minority - 1)
            if k >= 1:
                sm = SMOTE(random_state=42, k_neighbors=k)
                X_train, y_train = sm.fit_resample(X_scaled, y)
        except Exception:
            pass

    try:
        best_model.fit(X_train, y_train)
    except Exception as e:
        print(f"⚠️  Fit failed ({e}), using RandomForest fallback")
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        best_model = (RandomForestClassifier(random_state=42)
                      if task == "classification" else
                      RandomForestRegressor(random_state=42))
        best_model.fit(X_train, y_train)

    best_feat_info = None
    if features:
        bf = features[0]
        best_feat_info = {
            "name"   : bf.get("name"),
            "formula": bf.get("formula"),
            "impact" : bf.get("impact", 0),
            "layer"  : bf.get("layer"),
        }

    pipeline = FeaturemindPipeline()
    pipeline.fit(
        df=df, target=target, model=best_model, scaler=scaler,
        feature_names=list(X.columns), task=task,
        opt_threshold=eval_res.get("opt_threshold", 0.5),
        model_name=model_name, base_score=eval_res.get("cv_mean", 0.0),
        best_feature=best_feat_info, col_types={},
    )

    print(f"✅ Production pipeline ready!")
    print(f"   Model     : {model_name}")
    print(f"   Features  : {len(list(X.columns))}")
    print(f"   Call: pipeline.save('my_pipeline_name') to save")
    return pipeline