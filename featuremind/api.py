"""
api.py — FastAPI Prediction Server
=====================================
featuremind v1.2.0

Launches a REST API for real-time predictions using a saved pipeline.

Usage:
    # Start the API server
    import featuremind as fm
    fm.serve("my_pipeline/")          # default port 8000

    # Or from command line:
    python -m featuremind.api my_pipeline/ --port 8000

API Endpoints (once running):
    GET  /                     Health check + pipeline info
    GET  /info                 Pipeline metadata (model, score, features, threshold)
    POST /predict              Predict from JSON payload
    POST /predict/batch        Predict from a CSV file upload
    GET  /docs                 Auto-generated Swagger UI (FastAPI built-in)

Predict endpoint — example request:
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{"data": [{"Age": 35, "Tenure": 24, "MonthlyCharges": 65.5}]}'

Response:
    {
      "predictions": [1],
      "probabilities": [[0.28, 0.72]],
      "model": "CatBoost",
      "threshold": 0.45,
      "task": "classification"
    }

Install:
    pip install fastapi uvicorn python-multipart
"""

import json
import os
import sys
import warnings
from typing import Any

warnings.filterwarnings("ignore")

# FastAPI is optional — graceful error if not installed
try:
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

import numpy as np
import pandas as pd

from .pipeline import FeaturemindPipeline


# ── Request schemas ───────────────────────────────────────────────────────────

if _HAS_FASTAPI:
    class PredictRequest(BaseModel):
        data: list[dict[str, Any]]

    class PredictResponse(BaseModel):
        predictions: list
        probabilities: list | None = None
        model: str
        threshold: float
        task: str
        n_samples: int


# ── Server factory ────────────────────────────────────────────────────────────

def create_app(pipeline_path: str) -> "FastAPI":
    """
    Create a FastAPI app from a saved pipeline path.
    Returns the app object (useful for testing or deployment).
    """
    if not _HAS_FASTAPI:
        raise ImportError(
            "FastAPI not installed. Run: pip install fastapi uvicorn python-multipart")

    # Load pipeline at startup
    pipeline = FeaturemindPipeline.load(pipeline_path)

    app = FastAPI(
        title       = "featuremind Prediction API",
        description = f"Real-time predictions using featuremind v1.2.0 pipeline.\n"
                      f"Model: {pipeline.model_name} | Target: {pipeline.target} | "
                      f"Score: {pipeline.base_score:.4f}",
        version     = "1.2.0",
    )

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/")
    async def root():
        return {
            "status"   : "🟢 featuremind API running",
            "model"    : pipeline.model_name,
            "target"   : pipeline.target,
            "task"     : pipeline.task,
            "score"    : pipeline.base_score,
            "features" : len(pipeline.feature_names),
            "version"  : "1.2.0",
            "docs"     : "/docs",
        }

    # ── Pipeline info ─────────────────────────────────────────────────────────
    @app.get("/info")
    async def info():
        return {
            "model_name"    : pipeline.model_name,
            "task"          : pipeline.task,
            "target"        : pipeline.target,
            "cv_score"      : pipeline.base_score,
            "threshold"     : pipeline.opt_threshold,
            "n_features"    : len(pipeline.feature_names),
            "feature_names" : pipeline.feature_names[:20],   # first 20
            "trained_at"    : pipeline.trained_at,
            "training_rows" : pipeline.training_rows,
            "best_feature"  : pipeline.best_feature,
        }

    # ── Single / batch predict (JSON) ─────────────────────────────────────────
    @app.post("/predict")
    async def predict(request: PredictRequest):
        try:
            df   = pd.DataFrame(request.data)
            preds = pipeline.predict(df).tolist()

            probabilities = None
            if pipeline.task == "classification" and hasattr(pipeline.model, "predict_proba"):
                try:
                    X     = pipeline._preprocess(df)
                    proba = pipeline.model.predict_proba(X).tolist()
                    probabilities = proba
                except Exception:
                    pass

            return {
                "predictions"  : preds,
                "probabilities": probabilities,
                "model"        : pipeline.model_name,
                "threshold"    : pipeline.opt_threshold,
                "task"         : pipeline.task,
                "n_samples"    : len(preds),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    # ── CSV file upload predict ───────────────────────────────────────────────
    @app.post("/predict/batch")
    async def predict_batch(file: UploadFile = File(...)):
        try:
            import io
            contents = await file.read()
            df       = pd.read_csv(io.StringIO(contents.decode("utf-8")))
            preds    = pipeline.predict(df).tolist()

            result   = df.copy()
            result["prediction"] = preds

            if pipeline.task == "classification" and hasattr(pipeline.model, "predict_proba"):
                try:
                    X     = pipeline._preprocess(df)
                    proba = pipeline.model.predict_proba(X).max(axis=1)
                    result["confidence_pct"] = (proba * 100).round(1).tolist()
                except Exception:
                    pass

            return JSONResponse(content={
                "n_samples"  : len(preds),
                "predictions": preds,
                "model"      : pipeline.model_name,
                "task"       : pipeline.task,
                "columns"    : list(df.columns),
            })
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Batch prediction error: {e}")

    return app


def serve(pipeline_path: str, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Launch the FastAPI prediction server.

    Args:
        pipeline_path : Path to saved pipeline directory.
        host          : Host to bind (default: 0.0.0.0 = all interfaces).
        port          : Port number (default: 8000).
        reload        : Auto-reload on code changes (development only).

    Example:
        import featuremind as fm
        fm.serve("churn_pipeline/", port=8000)
    """
    if not _HAS_FASTAPI:
        print("❌ FastAPI not installed.")
        print("   Install with: pip install fastapi uvicorn python-multipart")
        return

    if not os.path.exists(pipeline_path):
        print(f"❌ Pipeline not found at '{pipeline_path}'")
        print("   Train first: pipeline = fm.train('data.csv'); pipeline.save('my_pipeline')")
        return

    print(f"\n🚀 Starting featuremind API server")
    print(f"   Pipeline  : {pipeline_path}")
    print(f"   Address   : http://{host}:{port}")
    print(f"   Docs      : http://{host}:{port}/docs")
    print(f"   Health    : http://{host}:{port}/")
    print(f"\n   Press Ctrl+C to stop\n")

    # Create app and serve
    app = create_app(pipeline_path)
    uvicorn.run(app, host=host, port=port, reload=reload)