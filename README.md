# 🧠 featuremind v3.1.1

**One-line AutoML with Built-in Reliability, Leakage Detection & Explainability**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyPI version](https://img.shields.io/pypi/v/featuremind)](https://pypi.org/project/featuremind/)
[![Leakage Guard](https://img.shields.io/badge/Leakage%20Guard-Enabled-red)](https://github.com/Nivedithagowda2/featuremind)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Nivedithagowda2/featuremind/blob/main/LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)](https://github.com/Nivedithagowda2/featuremind)

---

> ⭐ If this project helps you, give it a star — it really helps!

---

## 📌 What is featuremind?

**featuremind** is a one-line AutoML library that handles the complete machine learning pipeline — from raw CSV to production-ready model — with built-in reliability checking, leakage detection, and feature engineering.

```python
import featuremind as fm
fm.analyze("data.csv")
```

That's it. One line. Full analysis, model selection, feature suggestions, SHAP importance, leakage check, and HTML report — all automated.

---

## 🧪 Tested Datasets

featuremind v3.1 has been verified on:

| Dataset                          | Type           | Score         | Notes                                               |
| -------------------------------- | -------------- | ------------- | --------------------------------------------------- |
| Telecom Churn (7,043 rows)       | Classification | 85.7% F1      | ✅ Stable, well-balanced                             |
| Credit Card Fraud (284,807 rows) | Classification | ~99% F1       | ⚠️ High score due to PCA-transformed separable data |
| Heart Failure Medical            | Classification | ~80% Accuracy | ✅ Works                                             |
| House Prices                     | Regression     | R² reported   | ✅ Works                                             |
| Generic CSVs                     | Auto-detected  | Auto-detected | ✅ Works                                             |

---

## 🚀 Key Features

### 🤖 Auto ML Pipeline

* Loads and cleans any CSV automatically
* Detects target column, task type (classification / regression), and data issues
* Trains 6 models: LogisticRegression, RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost
* Picks best model using cross-validation
* Auto hyperparameter tuning (RandomizedSearchCV)

### 🛡️ Leakage Guard (Core Feature)

* Detects if any feature formula references the target column
* Flags columns with suspiciously high correlation with target (>0.95)
* Smart ID detection (non-generalizable columns)
* Warns user before model training (no silent failures)

### 🔍 Reliability Engine

* Detects unrealistic scores (>0.98)
* Adjusts confidence level automatically:

  * > 0.99 → Low confidence ❌
  * > 0.98 → Medium ⚠️
* Highlights possible issues:

  * Data leakage
  * Overfitting
  * Sampling bias

### ⚖️ Class Imbalance Handling

* Detects imbalance automatically
* Applies SMOTE (if available)
* Falls back to class weights
* Switches evaluation metric to F1 when needed

### 📊 SHAP Explainability

* Computes SHAP values for model explainability
* Displays top features influencing predictions
* Helps identify real business drivers

### 🔬 Feature Engineering (Multi-layer)

* Domain-aware features: Telecom · Medical · Real Estate · Finance · HR
* Interactions, ratios, log transforms, polynomial features
* Only surfaces features that improve performance

### 🏗️ Production Pipeline

* Save trained model + preprocessing pipeline
* Load and predict on new/unseen data
* Handles missing columns and unseen categories

### 🏆 Experiment Tracking

* Logs every run automatically
* Leaderboard of models and scores
* Export results to CSV

### 🌐 REST API (Optional)

* FastAPI-based prediction server
* Ready-to-use endpoints for deployment

---

## 🆚 Why featuremind?

| Capability          | featuremind | Typical AutoML Tools |
| ------------------- | ----------- | -------------------- |
| One-line usage      | ✅           | ❌                    |
| Leakage detection   | ✅           | ❌                    |
| Reliability scoring | ✅           | ❌                    |
| SHAP explainability | ✅           | ⚠️                   |
| Production pipeline | ✅           | ✅                    |

---

## 📦 Installation

```bash
pip install featuremind

# (Recommended) Install advanced ML libraries
pip install xgboost lightgbm catboost shap imbalanced-learn

# Optional API support
pip install fastapi uvicorn python-multipart
```

---

## 🚀 Quick Start

```python
import featuremind as fm

fm.analyze("data.csv")
fm.check_leakage("data.csv", target="Churn")

pipeline = fm.train("data.csv", target="Churn")
pipeline.save("churn_pipeline")

pipeline = fm.load_pipeline("churn_pipeline")
results = pipeline.predict_df(new_data)

fm.get_tracker().leaderboard()

fm.serve("churn_pipeline/", port=8000)
```

---

## 🎬 Example Output

```text
🧠 featuremind v3.1.1 — Starting Analysis
🎯 Best Model   : LightGBM
📊 Score        : 0.8569 (F1-weighted)
🔒 Confidence   : High ✅
🛡️ Leakage      : None detected
```

---

## 📁 Project Structure

```
featuremind_project/
│
├── featuremind/
│   ├── analyzer.py
│   ├── feature_engineer.py
│   ├── evaluator.py
│   ├── leakage_guard.py
│   ├── importance.py
│   ├── reporter.py
│   ├── html_reporter.py
│   ├── insights.py
│   ├── pipeline.py
│   ├── tracker.py
│   └── api.py
│
├── setup.py
├── requirements.txt
├── test.py
└── README.md
```

---

## ⚠️ Notes

* High accuracy (>0.98) may indicate:

  * Data leakage
  * Highly separable datasets
  * Sampling bias

* Always validate models on unseen data.

---

## 📊 Output Files

* `featuremind_report.html` → Full analysis report
* `featuremind_report.png` → Feature visualization
* `enhanced_data.csv` → Dataset with engineered features
* `featuremind_experiments.csv` → Experiment logs
* `pipeline/` → Saved production model

---

## 💡 Use Cases

* Telecom churn prediction
* Fraud detection
* Healthcare predictions
* Real estate pricing
* HR analytics
* Any tabular ML problem

---

## 🔥 Why Developers Love featuremind

* ⚡ Go from raw data → model in **1 line**
* 🛡️ Built-in **leakage detection** (rare in AutoML)
* 📊 **Explainable AI (SHAP)** included by default
* 🧠 **Reliability scoring** (not just accuracy)
* 🏗️ Direct **production pipeline export**

👉 Not just AutoML — this is **AutoML + Trust Layer**

---

## 🔮 Roadmap

* Time-series support
* Deep learning integration
* Streamlit dashboard
* Cloud deployment

---

## 📄 License

MIT License

---

## 👩‍💻 Author

**Niveditha** — Data Scientist & ML Engineer  

🔗 LinkedIn: https://www.linkedin.com/in/niveditha-89ba04356/  
📦 PyPI: https://pypi.org/project/featuremind/  
💻 GitHub: https://github.com/Nivedithagowda2/featuremind

---
## 🎬 Demo Video

Click below to watch the full demo:

[▶️ Watch FeatureMind Demo](https://drive.google.com/file/d/1ta4vGuSruQpHMUtxmXqwdH7xvhJLY_4X/view)



> ⭐ If this project helps you, consider giving it a star!
