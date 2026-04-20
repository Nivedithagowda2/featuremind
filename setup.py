from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="featuremind",
    version="3.1.1",
    author="Niveditha",
    description="Universal AutoML + Feature Engineering + Explainability Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    packages=find_packages(),
    python_requires=">=3.8",

    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
    ],

    extras_require={
        "full": [
            "xgboost",
            "lightgbm",
            "catboost",
            "shap",
            "imbalanced-learn",
        ],
        "api": [
            "fastapi",
            "uvicorn",
            "python-multipart",
        ],
    },

    entry_points={
        "console_scripts": [
            "featuremind=featuremind.cli:main"
        ]
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    license="MIT",
    url="https://github.com/Nivedithagowda2/featuremind",
)