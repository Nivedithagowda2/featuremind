import featuremind as fm

# Run analysis
fm.analyze("data/sample_data.csv", target="Churn")

# Train pipeline
pipeline = fm.train("data/sample_data.csv", target="Churn")

# Save pipeline
pipeline.save("demo_pipeline")