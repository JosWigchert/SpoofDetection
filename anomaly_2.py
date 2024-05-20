import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# Load benign and anomalous datasets
benign = pd.read_csv("data/Satellite.csv")
anomalous = pd.read_csv("data/Ground.csv")

# Define categorical columns if any
categorical_cols = []  # Update this if you have categorical columns

# Define numerical columns
numerical_cols = benign.columns

# Define preprocessing steps for numerical and categorical data
numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[("encoder", OrdinalEncoder())])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Combine preprocessing and model into a single pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", IsolationForest(contamination=0.005)),
    ]
)  # You can adjust the contamination parameter as needed

# Fit the pipeline to the benign data
pipeline.fit(benign)

# Predict anomaly scores in both datasets
benign_scores = pipeline.decision_function(benign)
anomalous_scores = pipeline.decision_function(anomalous)

# Add anomaly scores to the datasets
benign["anomaly_score"] = benign_scores
anomalous["anomaly_score"] = anomalous_scores

# Print datasets
print("Benign Data:")
print(benign.head())
print("\nAnomalous Data:")
print(anomalous.head())

# Plot benign data
plt.figure(figsize=(10, 5))

# Plot benign data
plt.hist(benign["anomaly_score"], bins=40, color="blue", alpha=0.5, label="Benign")

# Plot anomalous data
plt.hist(anomalous["anomaly_score"], bins=40, color="red", alpha=0.5, label="Anomalous")

plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.title("Distribution of Anomaly Scores")
plt.legend()
plt.grid(True)
plt.show()
