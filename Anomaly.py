import pandas as pd
import seaborn as sns
from pyod.models.mad import MAD
from sklearn.preprocessing import OrdinalEncoder


def anomaly_detection_with_mad(benign, anomalous):
    mad_model = MAD()
    mad_model.fit(benign)

    normal = mad_model.predict(benign)
    outliers = mad_model.predict(anomalous)

    print("Normal:", normal)
    print("Normal:", outliers)

    return outliers


benign = pd.read_csv("data/Satellite.csv")
anomalous = pd.read_csv("data/Ground.csv")

# anomalies = anomaly_detection_with_mad(benign, anomalous)
# print("Anomalies:", anomalies)


benign.info()
anomalous.info()


import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

# Assuming df is your dataframe
# Load your dataframe

# Define categorical columns if any
categorical_cols = []

# Separate numerical columns
numerical_cols = df.columns.difference(categorical_cols)

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

# Define your anomaly detection model
model = IsolationForest(
    contamination=0.05
)  # You can adjust the contamination parameter as needed

# Combine preprocessing and model into a single pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Fit the pipeline to your data
pipeline.fit(df)

# Predict outliers/anomalies
predictions = pipeline.predict(df)

# Add the predictions to your dataframe
df["anomaly"] = predictions

# Anomalies will have a value of -1, normal points will have a value of 1
# You can further analyze the anomalies based on your requirements
