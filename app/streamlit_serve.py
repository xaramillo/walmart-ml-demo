import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import mlflow

# Add the root directory to sys.path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import preprocessing_pipeline


# Load XGBoost model
MODEL_DIR = "./models"
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xg_boost.joblib")

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

st.title("Operational Event Prediction")


model = load_model(XGB_MODEL_PATH)
model_loaded = model is not None
model_name = "XGBoost"


if not model_loaded:
    st.error(f"Trained model '{model_name}' not found. Please run the training pipeline and ensure the file exists in '{MODEL_DIR}'.")
    st.stop()

# Load preprocessing pipeline


# Define features for this use case
numeric_features = ["volt", "rotate", "pressure", "vibration"]
categorical_features = []
preprocessor = preprocessing_pipeline(numeric_features, categorical_features)


# User input
st.header("Enter input values to predict 'failure'")
input_data = {}
for col in numeric_features:
    input_data[col] = st.number_input(f"{col}", value=0.0)
input_df = pd.DataFrame([input_data])

# Preprocess and predict
if st.button("Predict"):
    try:
        X_proc = preprocessor.transform(input_df)
        pred = model.predict(X_proc)
        st.success(f"'failure' prediction: {pred[0]}")
        # Log prediction in MLflow
        with mlflow.start_run(run_name="streamlit_prediction", nested=True):
            mlflow.log_params(input_data)
            mlflow.log_metric("prediction", str(pred[0]))
    except Exception as e:
        st.error(f"Prediction error: {e}")