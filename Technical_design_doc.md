
# Technical design document

## Description of the repo

This repo contains a complete and modular pipeline for the development, training, evaluation, interpretation, and deployment of machine learning models, with a focus on engineering best practices and experiment traceability using MLflow. It includes scripts for preprocessing, training, evaluation, interpretability , and Streamlit application.

---

## Project Structure

```
root/
├── app/                        # Interactive applications and scripts
│   ├── streamlit_serve.py      # Streamlit app for prediction
├── data/                       # Data (not included in the repo)
│   └── raw/                    # Raw data
├── models/                     # Trained models (saved as .joblib)
├── src/                        # Pipeline source code
│   ├── ingest.py               # Data ingestion
│   ├── preprocess.py           # Data preprocessing
│   ├── train.py                # Model training
│   ├── evaluate.py             # Model evaluation
├── main.py                     # Main pipeline (end-to-end)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image for reproducibility/deployment
├── run_streamlit.sh            # Script to launch the Streamlit app
└── README.md                   # Project documentation
```

## Pipeline Execution

This performs:
- Data ingestion and preprocessing
- Training of Random Forest and XGBoost (with GridSearch)
- Evaluation and logging of metrics, parameters, and artifacts in MLflow
- Automatic interpretability (SHAP, LIME, PDP)
- Model registration and versioning

---

## Interpretability and Explainability

The pipeline automatically generates:
- **SHAP**: summary plots (bar and dot), local force plot, dependence plot of the most important feature.
- **LIME**: local explanation for one instance.
- **PDP**: partial dependence plots for the two most important features.

All artifacts are logged in MLflow and can be viewed from the UI.

---

## Best Practices

- **MLflow**: All experiments, metrics, parameters, and artifacts are automatically logged.
- **Docker**: The environment is reproducible and portable.
- **Modularity**: The code is organized by pipeline stages.
- **Interpretability**: Includes automatic explainability to facilitate decision making.

---
