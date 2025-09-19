import os
import mlflow
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import ingest, preprocess, train, evaluate, explainability

def main():
    """
    Main orchestrator of the ML pipeline: ingestion, preprocessing, training, evaluation, and explainability.
    """
    data_path = 'data/dataset.csv'
    target_column = 'failure'

    print("[INFO] Starting data ingestion...")
    df = ingest.load_data(data_path)
    print(f"[INFO] Data loaded: shape={df.shape}")

    print("[INFO] Defining numerical and categorical variables...")
    numeric_features = ['volt', 'rotate', 'pressure', 'vibration']
    categorical_features = []
    print(f"[INFO] Numerical features: {numeric_features}")
    print(f"[INFO] Categorical features: {categorical_features}")

    print("[INFO] Starting feature preprocessing...")
    preprocessor = preprocess.preprocessing_pipeline(numeric_features, categorical_features)
    X = df.drop(columns=[target_column])
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    print(f"[INFO] X_processed shape: {X_processed.shape}")
    print(f"[INFO] feature_names: {feature_names}")

    print("[INFO] Preprocessing target variable...")
    target_preprocessor = preprocess.preprocessing_target_pipeline([target_column])
    y = df[target_column].to_frame()
    y_processed = target_preprocessor.fit_transform(y).ravel()
    print(f"[INFO] y_processed shape: {y_processed.shape}")

    print("[INFO] Starting model training...")
    mlflow.set_experiment('ml-pipeline')
    with mlflow.start_run() as run:
        model, X_test, y_test = train.train_xgboost(X_processed, y_processed)
    print(f"[INFO] Model trained. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    print("[INFO] Evaluating the model...")
    resultados = evaluate.evaluate_model(model, X_test, y_test, labels=np.unique(y_processed))
    for k, v in resultados.items():
        if k not in ['confusion_matrix', 'classification_report', 'y_pred']:
            mlflow.log_metric(k, v)
    print("[INFO] Evaluation completed. Main metrics logged to MLflow.")

    print("[INFO] Saving confusion matrix as artifact...")
    plt.figure(figsize=(8,6))
    cm = resultados['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_processed), yticklabels=np.unique(y_processed))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    fname = 'confusion_matrix.png'
    plt.savefig(fname)
    plt.close()
    mlflow.log_artifact(fname)
    os.remove(fname)
    print("[INFO] Confusion matrix saved and logged.")

    print("[INFO] Running explainability (SHAP, LIME, PDP)...")
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    class_names = [str(c) for c in np.unique(y_processed)]
    # SHAP Section
    shap_ranking, shap_importance = explainability.explain_shap(
            model, X_processed_df, X_test_df, feature_names, model_name="XGBoost"
        )
    print(f"[INFO] SHAP ranking: {shap_ranking}")
    # Log top feature and the full ranking as parameters and as a json file
    if shap_ranking:
        top_feature = shap_ranking[0][0]
        mlflow.log_param('xgb_top_feature_shap', top_feature)
    
        with open('shap_ranking.json', 'w') as f:
            json.dump(shap_ranking, f)
        mlflow.log_artifact('shap_ranking.json')
        os.remove('shap_ranking.json')
        with open('shap_importance.json', 'w') as f:
            json.dump(shap_importance, f)
        mlflow.log_artifact('shap_importance.json')
        os.remove('shap_importance.json')
    # LIME Section
    lime_explanation = explainability.explain_lime(
        model, X_processed_df, X_test_df, feature_names, class_names, idx=0, model_name="XGBoost"
    )
    print(f"[INFO] LIME local explanation idx=0: {lime_explanation}")
    with open('lime_explanation.json', 'w') as f:
        json.dump(lime_explanation, f)
    mlflow.log_artifact('lime_explanation.json')
    os.remove('lime_explanation.json')
    # PDP: value tables for the top 2 most important features
    importances = np.abs(model.feature_importances_)
    top2 = np.argsort(importances)[-2:]
    pdp_dict = explainability.explain_pdp(
        model, X_test_df, feature_names, top2, model_name="XGBoost"
    )
    print(f"[INFO] PDP dict: {pdp_dict}")
    with open('pdp_dict.json', 'w') as f:
        json.dump(pdp_dict, f)
    mlflow.log_artifact('pdp_dict.json')
    os.remove('pdp_dict.json')
    print("[INFO] Explainability completed and logged.")

    print("[INFO] Saving models and parameters to MLflow...")
    mlflow.sklearn.log_model(model, 'XGBoost')
    mlflow.sklearn.log_model(preprocessor, 'preprocessor')
    mlflow.sklearn.log_model(target_preprocessor, 'target_preprocessor')
    mlflow.log_param('data_path', data_path)
    mlflow.log_param('target_column', target_column)
    mlflow.log_param('numeric_features', numeric_features)
    mlflow.log_param('categorical_features', categorical_features)
    mlflow.log_param('xgb_hyperparameters', model.get_params())
    mlflow.log_param('xgb_top_feature_shap', top_feature)
    if os.path.exists('requirements.txt'):
        mlflow.log_artifact('requirements.txt')
    print("[INFO] Pipeline finished successfully.")
if __name__ == "__main__":
    main()