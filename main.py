import os
import mlflow
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

    # Data ingestion
    df = ingest.load_data(data_path)

    # Define numerical and categorical variables
    numeric_features = ['volt', 'rotate', 'pressure', 'vibration']
    categorical_features = []

    # Preprocessing
    preprocessor = preprocess.preprocessing_pipeline(numeric_features, categorical_features)
    X = df.drop(columns=[target_column])
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    target_preprocessor = preprocess.preprocessing_target_pipeline([target_column])
    y = df[target_column].to_frame()
    y_processed = target_preprocessor.fit_transform(y).ravel()

    # Training
    mlflow.set_experiment('ml-pipeline')
    with mlflow.start_run() as run:
        modelo, X_test, y_test = train.train_xgboost(X_processed, y_processed)

        # Evaluation
        resultados = evaluate.evaluate_model(modelo, X_test, y_test, labels=np.unique(y_processed))
        for k, v in resultados.items():
            if k not in ['confusion_matrix', 'classification_report', 'y_pred']:
                mlflow.log_metric(k, v)

        # Save confusion matrix as artifact
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

        # Explainability
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        class_names = [str(c) for c in np.unique(y_processed)]
        shap_paths, top_feature = explainability.explain_shap(
            modelo, X_processed_df, X_test_df, feature_names, model_name="XGBoost"
        )
        for path in shap_paths:
            mlflow.log_artifact(path)
        lime_path = explainability.explain_lime(
            modelo, X_processed_df, X_test_df, feature_names, class_names, idx=0, model_name="XGBoost"
        )
        mlflow.log_artifact(lime_path)
        importances = np.abs(modelo.feature_importances_)
        top2 = np.argsort(importances)[-2:]
        pdp_path = explainability.explain_pdp(
            modelo, X_test_df, feature_names, top2, model_name="XGBoost"
        )
        mlflow.log_artifact(pdp_path)

        # Save models and parameters
        mlflow.sklearn.log_model(modelo, 'XGBoost')
        mlflow.sklearn.log_model(preprocessor, 'preprocessor')
        mlflow.sklearn.log_model(target_preprocessor, 'target_preprocessor')
        mlflow.log_param('data_path', data_path)
        mlflow.log_param('target_column', target_column)
        mlflow.log_param('numeric_features', numeric_features)
        mlflow.log_param('categorical_features', categorical_features)
        mlflow.log_param('xgb_hyperparameters', modelo.get_params())
        mlflow.log_param('xgb_top_feature_shap', top_feature)
        if os.path.exists('requirements.txt'):
            mlflow.log_artifact('requirements.txt')
if __name__ == "__main__":
    main()