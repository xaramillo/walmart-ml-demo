import shap

import lime.lime_tabular

from sklearn.inspection import PartialDependenceDisplay
import numpy as np


"""
Explainability functions for multiclass classification models.
Includes SHAP, LIME, and PDP.
"""

def explain_shap(modelo, X_train, X_test, feature_names, output_dir='explainability', model_name="XGBoost"):
    """
    Generates SHAP plots (summary, force, dependence) for the model and saves them in output_dir.
    """
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_test)
    n_features = X_test.shape[1]
    if len(feature_names) != n_features:
        raise ValueError(f"feature_names len: {len(feature_names)}, X_test len: {n_features} mismatch")
    feature_names = list(feature_names)
    # Handle multiclass: shap_values is a list of arrays (one per class)
    if isinstance(shap_values, list):
        # For multiclass, average the absolute SHAP values across all classes and samples
        importances = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # For binary/classic, just take the mean absolute value
        importances = np.abs(shap_values).mean(axis=0)
    # Ensure all importances are float scalars (robust: always reduce to scalar)
    shap_importance = {fname: float(np.mean(val)) for fname, val in zip(feature_names, importances)}
    # Sort by importance
    shap_ranking = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
    return shap_ranking, shap_importance

def explain_lime(modelo, X_train, X_test, feature_names, class_names, idx=0, output_dir='explainability', model_name="XGBoost"):
    """
    Generates a local LIME explanation for one instance and saves it in output_dir.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    exp = explainer.explain_instance(X_test[idx], modelo.predict_proba)
    lime_explanation = exp.as_list()
    return lime_explanation

def explain_pdp(modelo, X_test, feature_names, top2_idx, output_dir='explainability', model_name="XGBoost"):
    """
    Generates PDP plots for the two most important features and saves them in output_dir.
    """
    pdp_result = PartialDependenceDisplay.from_estimator(
        modelo, X_test, features=top2_idx, feature_names=feature_names
    )
    # Extrae los valores PDP y los devuelve como dict
    pdp_dict = {}
    for i, idx in enumerate(top2_idx):
        fname = feature_names[idx]
        values = pdp_result.pd_lines[i][1] if hasattr(pdp_result, 'pd_lines') else None
        pdp_dict[fname] = values
    return pdp_dict
