import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import os

"""
Explainability functions for multiclass classification models.
Includes SHAP, LIME, and PDP.
"""

def explain_shap(modelo, X_train, X_test, feature_names, output_dir='explainability', model_name="XGBoost"):
    """
    Generates SHAP plots (summary, force, dependence) for the model and saves them in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_test)
    # Summary plot
    summary_path = os.path.join(output_dir, f'shap_summary_{model_name}.png')
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(summary_path)
    plt.close()
    # Dependence plot para la feature más importante
    importances = np.abs(shap_values).mean(axis=0)
    top_feature = np.argmax(importances)
    dep_path = os.path.join(output_dir, f'shap_dependence_{model_name}.png')
    plt.figure()
    shap.dependence_plot(top_feature, shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(dep_path)
    plt.close()
    return [summary_path, dep_path], top_feature

def explain_lime(modelo, X_train, X_test, feature_names, class_names, idx=0, output_dir='explainability', model_name="XGBoost"):
    """
    Generates a local LIME explanation for one instance and saves it in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    exp = explainer.explain_instance(X_test[idx], modelo.predict_proba)
    lime_path = os.path.join(output_dir, f'lime_{model_name}_idx{idx}.png')
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(lime_path)
    plt.close(fig)
    return lime_path

def explain_pdp(modelo, X_test, feature_names, top2_idx, output_dir='explainability', model_name="XGBoost"):
    """
    Generates PDP plots for the two most important features and saves them in output_dir.
    """
    from sklearn.inspection import plot_partial_dependence
    os.makedirs(output_dir, exist_ok=True)
    pdp_path = os.path.join(output_dir, f'pdp_{model_name}.png')
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_partial_dependence(modelo, X_test, features=top2_idx, feature_names=feature_names, ax=ax)
    plt.tight_layout()
    plt.savefig(pdp_path)
    plt.close(fig)
    return pdp_path
