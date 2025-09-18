from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
import numpy as np

def evaluate_model(model, X_test, y_test, average='weighted', zero_division=0, labels=None):
    """
    Evaluates the model using multiple metrics and returns a dictionary with the results.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average=average, zero_division=zero_division)
    prec = precision_score(y_test, y_pred, average=average, zero_division=zero_division)
    f1 = f1_score(y_test, y_pred, average=average, zero_division=zero_division)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=zero_division)
    print(f"Accuracy: {acc}")
    print(f"Recall: {rec}")
    print(f"Precision: {prec}")
    print(f"F1-score: {f1}")
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=zero_division))
    return {
        'accuracy': acc,
        'recall': rec,
        'precision': prec,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred': y_pred
    }