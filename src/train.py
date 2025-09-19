from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import os
from joblib import dump


def save_best_model(model, output_dir, output_name):
    """
    Saves the best trained model in joblib format.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dump(model, os.path.join(output_dir, str(output_name) + '.joblib'))



def train_xgboost(X, y, output_dir='models/', seed=23):
    """
    Trains an XGBoost classifier to predict the 'failure' column.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    param_grid = {
        'n_estimators': [1000],# [500, 1000], #1000
        'max_depth': [None],# [None, 10], #None
        'learning_rate': [0.1],#[0.01, .1], #.1
    }
    grid = GridSearchCV(
        XGBClassifier(eval_metric='mlogloss', random_state=seed),
        param_grid=param_grid,
        cv=2,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    save_best_model(model=best_model, output_dir=output_dir, output_name="xg_boost")
    print("Best values for XGBoost:")
    print(grid.best_params_)
    return best_model, X_test, y_test