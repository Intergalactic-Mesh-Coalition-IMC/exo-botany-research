import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Define experiment manager function
def run_experiment(model, X, y, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))
    return np.mean(scores)

# Define hyperparameter tuning function
def tune_hyperparameters(model, X, y, hyperparameters):
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(model, hyperparameters, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    return grid_search.best_score_, grid_search.best_params_
