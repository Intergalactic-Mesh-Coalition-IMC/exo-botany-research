import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Define model selection function
def select_model(X, y):
    # Define models and hyperparameters to tune
    models = [
        RandomForestClassifier(),
        SVC(),
        LogisticRegression()
    ]
    hyperparameters = [
        {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10]},
        {'C': [1, 10, 100], 'kernel': ['linear', 'rbf']},
        {'C': [1, 10, 100], 'penalty': ['l1', 'l2']}
    ]

    # Perform grid search for each model
    results = []
    for model, hyperparameter in zip(models, hyperparameters):
        grid_search = GridSearchCV(model, hyperparameter, cv=5, scoring='accuracy')
        grid_search.fit(X, y)
        results.append((model.__class__.__name__, grid_search.best_score_, grid_search.best_params_))

    # Return the best model and its hyperparameters
    best_model, best_score, best_params = max(results, key=lambda x: x[1])
    print(f'Best model: {best_model} with score {best_score} and hyperparameters {best_params}')
    return best_model, best_params

# Define model ensembling function
def ensemble_models(models, X, y):
    # Define weights for each model
    weights = [1/len(models)] * len(models)

    # Perform weighted voting
    predictions = []
    for model, weight in zip(models, weights):
        predictions.append(model.predict(X) * weight)
    predictions = np.array(predictions).sum(axis=0)

    # Return the ensemble model
    return predictions
