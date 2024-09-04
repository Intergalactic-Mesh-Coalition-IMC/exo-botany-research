# Define hyperparameter tuning settings
HYPERPARAMETERS = {
    'RandomForestClassifier': {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10]
    },
    'SVC': {
        'C': [1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    'LogisticRegression': {
        'C': [1, 10, 100],
        'penalty': ['l1', 'l2'],
        'max_iter': [100, 500, 1000]
    }
}

# Define hyperparameter tuning search space
SEARCH_SPACE = {
    'RandomForestClassifier': {
        'n_estimators': {'low': 10, 'high': 100},
        'max_depth': {'low': 5, 'high': 10}
    },
    'SVC': {
        'C': {'low': 1, 'high': 100},
        'gamma': {'low': 0.1, 'high': 10}
    },
    'LogisticRegression': {
        'C': {'low': 1, 'high': 100},
        'max_iter': {'low': 100, 'high': 1000}
    }
}
