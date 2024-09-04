# Define model evaluation function
def evaluate_model(model, X, y):
    # Evaluate model using cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print('Cross-validation scores:', scores)
    print('Mean cross-validation score:', np.mean(scores))

    # Evaluate model using classification metrics
    y_pred = model.predict(X)
    print('Accuracy:', accuracy_score(y, y_pred))
    print('Classification Report:')
    print(classification_report(y, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y, y_pred))

# Define model selection function
def select_model(models, X, y):
    # Evaluate each model using cross-validation
    scores = []
    for model in models:
        scores.append(np.mean(cross_val_score(model, X, y, cv=5, scoring='accuracy')))
    best_model_index = np.argmax(scores)
    return models[best_model_index]

# Define model comparison function
def compare_models(models, X, y):
    # Evaluate each model using cross-validation
    scores = []
    for model in models:
        scores.append(np.mean(cross_val_score(model, X, y, cv=5, scoring='accuracy')))
    print('Model Comparison:')
    for i, model in enumerate(models):
        print(f'Model {i+1}: {model.__class__.__name__} with score {scores[i]}')

# Define feature importance evaluation function
def evaluate_feature_importance(model, X, y):
    # Evaluate feature importance using permutation importance
    from sklearn.inspection import permutation_importance
    importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    print('Feature Importance:')
    for i, feature in enumerate(X.columns):
        print(f'{feature}: {importance.importances_mean[i]}')
