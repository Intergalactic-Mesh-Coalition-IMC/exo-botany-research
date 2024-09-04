import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load plant data
plant_data = pd.read_csv('../processed/plant_data.csv')

# Preprocess data
X = plant_data.drop(['Species'], axis=1)
y = plant_data['Species']

# Encode species labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define decision tree classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Define random forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Define XGBoost classifier
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=10, max_depth=6, learning_rate=0.1, n_estimators=100, n_jobs=-1)
xgb_model.fit(X_train, y_train)

# Evaluate models
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

print(f'Decision Tree Accuracy: {dt_acc:.2f}')
print(f'Random Forest Accuracy: {rf_acc:.2f}')
print(f'XGBoost Accuracy: {xgb_acc:.2f}')

# Plot decision tree
plt.figure(figsize=(10, 8))
plot_tree(dt_model, filled=True)
plt.show()

# Plot feature importance
feature_importances = rf_model.feature_importances_
feature_names = X.columns
plt.barh(range(len(feature_names)), feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature Names')
plt.show()

# Save models
dt_model.save('dt_model.h5')
rf_model.save('rf_model.h5')
xgb_model.save('xgb_model.h5')
