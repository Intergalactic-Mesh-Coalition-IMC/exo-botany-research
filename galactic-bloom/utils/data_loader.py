import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Define data loading function
def load_data(dataset):
    if dataset == 'iris':
        data = load_iris()
        return pd.DataFrame(data.data, columns=data.feature_names), data.target
    elif dataset == 'custom':
        # Load custom dataset from CSV file
        data = pd.read_csv('custom_dataset.csv')
        return data.drop('Target', axis=1), data['Target']
    else:
        raise ValueError('Invalid dataset')

# Define data splitting function
def split_data(X, y, test_size=0.2):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
