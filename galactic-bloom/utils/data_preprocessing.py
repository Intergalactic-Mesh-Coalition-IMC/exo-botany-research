import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Define data preprocessing function
def preprocess_data(data):
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data[['Temperature', 'Humidity', 'Light', 'Nutrient']] = imputer.fit_transform(data[['Temperature', 'Humidity', 'Light', 'Nutrient']])

    # Encode categorical variables
    le = LabelEncoder()
    data['Plant Species'] = le.fit_transform(data['Plant Species'])

    # Scale numerical variables
    scaler = StandardScaler()
    data[['Temperature', 'Humidity', 'Light', 'Nutrient']] = scaler.fit_transform(data[['Temperature', 'Humidity', 'Light', 'Nutrient']])

    return data

# Define data transformation function
def transform_data(data):
    # Create polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2)
    data_poly = poly.fit_transform(data[['Temperature', 'Humidity', 'Light']])

    # Create interaction features
    from sklearn.preprocessing import InteractionFeatures
    interaction = InteractionFeatures()
    data_interaction = interaction.fit_transform(data[['Temperature', 'Humidity', 'Light']])

    return pd.concat([data, pd.DataFrame(data_poly, columns=poly.get_feature_names(['Temperature', 'Humidity', 'Light'])), pd.DataFrame(data_interaction, columns=interaction.get_feature_names(['Temperature', 'Humidity', 'Light']))], axis=1)

# Define data preprocessing pipeline
def create_preprocessing_pipeline():
    numeric_features = ['Temperature', 'Humidity', 'Light', 'Nutrient']
    categorical_features = ['Plant Species']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', LabelEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor
