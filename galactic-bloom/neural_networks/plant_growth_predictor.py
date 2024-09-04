import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load plant growth data
plant_growth_data = pd.read_csv('../processed/plant_growth_data.csv')

# Preprocess data
X = plant_growth_data.drop(['Biomass (g)'], axis=1)
y = plant_growth_data['Biomass (g)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define random forest regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Define LSTM neural network model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')

# Train LSTM model
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate models
rf_mse = mean_squared_error(y_test, rf_model.predict(X_test))
lstm_mse = mean_squared_error(y_test, lstm_model.predict(X_test))

print(f'Random Forest MSE: {rf_mse:.2f}')
print(f'LSTM MSE: {lstm_mse:.2f}')

# Save models
rf_model.save('rf_model.h5')
lstm_model.save('lstm_model.h5')
