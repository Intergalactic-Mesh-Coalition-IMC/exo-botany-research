import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define plant growth model
def plant_growth(state, t, params):
    temperature, humidity, light, nutrient = state
    growth_rate, water_absorption, photosynthesis_rate = params
    dtemperature_dt = growth_rate * temperature * (1 - temperature / 100)
    dhumidity_dt = water_absorption * humidity * (1 - humidity / 100)
    dlight_dt = photosynthesis_rate * light * (1 - light / 100)
    dnutrient_dt = growth_rate * nutrient * (1 - nutrient / 100)
    return [dtemperature_dt, dhumidity_dt, dlight_dt, dnutrient_dt]

# Define simulation parameters
params = [0.1, 0.2, 0.3]  # growth rate, water absorption, photosynthesis rate
initial_state = [20, 50, 30, 40]  # initial temperature, humidity, light, nutrient
t = np.arange(0, 100, 0.1)  # time array

# Run simulation
state = odeint(plant_growth, initial_state, t, args=(params,))
temperature, humidity, light, nutrient = state.T

# Plot simulation results
plt.plot(t, temperature, label='Temperature')
plt.plot(t, humidity, label='Humidity')
plt.plot(t, light, label='Light')
plt.plot(t, nutrient, label='Nutrient')
plt.xlabel('Time')
plt.ylabel('State')
plt.legend()
plt.show()

# Save simulation results
pd.DataFrame({'Time': t, 'Temperature': temperature, 'Humidity': humidity, 'Light': light, 'Nutrient': nutrient}).to_csv('growth_simulation.csv', index=False)
