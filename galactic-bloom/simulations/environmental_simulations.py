import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Define environmental simulation model
def environmental_simulation(state, t, params):
    temperature, humidity, light, co2 = state
    temperature_mean, temperature_std, humidity_mean, humidity_std, light_mean, light_std, co2_mean, co2_std = params
    dtemperature_dt = norm.rvs(temperature_mean, temperature_std)
    dhumidity_dt = norm.rvs(humidity_mean, humidity_std)
    dlight_dt = norm.rvs(light_mean, light_std)
    dco2_dt = norm.rvs(co2_mean, co2_std)
    return [dtemperature_dt, dhumidity_dt, dlight_dt, dco2_dt]

# Define simulation parameters
params = [20, 5, 50, 10, 30, 5, 400, 50]  # temperature mean, temperature std, humidity mean, humidity std, light mean, light std, co2 mean, co2 std
initial_state = [20, 50, 30, 400]  # initial temperature, humidity, light, co2
t = np.arange(0, 100, 0.1)  # time array

# Run simulation
state = np.zeros((len(t), 4))
state[0] = initial_state
for i in range(1, len(t)):
    state[i] = state[i-1] + environmental_simulation(state[i-1], t[i], params)

# Plot simulation results
plt.plot(t, state[:, 0], label='Temperature')
plt.plot(t, state[:, 1], label='Humidity')
plt.plot(t, state[:, 2], label='Light')
plt.plot(t, state[:, 3], label='CO2')
plt.xlabel('Time')
plt.ylabel('State')
plt.legend()
plt.show()

# Save simulation results
pd.DataFrame({'Time': t, 'Temperature': state[:, 0], 'Humidity': state[:, 1], 'Light': state[:, 2], 'CO2': state[:, 3]}).to_csv('environmental_simulation.csv', index=False)
