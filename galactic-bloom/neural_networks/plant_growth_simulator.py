import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define plant growth model parameters
params = {
    'k1': 0.1,  # growth rate
    'k2': 0.05,  # decay rate
    'k3': 0.01,  # nutrient uptake rate
    'k4': 0.005,  # water uptake rate
    'N0': 10,  # initial nutrient concentration
    'W0': 50,  # initial water concentration
    'B0': 0.1,  # initial biomass
    't_end': 30,  # simulation time
    'dt': 0.1  # time step
}

# Define plant growth model equations
def plant_growth_model(state, t, params):
    B, N, W = state
    k1, k2, k3, k4, N0, W0, B0, t_end, dt = params.values()
    dBdt = k1 * B * (1 - B / B0) - k2 * B
    dNdt = -k3 * B * N / (N + N0)
    dWdt = -k4 * B * W / (W + W0)
    return [dBdt, dNdt, dWdt]

# Simulate plant growth
t = np.arange(0, params['t_end'], params['dt'])
state0 = [params['B0'], params['N0'], params['W0']]
state = odeint(plant_growth_model, state0, t, args=(params,))

# Plot simulation results
plt.plot(t, state[:, 0], label='Biomass')
plt.plot(t, state[:, 1], label='Nutrient Concentration')
plt.plot(t, state[:, 2], label='Water Concentration')
plt.xlabel('Time (days)')
plt.ylabel('Concentration')
plt.legend()
plt.show()
