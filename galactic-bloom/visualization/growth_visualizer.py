import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Load growth simulation data
data = pd.read_csv('growth_simulation.csv')

# Define 3D plot function
def plot_3d(data, x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[x], data[y], data[z])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.show()

# Define PCA function
def apply_pca(data, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

# Apply PCA to reduce dimensionality
pca_data = apply_pca(data[['Temperature', 'Humidity', 'Light', 'Nutrient']], 2)

# Plot 2D PCA results
plt.scatter(pca_data[:, 0], pca_data[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Plot 3D growth simulation results
plot_3d(data, 'Temperature', 'Humidity', 'Light')

# Plot interactive 3D growth simulation results
from plotly.offline import iplot
import plotly.graph_objs as go

fig = go.Figure(data=[go.Scatter3d(
    x=data['Temperature'],
    y=data['Humidity'],
    z=data['Light'],
    mode='markers',
    marker=dict(
        size=5,
        color=data['Nutrient'],
        colorscale='Viridis',
        showscale=True
    )
)])
iplot(fig)
