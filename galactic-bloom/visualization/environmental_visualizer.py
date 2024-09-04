import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load environmental simulation data
data = pd.read_csv('environmental_simulation.csv')

# Define heatmap function
def plot_heatmap(data, x, y):
    plt.figure(figsize=(10, 10))
    sns.heatmap(data[[x, y]].corr(), annot=True, cmap='coolwarm', square=True)
    plt.show()

# Define word cloud function
def plot_wordcloud(data, column):
    wordcloud = WordCloud().generate(' '.join(data[column].astype(str)))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# Plot heatmap of environmental variables
plot_heatmap(data, 'Temperature', 'Humidity')

# Plot word cloud of CO2 levels
plot_wordcloud(data, 'CO2')

# Plot interactive time series plot of environmental variables
from plotly.offline import iplot
import plotly.graph_objs as go

fig = go.Figure(data=[go.Scatter(
    x=data['Time'],
    y=data['Temperature'],
    name='Temperature'
), go.Scatter(
    x=data['Time'],
    y=data['Humidity'],
    name='Humidity'
), go.Scatter(
    x=data['Time'],
    y=data['Light'],
    name='Light'
), go.Scatter(
    x=data['Time'],
    y=data['CO2'],
    name='CO2'
)])
iplot(fig)
