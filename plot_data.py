# Description: This script has a method which plots the position of the input data in an interacive 3D plot.

import plotly.express as px

def plot_3d(input_data):
    positions = input_data[:, 0:3]

    fig = px.scatter_3d(positions, x=0, y=1, z=2, color=0, size_max=18, opacity=0.7, title="Position of the input data in 3D")
    fig.show()