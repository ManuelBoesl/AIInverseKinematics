# Description: This script has a method which plots the position of the input data in an interacive 3D plot.

import plotly.express as px

def plot_3d(input_data):
    positions = input_data[:, 0:3]

    fig = px.scatter_3d(positions, x=0, y=1, z=2, color=0, size_max=18, opacity=0.7, title="Position of the input data in 3D")
    # set the range for the axes between -1 and 1 and add x, y, z labels
    fig.update_layout(scene=dict(xaxis=dict(range=[-1, 1], title='X'),
                                yaxis=dict(range=[-1, 1], title='Y'),
                                zaxis=dict(range=[-1, 1], title='Z')))
    fig.show()