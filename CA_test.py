# %%
import numpy as np
import pandas as pd
import plotly.express as px
# from scipy.signal import convolve2d




rows, cols = 40,40

# lab=np.zeros((rows, cols))
lab=np.random.randint(-1, 1, (rows+1, cols+1))
# lab=np.array([
#     [ 1,  0,  0, -1,  0, -1],
#     [ 0, -1,  0,  0,  0,  0],
#     [ 0,  0,  0,  0, -1,  0],
#     [ 0, -1, -1, -1, -1, -1],
#     [ 0,  0, -1,  0,  0,  0],
#     [ 0,  0,  0,  0, -1,  1]
# ])
lab[0, 0] = 1
# lab[rows, cols] = 1

rule = [
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
]
px.imshow(lab)

# %%
def action(i:int, j:int, arr: np.ndarray)->np.ndarray:
    current = arr[i,j]
    window = np.pad(arr, 1)[i: (i+3), j: j+3]
    if window.max() and current != -1:
        return 1
    return current
    # return (window@rule).max()


def tick(lab: np.ndarray):
    new_lab = lab.copy()
    for i in range(lab.shape[0]):
        for j in range(lab.shape[1]):
            new_lab[i,j] = action(i,j, lab)
    return new_lab

ticks = [lab]
ticks.append(tick(ticks[-1]))
while not (ticks[-2] == ticks[-1]).all() and len(ticks) < 70:
    ticks.append(tick(ticks[-1]))
    

# %%
import plotly.graph_objs as go
fig = go.Figure(
    data=[go.Heatmap(z=ticks[0])],
    layout=go.Layout(
        title="Frame 0",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]
    ),
    frames=[go.Frame(data=[go.Heatmap(z=ticks[i])],
                     layout=go.Layout(title_text=f"Frame {i}"))
            for i in range(1, len(ticks))]
)

fig.show()
# %%
