# %%
import numpy as np
import pandas as pd
import plotly.express as px
# from scipy.signal import convolve2d


# state = [0,1,2]
state_len = 3
state = np.random.randint(0,5, state_len).tolist()
state =[3,2,1]

states = [state]
for generation in range(10):
    actions = []
    for i in range(state_len):
        right = state[i+1] if i<state_len - 1 else np.inf
        current = state[i]
        left = state[i-1] if i>0 else -np.inf

        if not left < current:
            actions.append(-1)
            continue

        if not current < right:
            actions.append(+1)
            continue
        
        actions.append(0)
    
    
    if states[-1] == new_state:
        break
    states.append(new_state)


n_generations = len(states)
df = pd.DataFrame({
    "generation": [[generation]*state_len for generation in range(n_generations)],
    "index": [list(range(state_len))]*n_generations,
    "value": states,
}).apply(pd.Series.explode).reset_index()

px.bar(df, x='index', y = "value", animation_frame="generation")
# %%
