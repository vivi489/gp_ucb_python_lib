import numpy as np
import pandas as pd

gp_param_dic = {
    'x': np.arange(-5, 5, 0.5),
    'y': np.arange(-5, 5, 0.5)
}

gp_param2lda_param = {
    "x": lambda x: x,
    "y": lambda x: x
}

sorted_keys = sorted(gp_param_dic.keys())

meshgrid = np.array(np.meshgrid(*[gp_param_dic[k] for k in sorted_keys]))

print(meshgrid.shape)
grid_df = pd.DataFrame(meshgrid.reshape(meshgrid.shape[0], -1).T, columns=["gp_" + x for x in sorted_keys])

for key in sorted_keys:
    grid_df[key] = grid_df["gp_" + key].apply(gp_param2lda_param[key])

print(grid_df)
grid_df.to_csv('gaussian_param_2dim.csv', index=False)
