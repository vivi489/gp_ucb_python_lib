import numpy as np
import pandas as pd

gp_param_dic = {
    "alpha": np.arange(-2, 2.01, 0.2),
    "beta": np.arange(-2, 2.01, 0.2),
    "n_cluster": np.arange(5, 20.1).astype(int)
}

gp_param2lda_param = {
    "alpha": lambda x: 10 ** x,
    "beta": lambda x: 10 ** x,
    "n_cluster": lambda x: x
}

# del gp_param_dic['n_cluster']
# del gp_param2lda_param['n_cluster']

sorted_keys = sorted(gp_param_dic.keys())

meshgrid = np.array(np.meshgrid(*[gp_param_dic[k] for k in sorted_keys]))

print(meshgrid.shape)
grid_df = pd.DataFrame(meshgrid.reshape(meshgrid.shape[0], -1).T, columns=["gp_" + x for x in sorted_keys])

for key in sorted_keys:
    grid_df[key] = grid_df["gp_" + key].apply(gp_param2lda_param[key])

print(grid_df)
grid_df.to_csv('lda_param_2dim.csv', index=False)