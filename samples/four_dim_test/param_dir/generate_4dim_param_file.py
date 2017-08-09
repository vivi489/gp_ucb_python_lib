import os

import numpy as np
import pandas as pd

from gphypo.util import mkdir_if_not_exist

gp_param_dic = {
    'x': np.arange(-3, 4, 1),
    'y': np.arange(-3, 4, 1),
    'z': np.arange(-3, 4, 1),
    'w': np.arange(-3, 4, 1)

}

gp_param2gaussian_param = {
    "x": lambda x: x,
    "y": lambda x: x,
    "z": lambda x: x,
    "w": lambda x: x
}
output_dir = 'csv_files'
mkdir_if_not_exist(output_dir)

for k, v in gp_param_dic.items():
    output_filename = os.path.join(output_dir, k) + ".csv"
    res = pd.DataFrame({
        k: v
    })
    res['gp_' + k] = res[k].apply(gp_param2gaussian_param[k])

    res.to_csv(output_filename, index=False)
    print(output_filename + " was created")
