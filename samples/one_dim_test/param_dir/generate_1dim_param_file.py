import os

import numpy as np
import pandas as pd

from gphypo.util import mkdir_if_not_exist

bo_param_dic = {
    'x': np.arange(-5, 5.1, 0.25)
}

bo_param2gaussian_param = {
    "x": lambda x: x
}

output_dir = 'csv_files'
mkdir_if_not_exist(output_dir)

for k, v in bo_param_dic.items():
    output_filename = os.path.join(output_dir, k) + ".csv"
    res = pd.DataFrame({
        k: v
    })
    res['bo_' + k] = res[k].apply(bo_param2gaussian_param[k])

    res.to_csv(output_filename, index=False)
    print(output_filename + " was created")
