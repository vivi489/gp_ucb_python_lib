import os
import sys

import numpy as np
import pandas as pd

from gphypo.util import mkdir_if_not_exist

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

output_dir = 'csv_files'
mkdir_if_not_exist(output_dir)

for k, v in gp_param_dic.items():
    output_filename = os.path.join(output_dir, k) + ".csv"
    res = pd.DataFrame({
        k: v
    })
    res['bo_' + k] = res[k].apply(gp_param2lda_param[k])

    res.to_csv(output_filename, index=False)
    print(output_filename + " was created")
