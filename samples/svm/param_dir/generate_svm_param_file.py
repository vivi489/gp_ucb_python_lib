import os

import numpy as np
import pandas as pd

from gphypo.util import mkdir_if_not_exist

gp_param_dic = {
    "c": np.arange(-2, 5.01, 0.2),
    "gamma": np.arange(-2, 5.01, 0.2),
}

gp_param2lda_param = {
    "c": lambda x: 10 ** x,
    "gamma": lambda x: 10 ** x
}

output_dir = 'csv_files'
mkdir_if_not_exist(output_dir)

for k, v in gp_param_dic.items():
    output_filename = os.path.join(output_dir, k) + ".csv"
    res = pd.DataFrame({
        k: v
    })
    res['transformed_' + k] = res[k].apply(gp_param2lda_param[k])

    res.to_csv(output_filename, index=False)
    print(output_filename + " was created")
