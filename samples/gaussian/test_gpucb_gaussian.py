# coding: utf-8
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from env import GaussianEnvironment
from gpucb import GPUCB

# from tqdm import tqdm_notebook as tqdm


########################
output_dir = 'output_gaussian'

parameter_dir = os.path.join('param_dir', 'csv_files')
result_filename = 'gaussian_result_2dim.csv'

# reload = True
reload = False
n_iter = 100

beta = 36.
########################


param_names = sorted([x.replace('.csv', '') for x in os.listdir(parameter_dir)])

gp_param2model_param_dic = {}

gp_param_list = []
for param_name in param_names:
    param_df = pd.read_csv(os.path.join(parameter_dir, param_name + '.csv'))
    gp_param_list.append(param_df[param_name].values)

    param_df.set_index(param_name, inplace=True)

    gp_param2model_param_dic[param_name] = param_df.to_dict()['gp_' + param_name]

env = GaussianEnvironment(gp_param2model_param_dic=gp_param2model_param_dic, result_filename=result_filename,
                          output_dir=output_dir,
                          reload=reload)

agent = GPUCB(np.meshgrid(*gp_param_list), env, beta=beta)

for i in tqdm(range(n_iter)):
    try:
        agent.learn()
        agent.plot(output_dir=output_dir)

    except KeyboardInterrupt:
        print("Learnig process was forced to stop!")
        break
