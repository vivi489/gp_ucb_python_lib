# coding: utf-8
import os

import numpy as np
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm

from env import GaussianEnvironment
from gpucb import GPUCB


########################
output_dir = 'output_gaussian'
paramerter_filename = 'gaussian_param_2dim.csv'
result_filename = 'gaussian_result_2dim.csv'

reload = True
# reload = False
n_iter = 100

beta = 36.
########################

env = GaussianEnvironment(parameter_filename=paramerter_filename, result_filename=result_filename, output_dir=output_dir,
                      reload=reload)

agent = GPUCB(env, beta=beta)

for i in tqdm(range(n_iter)):
    try:
        agent.learn()
        agent.plot(output_dir=output_dir)

    except KeyboardInterrupt:
        print("Learnig process was forced to stop!")
        break
