# coding: utf-8
import os

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

from env import LDA_Environment
from gpucb import GPUCB

########################
output_dir = 'output'
paramerter_filename = 'lda_param_2dim.csv'
result_filename = 'lda_result_2dim.csv'

reload = True
# reload = False
n_iter = 100

beta = 36.
########################

env = LDA_Environment(parameter_filename=paramerter_filename, result_filename=result_filename, output_dir=output_dir,
                      reload=reload)

agent = GPUCB(env, beta=beta)

for i in tqdm(range(n_iter)):
    try:
        agent.learn()
        agent.plot(output_dir=output_dir)

    except KeyboardInterrupt:
        print("Learnig process was forced to stop!")
        break
