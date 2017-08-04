# coding: utf-8
import os

import numpy as np
import pandas as pd

from gphypo.egmrf_ucb import EGMRF_UCB
from gphypo.env import BasicEnvironment
from gphypo.util import mkdir_if_not_exist, plot_loss

# from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
# from tqdm import tqdm

SCALE = 0.01
OFFSET = 100000


class SinEnvironment(BasicEnvironment):
    def __init__(self, gp_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(gp_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x):
        x = np.array(x)
        y = x * np.sin(x) * SCALE + OFFSET
        # y = x * SCALE + OFFSET

        if y.shape == (1,):
            return y[0]

        return y


########################
ndim = 1

BETA = 5  ## sqrt(BETA) controls the ratio between ucb and mean

NORMALIZE_OUTPUT = True
MEAN, STD = 0, 1

reload = False
n_iter = 200
N_EARLY_STOPPING = 100

ALPHA = MEAN  # prior:

GAMMA_Y = 5 / ((STD * ndim) ** 2)  # weight of adjacent

GAMMA = (2 * ndim) * GAMMA_Y
GAMMA0 = 0.01 * GAMMA
IS_EDGE_NORMALIZED = True

BURNIN = 0
UPDATE_HYPERPARAM = False
UPDATE_ONLY_GAMMA_Y = True
INITIAL_K = 10
INITIAL_THETA = 10

# kernel = Matern(2.5)

output_dir = 'output'
parameter_dir = os.path.join('param_dir', 'csv_files')
result_filename = os.path.join(output_dir, 'gaussian_result_2dim.csv')

########################

### temporary ###
import shutil

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
##################


print('GAMMA: ', GAMMA)
print('GAMMA_Y: ', GAMMA_Y)
print('GAMMA0:', GAMMA0)

mkdir_if_not_exist(output_dir)

param_names = sorted([x.replace('.csv', '') for x in os.listdir(parameter_dir)])

gp_param2model_param_dic = {}

gp_param_list = []
for param_name in param_names:
    param_df = pd.read_csv(os.path.join(parameter_dir, param_name + '.csv'))
    gp_param_list.append(param_df[param_name].values)

    param_df.set_index(param_name, inplace=True)

    gp_param2model_param_dic[param_name] = param_df.to_dict()['gp_' + param_name]

env = SinEnvironment(gp_param2model_param_dic=gp_param2model_param_dic, result_filename=result_filename,
                     output_dir=output_dir,
                     reload=reload)

# agent = GPUCB(np.meshgrid(*gp_param_list), env, beta=BETA, gt_available=True, my_kernel=kernel)
agent = EGMRF_UCB(np.meshgrid(*gp_param_list), env, GAMMA=GAMMA, GAMMA0=GAMMA0, GAMMA_Y=GAMMA_Y, ALPHA=ALPHA, BETA=BETA,
                  is_edge_normalized=IS_EDGE_NORMALIZED, gt_available=True, n_early_stopping=N_EARLY_STOPPING,
                  burnin=BURNIN,
                  normalize_output=NORMALIZE_OUTPUT, update_hyperparam=UPDATE_HYPERPARAM,
                  update_only_gamma_y=UPDATE_ONLY_GAMMA_Y,
                  initial_k=INITIAL_K, initial_theta=INITIAL_THETA)

# for i in tqdm(range(n_iter)):
for i in range(n_iter):
    try:
        flg = agent.learn()

        agent.plot(output_dir=output_dir)

        if flg == False:
            print("Early Stopping!!!")
            print(agent.bestX)
            print(agent.bestT)
            break

    except KeyboardInterrupt:
        print("Learnig process was forced to stop!")
        break

plot_loss(agent.Treal, 'reward.png')
