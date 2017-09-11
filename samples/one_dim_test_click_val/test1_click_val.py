# coding: utf-8
import os
import random

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm

from gphypo.egmrf_ucb import EGMRF_UCB
from gphypo.env import BasicEnvironment
from gphypo.util import mkdir_if_not_exist, plot_loss


# from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
# from tqdm import tqdm

def sigmoid(x):
    sigmoid_range = 34.538776394910684

    # if x <= -sigmoid_range:
    #     return 1e-15
    # if x >= sigmoid_range:
    #     return 1.0 - 1e-15

    return np.nan_to_num(1.0 / (1.0 + np.exp(-x)))


def flip(p):
    return (random.random() < p)


class ClickOneDimGaussianEnvironment(BasicEnvironment):
    def __init__(self, gp_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(gp_param2model_param_dic, result_filename, output_dir, reload)


    def run_model(self, model_number, x, calc_gt=False, n_exp=1):
        y = norm.pdf(x, loc=-3, scale=0.8) + norm.pdf(x, loc=3, scale=0.7) + norm.pdf(x, loc=0, scale=1.5)

        prob = sigmoid(y)

        if calc_gt:
            return logit(prob)

        if n_exp > 1:
            return np.random.binomial(n=n_exp, p=prob)

        clicked = int(flip(prob))
        return clicked


class ClickSinEnvironment(BasicEnvironment):
    def __init__(self, gp_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(gp_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x, calc_gt=False, n_exp=1):
        prob = sigmoid(np.sin(x))

        if calc_gt:
            return logit(prob)

        if n_exp > 1:
            return np.random.binomial(n=n_exp, p=prob)

        clicked = int(flip(prob))
        return clicked


########################
ndim = 1

BETA = 5  ## sqrt(BETA) controls the ratio between ucb and mean

NORMALIZE_OUTPUT = 'zero_mean_unit_var'
# NORMALIZE_OUTPUT = 'zero_one'
# NORMALIZE_OUTPUT = None
MEAN, STD = 0, 1

reload = False
# reload = True
n_iter = 200
N_EARLY_STOPPING = 100

ALPHA = 1  # prior:

GAMMA_Y_ = 10 / ((STD * ndim) ** 2)  # weight of adjacent
GAMMA = (2 * ndim) * GAMMA_Y_

GAMMA_Y = 0.01 / ((STD * ndim) ** 2)  # weight of adjacent

GAMMA0 = 0.01 * GAMMA
IS_EDGE_NORMALIZED = True

BURNIN = True
UPDATE_HYPERPARAM_FUNC = 'pairwise_sampling'

INITIAL_K = 1
INITIAL_THETA = 1

# kernel = Matern(2.5)

output_dir = 'output'
parameter_dir = os.path.join('param_dir', 'csv_files')
result_filename = os.path.join(output_dir, 'gaussian_result_1dim_click.csv')

N_EXP = 1000
ACQUISITION_FUNC = 'ei'
########################

### temporary ###
import shutil

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
# ##################


print('GAMMA: ', GAMMA)
print('GAMMA_Y: ', GAMMA_Y)
print('GAMMA0:', GAMMA0)

mkdir_if_not_exist(output_dir)

param_names = sorted([x.replace('.csv', '') for x in os.listdir(parameter_dir)])

# print (param_names)

gp_param2model_param_dic = {}

gp_param_list = []
for param_name in param_names:
    param_df = pd.read_csv(os.path.join(parameter_dir, param_name + '.csv'), dtype=str)
    gp_param_list.append(param_df[param_name].values)

    param_df.set_index(param_name, inplace=True)

    gp_param2model_param_dic[param_name] = param_df.to_dict()['gp_' + param_name]

env = ClickSinEnvironment(gp_param2model_param_dic=gp_param2model_param_dic, result_filename=result_filename,
                          output_dir=output_dir,
                          reload=reload)
# env = ClickOneDimGaussianEnvironment(gp_param2model_param_dic=gp_param2model_param_dic, result_filename=result_filename,
#                           output_dir=output_dir,
#                           reload=reload)

# agent = GPUCB(np.meshgrid(*gp_param_list), env, beta=BETA, gt_available=True, my_kernel=kernel)
agent = EGMRF_UCB(gp_param_list, env, GAMMA=GAMMA, GAMMA0=GAMMA0, GAMMA_Y=GAMMA_Y, ALPHA=ALPHA, BETA=BETA,
                  is_edge_normalized=IS_EDGE_NORMALIZED, gt_available=True, n_early_stopping=N_EARLY_STOPPING,
                  burnin=BURNIN,
                  normalize_output=NORMALIZE_OUTPUT, update_hyperparam_func=UPDATE_HYPERPARAM_FUNC,
                  initial_k=INITIAL_K, initial_theta=INITIAL_THETA, n_exp=N_EXP, acquisition_func=ACQUISITION_FUNC)

# for i in tqdm(range(n_iter)):
for i in range(n_iter):
    try:
        # flg = agent.learn()

        # agent.plot(output_dir=output_dir)
        flg = agent.learn_from_clicks()

        agent.save_mu_sigma_csv()


        if flg == False:
            print("Early Stopping!!!")
            print(agent.bestX)
            print(agent.bestT)
            break

    except KeyboardInterrupt:
        print("Learnig process was forced to stop!")
        # print(agent.X)
        # print(agent.Treal)
        break

plot_loss(agent.point_info_manager.T_seq, 'reward.png')
