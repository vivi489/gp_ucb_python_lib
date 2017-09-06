# coding: utf-8
import os
import random

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import multivariate_normal

from gphypo.env import BasicEnvironment
from gphypo.gmrf_bo import GMRF_BO
from gphypo.util import mkdir_if_not_exist, plot_1dim


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


class ClickTwoDimGaussianEnvironment(BasicEnvironment):
    def __init__(self, bo_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(bo_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x, calc_gt=False, n_exp=1):
        mean1 = [3, 3]
        cov1 = [[2, 0], [0, 2]]

        mean2 = [-2, -2]
        cov2 = [[1, 0], [0, 1]]

        mean3 = [3, -3]
        cov3 = [[0.7, 0], [0, 0.7]]

        assert x.ndim in [1, 2]

        prob = multivariate_normal.pdf(x, mean=mean1, cov=cov1) + multivariate_normal.pdf(x, mean=mean2, cov=cov2) \
               + multivariate_normal.pdf(x, mean=mean3, cov=cov3)
        prob *= 3

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

ACQUISITION_FUNC = 'ucb'
ACQUISITION_PARAM_DIC = {
    'beta': 5
}
########################

### temporary ###
import shutil

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
# ##################
mu_sigma_fn = './mu2ratio/mu_sigma.csv'
ratio_fn = './mu2ratio/ratios.csv'
point_fn = './mu2ratio/point_info.csv'
n_total_exp = 100000

print('GAMMA: ', GAMMA)
print('GAMMA_Y: ', GAMMA_Y)
print('GAMMA0:', GAMMA0)

mkdir_if_not_exist(output_dir)

param_names = sorted([x.replace('.csv', '') for x in os.listdir(parameter_dir)])

# print (param_names)

bo_param2model_param_dic = {}

bo_param_list = []
for param_name in param_names:
    param_df = pd.read_csv(os.path.join(parameter_dir, param_name + '.csv'), dtype=str)
    bo_param_list.append(param_df[param_name].values)

    param_df.set_index(param_name, inplace=True)

    bo_param2model_param_dic[param_name] = param_df.to_dict()['bo_' + param_name]

env = ClickTwoDimGaussianEnvironment(bo_param2model_param_dic=bo_param2model_param_dic, result_filename=result_filename,
                                     output_dir=output_dir,
                                     reload=reload)

agent = GMRF_BO(bo_param_list, env, GAMMA=GAMMA, GAMMA0=GAMMA0, GAMMA_Y=GAMMA_Y, ALPHA=ALPHA,
                is_edge_normalized=IS_EDGE_NORMALIZED, gt_available=True, n_early_stopping=N_EARLY_STOPPING,
                burnin=BURNIN,
                normalize_output=NORMALIZE_OUTPUT, update_hyperparam_func=UPDATE_HYPERPARAM_FUNC,
                initial_k=INITIAL_K, initial_theta=INITIAL_THETA, acquisition_func=ACQUISITION_FUNC,
                acquisition_param_dic=ACQUISITION_PARAM_DIC, n_ctr=n_total_exp)

# for i in tqdm(range(n_iter)):

agent.save_mu_sigma_csv(outfn=mu_sigma_fn, point_info_fn=point_fn)

# agent.learn_from_clicks()

for i in range(n_iter):
    try:
        flg = agent.learn_from_clicks()
        # agent.sample_randomly()
        agent.plot_click_distribution(output_dir)

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

plot_1dim([agent.total_clicked_ratio_list, agent.randomly_total_clicked_ratio_list], 'total_clicked_ratio_list.png')
print(agent.total_clicked_ratio_list)
print(agent.randomly_total_clicked_ratio_list)
