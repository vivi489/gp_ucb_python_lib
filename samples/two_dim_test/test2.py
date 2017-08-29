# coding: utf-8
import os

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from gphypo.egmrf_ucb import EGMRF_UCB, create_normalized_X_grid
# from gphypo.egmrf_ucb import EGMRF_UCB
from gphypo.env import BasicEnvironment
from gphypo.normalization import zero_mean_unit_var_normalization
from gphypo.util import mkdir_if_not_exist, plot_loss


# from tqdm import tqdm_notebook as tqdm

class GaussianEnvironment(BasicEnvironment):
    def __init__(self, gp_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(gp_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x, calc_gt=False, n_exp=1):

        mean1 = [3, 3]
        cov1 = [[2, 0], [0, 2]]

        mean2 = [-2, -2]
        # cov2 = [[1, 0], [0, 1]]
        cov2 = [[1.5, 0], [0, 1.5]]

        # mean3 = [3, -3]
        # cov3 = [[0.6, 0], [0, 0.6]]

        mean3 = [3, -3]
        cov3 = [[0.5, 0], [0, 0.5]]

        mean4 = [0, 0]
        cov4 = [[0.1, 0], [0, 0.1]]

        assert x.ndim in [1, 2]

        y = multivariate_normal.pdf(x, mean=mean1, cov=cov1) + multivariate_normal.pdf(x, mean=mean2, cov=cov2) \
            + multivariate_normal.pdf(x, mean=mean3,
                                      cov=cov3)  # - multivariate_normal.pdf(x, mean=mean4, cov=cov4) * 0.1

        return y * 1000 + 100


########################
ndim = 2

BETA = 5  ## sqrt(BETA) controls the ratio between ucb and mean

# NORMALIZE_OUTPUT = 'zero_mean_unit_var'
NORMALIZE_OUTPUT = 'zero_one'
# NORMALIZE_OUTPUT = None
MEAN, STD = 0, 1

# reload = True
reload = False
n_iter = 1000
N_EARLY_STOPPING = 1000

# ALPHA = MEAN  # prior:
ALPHA = ndim ** 2

GAMMA = 10 ** (-2) * 2 * ndim
GAMMA0 = 0.01 * GAMMA
GAMMA_Y = 10 ** (-2)  # weight of adjacen

IS_EDGE_NORMALIZED = True

# kernel = Matern(2.5)

BURNIN = 0  # TODO
INITIAL_K = 10
INITIAL_THETA = 10

UPDATE_HYPERPARAM_FUNC = 'pairwise_sampling'  # None

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
    param_df = pd.read_csv(os.path.join(parameter_dir, param_name + '.csv'), dtype=str)
    gp_param_list.append(param_df[param_name].values)

    param_df.set_index(param_name, inplace=True)

    gp_param2model_param_dic[param_name] = param_df.to_dict()['gp_' + param_name]


def main():
    env = GaussianEnvironment(gp_param2model_param_dic=gp_param2model_param_dic, result_filename=result_filename,
                              output_dir=output_dir,
                              reload=reload)

    # agent = GPUCB(np.meshgrid(*gp_param_list), env, beta=BETA, gt_available=True, my_kernel=kernel)
    agent = EGMRF_UCB(np.meshgrid(*gp_param_list), env, GAMMA=GAMMA, GAMMA0=GAMMA0, GAMMA_Y=GAMMA_Y, ALPHA=ALPHA,
                      BETA=BETA,
                      is_edge_normalized=IS_EDGE_NORMALIZED, gt_available=True, n_early_stopping=N_EARLY_STOPPING,
                      burnin=BURNIN,
                      normalize_output=NORMALIZE_OUTPUT, update_hyperparam_func=UPDATE_HYPERPARAM_FUNC,
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

    plot_loss(agent.point_info_manager.T_seq, 'reward.png')


def calc_real_gamma_y():
    env = GaussianEnvironment(gp_param2model_param_dic=gp_param2model_param_dic, result_filename=result_filename,
                              output_dir=output_dir,
                              reload=reload)

    meshgrid = np.array(np.meshgrid(*gp_param_list))
    X_grid = meshgrid.reshape(meshgrid.shape[0], -1).T
    normalized_X_grid = create_normalized_X_grid(meshgrid)

    x_list = list(gp_param2model_param_dic['x'].values())
    y_list = env.run_model(-1, x_list)
    y_list, y_mean, y_std = zero_mean_unit_var_normalization(y_list)
    adj_diff = (y_list[1:] - y_list[:-1]) ** 2
    real_var = adj_diff.sum() / (len(adj_diff))
    real_gamma_y = 1 / real_var
    print('real gamma_y is %s' % real_gamma_y)

    n_skip = 10
    x_even_list = x_list[::n_skip]
    y_even_list = env.run_model(-1, x_even_list)
    adj_even_diff = (y_even_list[1:] - y_even_list[:-1]) ** 2
    real_even_var = adj_even_diff.sum() / (len(adj_even_diff))
    real_even_gamma_y = 1 / real_even_var
    pred_gamma_y = n_skip ** 2 * real_even_gamma_y
    print('pred gamma_y is %s' % pred_gamma_y)

    print(len(adj_diff), len(adj_even_diff))


if __name__ == '__main__':
    # calc_real_gamma_y()
    main()
