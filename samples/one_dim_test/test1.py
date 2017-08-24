# coding: utf-8
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from gphypo.egmrf_ucb import EGMRF_UCB
from gphypo.env import BasicEnvironment
from gphypo.normalization import zero_mean_unit_var_normalization
from gphypo.util import mkdir_if_not_exist, plot_loss

# from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
# from tqdm import tqdm

SCALE = 0.01
OFFSET = 100000


class SinEnvironment(BasicEnvironment):
    def __init__(self, gp_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(gp_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x, calc_gt=False, n_exp=1):
        x = np.array(x)
        y = np.sin(x) + x * 0.1
        # y = x * np.sin(x) * SCALE + OFFSET
        # y = x * SCALE + OFFSET

        if y.shape == (1,):
            return y[0]

        return y


class OneDimGaussianEnvironment(BasicEnvironment):
    def __init__(self, gp_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(gp_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x, calc_gt=False, n_exp=1):

        if x.ndim == 1:
            pass
        elif x.ndim == 2:
            x = x.T
        else:
            return "OOPS"

        y = norm.pdf(x, loc=-3, scale=0.8) + norm.pdf(x, loc=3, scale=0.7) + norm.pdf(x, loc=0, scale=1.5)

        return y * 1000 + 100


########################
ndim = 1

BETA = 5  # 25  ## sqrt(BETA) controls the ratio between ucb and mean

NORMALIZE_OUTPUT = 'zero_mean_unit_var'  # ALPHA should be 1
# NORMALIZE_OUTPUT = 'zero_one' # ALPHA should be 1
# NORMALIZE_OUTPUT = None
MEAN, STD = 0, 1

reload = False
n_iter = 200
N_EARLY_STOPPING = None

ALPHA = ndim ** 2  # MEAN  # prior:
GAMMA = 6  # 10 ** (-2) * 2 * ndim
GAMMA0 = 0.01 * GAMMA
GAMMA_Y = 3  # 10 ** (-2)  # weight of adjacen

IS_EDGE_NORMALIZED = True

BURNIN = 0  # TODO
UPDATE_HYPERPARAM = False

INITIAL_K = 10
INITIAL_THETA = 10

UPDATE_ONLY_GAMMA_Y = True
PAIRWISE_SAMPLING = True
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


def main():
    # env = SinEnvironment(gp_param2model_param_dic=gp_param2model_param_dic, result_filename=result_filename,
    #                      output_dir=output_dir,
    #                      reload=reload)
    env = OneDimGaussianEnvironment(gp_param2model_param_dic=gp_param2model_param_dic, result_filename=result_filename,
                                    output_dir=output_dir,
                                    reload=reload)

    # agent = GPUCB(np.meshgrid(*gp_param_list), env, beta=BETA, gt_available=True, my_kernel=kernel)
    agent = EGMRF_UCB(np.meshgrid(*gp_param_list), env, GAMMA=GAMMA, GAMMA0=GAMMA0, GAMMA_Y=GAMMA_Y, ALPHA=ALPHA,
                      BETA=BETA,
                      is_edge_normalized=IS_EDGE_NORMALIZED, gt_available=True, n_early_stopping=N_EARLY_STOPPING,
                      burnin=BURNIN,
                      normalize_output=NORMALIZE_OUTPUT, update_hyperparam=UPDATE_HYPERPARAM,
                      update_only_gamma_y=UPDATE_ONLY_GAMMA_Y,
                      initial_k=INITIAL_K, initial_theta=INITIAL_THETA, does_pairwise_sampling=PAIRWISE_SAMPLING)

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


def calc_real_gamma_y():
    env = SinEnvironment(gp_param2model_param_dic=gp_param2model_param_dic, result_filename=result_filename,
                         output_dir=output_dir,
                         reload=reload)

    x_list = list(gp_param2model_param_dic['x'].values())
    y_list = env.run_model(-1, x_list)
    y_list, y_mean, y_std = zero_mean_unit_var_normalization(y_list)
    adj_diff = (y_list[1:] - y_list[:-1]) ** 2
    real_var = adj_diff.sum() / (len(adj_diff))
    real_gamma_y = 1 / real_var
    print(real_var)
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
