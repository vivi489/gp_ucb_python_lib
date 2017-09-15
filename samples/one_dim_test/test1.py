# coding: utf-8
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process.kernels import Matern

from gphypo.env import BasicEnvironment
from gphypo.gmrf_bo import GMRF_BO
from gphypo.gp_bo import GP_BO
from gphypo.normalization import zero_mean_unit_var_normalization
from gphypo.util import mkdir_if_not_exist, plot_1dim

# from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
# from tqdm import tqdm

SCALE = 0.01
OFFSET = 100000


class SinEnvironment(BasicEnvironment):
    def __init__(self, bo_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(bo_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x, calc_gt=False, n_exp=1):
        x = np.array(x)
        y = np.sin(x) + x * 0.1
        # y = x * np.sin(x) * SCALE + OFFSET
        # y = x * SCALE + OFFSET

        if y.shape == (1,):
            return y[0]

        return y


class OneDimGaussianEnvironment(BasicEnvironment):
    def __init__(self, bo_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(bo_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x, calc_gt=False, n_exp=1):
        assert x.ndim in [1, 2]

        y = norm.pdf(x, loc=-3, scale=0.15) + norm.pdf(x, loc=3, scale=0.7) + norm.pdf(x, loc=0, scale=1.0)

        # return y * 1000 + 100
        return y


########################
ndim = 1

# NORMALIZE_OUTPUT = 'zero_mean_unit_var'  # ALPHA should be 1
# NORMALIZE_OUTPUT = 'zero_one'  # ALPHA should be 1
NORMALIZE_OUTPUT = None
MEAN, STD = 0, 1

reload = False
# reload = True
n_iter = 100
N_EARLY_STOPPING = None

ALPHA = ndim ** 2  # MEAN  # prior:
GAMMA = 6  # 10 ** (-2) * 2 * ndim
GAMMA0 = 0.01 * GAMMA
GAMMA_Y = 3  # 10 ** (-2)  # weight of adjacen

IS_EDGE_NORMALIZED = True

BURNIN = False  # True  # TODO

INITIAL_K = 10
INITIAL_THETA = 10
UPDATE_HYPERPARAM_FUNC = 'pairwise_sampling'  # None

ACQUISITION_FUNC = 'en'  # 'ei'
ACQUISITION_PARAM_DIC = {
    'beta': 5, 
    'eps': 0.3,
    "par": 0.01
}

# ACQUISITION_FUNC = 'ei' # 'ei' or 'pi'
# ACQUISITION_PARAM_DIC = {
#     'par': 0.
# }

output_dir = 'output'
parameter_dir = os.path.join('param_dir', 'csv_files')
result_filename = os.path.join(output_dir, 'gaussian_result_1dim.csv')

########################

kernel = None
#
# kernel = C(1) * RBF(2)  # works well, but not so sharp

# kernel = Matern(nu=2.5)

# ### temporary ###
import shutil

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
##################


print('GAMMA: ', GAMMA)
print('GAMMA_Y: ', GAMMA_Y)
print('GAMMA0:', GAMMA0)

mkdir_if_not_exist(output_dir)

param_names = sorted([x.replace('.csv', '') for x in os.listdir(parameter_dir)])

bo_param2model_param_dic = {} 

bo_param_list = []
for param_name in param_names: # param_name is a param file's name
    param_df = pd.read_csv(os.path.join(parameter_dir, param_name + '.csv'), dtype=str) #makes index column type str instead of float
    
    # param_df has a column of its csv file name, e.g. "x"
    bo_param_list.append(param_df[param_name].values)
    param_df.set_index(param_name, inplace=True)
    
    #dict: param_file name -> column dict (the column with the name "bo_"+param_file name)
    #column dict: index column element -> cell value #index column is type str
    bo_param2model_param_dic[param_name] = param_df.to_dict()['bo_' + param_name]


def main():
    # env = SinEnvironment(bo_param2model_param_dic=bo_param2model_param_dic, result_filename=result_filename,
    #                      output_dir=output_dir,
    #                      reload=reload)
    env = OneDimGaussianEnvironment(bo_param2model_param_dic=bo_param2model_param_dic, result_filename=result_filename,
                                    output_dir=output_dir,
                                    reload=reload)

    agent = GMRF_BO(bo_param_list, env, GAMMA=GAMMA, GAMMA0=GAMMA0, GAMMA_Y=GAMMA_Y, ALPHA=ALPHA,
                    is_edge_normalized=IS_EDGE_NORMALIZED, gt_available=True, n_early_stopping=N_EARLY_STOPPING,
                    burnin=BURNIN,
                    normalize_output=NORMALIZE_OUTPUT, update_hyperparam_func=UPDATE_HYPERPARAM_FUNC,
                    initial_k=INITIAL_K, initial_theta=INITIAL_THETA, acquisition_func=ACQUISITION_FUNC,
                    acquisition_param_dic=ACQUISITION_PARAM_DIC)
    #
    # agent = GP_BO(bo_param_list, env, gt_available=True, my_kernel=kernel, burnin=BURNIN,
    #               normalize_output=NORMALIZE_OUTPUT, acquisition_func=ACQUISITION_FUNC,
    #               acquisition_param_dic=ACQUISITION_PARAM_DIC)

    # for i in tqdm(range(n_iter)):
    for i in range(n_iter):
        try:
            flg = agent.learn()
            agent.plot(output_dir=output_dir)
            agent.save_mu_sigma_csv()

            if flg == False:
                print("Early Stopping!!!")
                print(agent.bestX)
                print(agent.bestT)
                break

        except KeyboardInterrupt:
            print("Learnig process was forced to stop!")
            break

    plot_1dim(agent.point_info_manager.T_seq, 'reward.png')


def calc_real_gamma_y():
    env = SinEnvironment(bo_param2model_param_dic=bo_param2model_param_dic, result_filename=result_filename,
                         output_dir=output_dir,
                         reload=reload)

    x_list = list(bo_param2model_param_dic['x'].values())
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
