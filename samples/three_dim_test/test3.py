# coding: utf-8
import os

import numpy as np
import pandas as pd
from gphypo.env import BasicEnvironment
from gphypo.gmrf_bo import GMRF_BO
from gphypo.gpucb import GPUCB
from gphypo.egmrf_ucb import EGMRF_UCB
from gphypo.util import mkdir_if_not_exist, plot_1dim
from scipy.stats import multivariate_normal
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from tqdm import tqdm


# from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, Matern
# from tqdm import tqdm_notebook as tqdm

# SCALE = 1000
# OFFSET = 10

class ThreeDimGaussianEnvironment(BasicEnvironment):
    def __init__(self, bo_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(bo_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x, calc_gr=False, n_exp=1):
        mean1 = [3, 3, 3]
        cov1 = np.eye(3) * 1

        mean2 = [-4, -4, -4]
        cov2 = np.eye(3) * 0.8

        # mean3 = [-2, 5, 0]
        # cov3 = np.eye(3) * 1

        assert x.ndim in [1, 2]

        obs = multivariate_normal.pdf(x, mean=mean1, cov=cov1) \
              + multivariate_normal.pdf(x, mean=mean2, cov=cov2) \
            # + multivariate_normal.pdf(x, mean=mean3, cov=cov3)
        return obs


class ThreeDimEnvironment(BasicEnvironment):
    def __init__(self, bo_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(bo_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x, calc_gr=False, n_exp=1):
        mean1 = [3, 3]
        cov1 = [[2, 0], [0, 2]]

        mean2 = [-2, -2]
        cov2 = [[1, 0], [0, 1]]

        # mean3 = [3, -3]
        # cov3 = [[0.6, 0], [0, 0.6]]

        mean3 = [3, -3]
        cov3 = [[0.7, 0], [0, 0.7]]

        mean4 = [0, 0]
        cov4 = [[0.1, 0], [0, 0.1]]

        assert x.ndim in [1, 2]

        obs = (multivariate_normal.pdf(x[:2], mean=mean1, cov=cov1) + multivariate_normal.pdf(x[:2], mean=mean2,
                                                                                              cov=cov2) \
               + multivariate_normal.pdf(x[:2], mean=mean3, cov=cov3)) * 10 + x[2]
        return obs


########################
ndim = 3

BETA = 5  ## sqrt(BETA) controls the ratio between ucb and mean

NORMALIZE_OUTPUT = 'zero_mean_unit_var'
# NORMALIZE_OUTPUT = 'zero_one'
# NORMALIZE_OUTPUT = None

reload = False
# reload = True
n_iter = 1000
N_EARLY_STOPPING = 1000

ALPHA = ndim ** 2  # prior:
GAMMA = 10 ** (-2) * 2 * ndim
GAMMA0 = 0.01 * GAMMA
GAMMA_Y = 10 ** (-2)  # weight of adjacen

IS_EDGE_NORMALIZED = True

BURNIN = True
UPDATE_HYPERPARAM_FUNC = 'pairwise_sampling'  # None

INITIAL_K = 10
INITIAL_THETA = 10

kernel = Matern(nu=2.5)
# kernel = C(1) * RBF(2)  # works well, but not so sharp

output_dir = 'output'

parameter_dir = os.path.join('param_dir', 'csv_files')
result_filename = os.path.join(output_dir, 'gaussian_result_3dim.csv')

ACQUISITION_FUNC = 'ucb'  # 'ei'
ACQUISITION_PARAM_DIC = {
    'beta': 5
}

# ACQUISITION_FUNC = 'ei' # 'ei' or 'pi'
# ACQUISITION_PARAM_DIC = {
#     'par': 0.
# }

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

bo_param2model_param_dic = {}

bo_param_list = []
for param_name in param_names:
    param_df = pd.read_csv(os.path.join(parameter_dir, param_name + '.csv'), dtype=str)
    bo_param_list.append(param_df[param_name].values)

    param_df.set_index(param_name, inplace=True)

    bo_param2model_param_dic[param_name] = param_df.to_dict()['bo_' + param_name]

# env = ThreeDimEnvironment(bo_param2model_param_dic=bo_param2model_param_dic, result_filename=result_filename,
#                           output_dir=output_dir,
#                           reload=reload)
env = ThreeDimGaussianEnvironment(bo_param2model_param_dic=bo_param2model_param_dic, result_filename=result_filename,
                                  output_dir=output_dir,
                                  reload=reload)


agent = GMRF_BO(bo_param_list, env, GAMMA=GAMMA, GAMMA0=GAMMA0, GAMMA_Y=GAMMA_Y, ALPHA=ALPHA,
                is_edge_normalized=IS_EDGE_NORMALIZED, gt_available=True, n_early_stopping=N_EARLY_STOPPING,
                burnin=BURNIN,
                normalize_output=NORMALIZE_OUTPUT, update_hyperparam_func=UPDATE_HYPERPARAM_FUNC,
                initial_k=INITIAL_K, initial_theta=INITIAL_THETA, acquisition_func=ACQUISITION_FUNC,
                acquisition_param_dic=ACQUISITION_PARAM_DIC)

for i in tqdm(range(n_iter)):
    try:
        flg = agent.learn()

        agent.plot(output_dir=output_dir)

        agent.save_mu_sigma_csv()
        if flg == False:
            print("Early Stopping!!!")
            print(agent.bestX)
            print(agent.bestT)
            break


            # agent.saveTau('tau.csv')
            # break

    except KeyboardInterrupt:
        print("Learnig process was forced to stop!")
        break

plot_1dim(agent.point_info_manager.T_seq, 'reward.png')
