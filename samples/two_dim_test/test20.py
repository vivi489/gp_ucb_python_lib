# coding: utf-8
import os

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
#from gphypo.egmrf_ucb import create_normalized_X_grid
# from gphypo.egmrf_ucb import EGMRF_UCB
from gphypo.env import BasicEnvironment
from gphypo.gmrf_bo import GMRF_BO
#from gphypo.gpucb import GPUCB
from gphypo.normalization import zero_mean_unit_var_normalization
from gphypo.util import mkdir_if_not_exist


# from tqdm import tqdm_notebook as tqdm
from sklearn.gaussian_process.kernels import Matern


class GaussianEnvironment(BasicEnvironment):
    def __init__(self, bo_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(bo_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x, calc_gt=False, n_exp=1):

        mean1 = [3, 3]
        cov1 = [[2, 0], [0, 2]]
        # mean1 = [0, 0]
        # cov1 = [[0.5, 0], [0, 0.5]]

        mean2 = [-2, -2]
        cov2 = [[1, 0], [0, 1]]

        # mean2 = [-3, -3]
        # cov2 = [[1, 0], [0, 1]]
        # cov2 = [[1.5, 0], [0, 1.5]]
        # cov2 = [[3, 0], [0, 3]]

        # mean3 = [3, -3]
        # cov3 = [[0.6, 0], [0, 0.6]]

        mean3 = [3, -3]
        cov3 = [[0.7, 0], [0, 0.7]]

        #mean4 = [0, 0]
        #cov4 = [[0.1, 0], [0, 0.1]]

        assert x.ndim in [1, 2]

        y = multivariate_normal.pdf(x, mean=mean1, cov=cov1) + multivariate_normal.pdf(x, mean=mean2, cov=cov2) \
            + multivariate_normal.pdf(x, mean=mean3,
                                      cov=cov3)  # - multivariate_normal.pdf(x, mean=mean4, cov=cov4) * 0.1

        # return y * 1000 + 100
        return y


########################
ndim = 2

BETA = 5  ## sqrt(BETA) controls the ratio between ucb and mean

NORMALIZE_OUTPUT = 'zero_mean_unit_var'
# NORMALIZE_OUTPUT = 'zero_one'
# NORMALIZE_OUTPUT = None
MEAN, STD = 0, 1

# reload = True
reload = False
N_EARLY_STOPPING = 1000

# ALPHA = MEAN  # prior:
ALPHA = 0.001 # ndim ** 1

GAMMA = 10 ** (-2) * 2 * ndim
GAMMA0 = 0.01 * GAMMA
GAMMA_Y = 10 ** (-2)  # weight of adjacen

IS_EDGE_NORMALIZED = True

# kernel = Matern(nu=2.5)
kernel = C(1) * RBF(2)
# kernel = None

BURNIN = False  # TODO
INITIAL_K = 10
INITIAL_THETA = 10

UPDATE_HYPERPARAM_FUNC = 'pairwise_sampling'  # None

output_dir = 'output'# _gmrf_min0max1_easy'
parameter_dir = os.path.join('param_dir', 'csv_files')
result_filename = os.path.join(output_dir, 'gaussian_result_2dim.csv')

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

print (bo_param_list)

def main():
    env = GaussianEnvironment(bo_param2model_param_dic=bo_param2model_param_dic, result_filename=result_filename,
                              output_dir=output_dir,
                              reload=reload)


    agent = GMRF_BO(bo_param_list, env, GAMMA=GAMMA, GAMMA0=GAMMA0, GAMMA_Y=GAMMA_Y, ALPHA=ALPHA,
                    is_edge_normalized=IS_EDGE_NORMALIZED, gt_available=True, n_early_stopping=N_EARLY_STOPPING,
                    burnin=BURNIN,
                    normalize_output=NORMALIZE_OUTPUT, update_hyperparam_func=UPDATE_HYPERPARAM_FUNC,
                    initial_k=INITIAL_K, initial_theta=INITIAL_THETA, acquisition_func=ACQUISITION_FUNC,
                    acquisition_param_dic=ACQUISITION_PARAM_DIC)

    # for i in tqdm(range(n_iter)):
    for i in range(600):
        try:
            flg = agent.learn()
            agent.plot(output_dir=output_dir)
            # agent.save_mu_sigma_csv()

            if flg == False:
                print("Early Stopping!!!")
                print(agent.bestX)
                print(agent.bestT)
                break

        except KeyboardInterrupt:
            print("Learnig process was forced to stop!")
            break

    #plot_loss(agent.point_info_manager.T_seq, 'reward.png')



if __name__ == '__main__':
    main()
