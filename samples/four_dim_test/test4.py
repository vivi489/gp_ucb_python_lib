# coding: utf-8
import os

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm

from gphypo.egmrf_ucb import EGMRF_UCB
from gphypo.env import BasicEnvironment
from gphypo.util import mkdir_if_not_exist, plot_loss


# from tqdm import tqdm_notebook as tqdm

# SCALE = 1000
# OFFSET = 10

class FourDimEnvironment(BasicEnvironment):
    def __init__(self, gp_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(gp_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x, calc_gt=False, n_exp=1):
        x = np.array(x)

        # # inv_x = 1/(x+0.01)
        # # y = inv_x.sum(axis=0)
        # #
        # # y = x.sum(axis=0)
        #
        # y1 = - (x[0] - 2) ** 2 - (x[1] + 3) ** 2 - (x[2] - 4) ** 2 - (x[3]) ** 6
        # # y2 = - (x[0] + 2) **2 - (x[1] - 3)**2 - (x[2] + 1)**2
        # y2 = 0
        #
        # y = y1 + y2
        # # y = x * SCALE + OFFSET
        #
        # if y.shape == (1,):
        #     return y[0]

        # x = np.array(x)
        #
        # y1 = - (x[0] - 2) ** 2 - (x[1] + 3) ** 2 - (x[2] - 4) ** 2
        # # y2 = - (x[0] + 2) **2 - (x[1] - 4)**2 - (x[2] + 1)**2
        # y2 = 0
        #
        # y = y1 + y2
        # y *= np.sin(x[0])
        # # y = x * SCALE + OFFSET
        #
        # if y.shape == (1,):
        #     return y[0]
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

        if x.ndim == 1:
            pass
        elif x.ndim == 2:
            x = x.T
        else:
            return "OOPS"

        obs = (multivariate_normal.pdf(x[:2], mean=mean1, cov=cov1) + multivariate_normal.pdf(x[:2], mean=mean2,
                                                                                              cov=cov2) \
               + multivariate_normal.pdf(x[:2], mean=mean3, cov=cov3)) * 10 - (x[2]-1)**2 + x[3]

        return obs


########################
ndim = 4  # TODO: 4 dimension does not work...\

BETA = 5  ## sqrt(BETA) controls the ratio between ucb and mean

NORMALIZE_OUTPUT = True
MEAN, STD = 0, 1

reload = False
n_iter = 200
N_EARLY_STOPPING = 100

ALPHA = MEAN  # prior:
GAMMA_Y = 10 / ((STD * ndim) ** 2)  # weight of adjacent
GAMMA = 10 * GAMMA_Y
GAMMA0 = 0.01 * GAMMA
IS_EDGE_NORMALIZED = False


PAIRWISE_SAMPLING = True
UPDATE_ONLY_GAMMA_Y = True

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

env = FourDimEnvironment(gp_param2model_param_dic=gp_param2model_param_dic, result_filename=result_filename,
                         output_dir=output_dir,
                         reload=reload)

agent = EGMRF_UCB(np.meshgrid(*gp_param_list), env, GAMMA=GAMMA, GAMMA0=GAMMA0, GAMMA_Y=GAMMA_Y, ALPHA=ALPHA, BETA=BETA,
                  is_edge_normalized=IS_EDGE_NORMALIZED, gt_available=True, n_early_stopping=1000,
                  normalize_output=True, update_only_gamma_y=UPDATE_ONLY_GAMMA_Y, pairwise_sampling=PAIRWISE_SAMPLING)

for i in tqdm(range(n_iter)):
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
