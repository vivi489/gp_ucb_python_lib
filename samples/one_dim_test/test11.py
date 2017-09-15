# coding: utf-8
import os, subprocess, shutil, time

import numpy as np
import pandas as pd
from scipy.stats import norm

from gphypo.env import BasicEnvironment
from gphypo.gmrf_bo import GMRF_BO
from gphypo.gp_bo import GP_BO
from gphypo.normalization import zero_mean_unit_var_normalization
from gphypo.util import mkdir_if_not_exist, plot_1dim

# from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
# from tqdm import tqdm

# SCALE = 0.01
# OFFSET = 100000


class SinEnvironment(BasicEnvironment):
    def __init__(self, bo_param2model_param_dic, result_filename, output_dir, reload, noiseVar=0.025):
        super().__init__(bo_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x, calc_gt=False, n_exp=1):
        x = np.array(x)
        y = np.sin(x) + x * 0.1
        if not calc_gt:
            y += np.random.normal(loc=0, scale=np.sqrt(self.noiseVar))
        if y.shape == (1,):
            return y[0]
        return y


class OneDimGaussianEnvironment(BasicEnvironment):
    def __init__(self, bo_param2model_param_dic, result_filename, output_dir, reload, noiseVar=0.025):
        super().__init__(bo_param2model_param_dic, result_filename, output_dir, reload)
        self.noiseVar = noiseVar

    def run_model(self, model_number, x, calc_gt=False, n_exp=1):
        assert x.ndim in [1, 2]
        y = norm.pdf(x, loc=-3, scale=0.15) + norm.pdf(x, loc=3, scale=0.7) + norm.pdf(x, loc=0, scale=1.5)
        if not calc_gt:
            y += np.random.normal(loc=0, scale=np.sqrt(self.noiseVar))
        return y


########################
ndim = 1
NORMALIZE_OUTPUT = None
MEAN, STD = 0, 1

RELOAD = False
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

#ACQUISITION_FUNC = 'en'  # 'ei'
ACQUISITION_PARAM_DIC = {
    'beta': 5, 
    'eps': 0.10,
    "par": 0.01,
    "tsFactor": 3.0
}

OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
PARAMETER_DIR = os.path.join('param_dir', 'csv_files')

########################

kernel = None
#
# kernel = C(1) * RBF(2)  # works well, but not so sharp

# kernel = Matern(nu=2.5)


def test(ACQUISITION_FUNC):
    
    RESULT_FILENAME = os.path.join(OUTPUT_DIR, 'gaussian_result_1dim.csv')
    
    # ### temporary ###
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    ##################
    
    np.random.seed(int(time.time()))
    print('GAMMA: ', GAMMA)
    print('GAMMA_Y: ', GAMMA_Y)
    print('GAMMA0:', GAMMA0)
    
    mkdir_if_not_exist(OUTPUT_DIR)
    param_names = sorted([x.replace('.csv', '') for x in os.listdir(PARAMETER_DIR)])

    bo_param2model_param_dic = {} 
    bo_param_list = []
    for param_name in param_names: # param_name is a param file's name
        param_df = pd.read_csv(os.path.join(PARAMETER_DIR, param_name + '.csv'), dtype=str) #makes index column type str instead of float
        
        # always read the column of the same name as the file name -- param_name
        bo_param_list.append(param_df[param_name].values)
        # param_df has a column of its csv file name, e.g. "x"
        # and this column is set as the index column
        param_df.set_index(param_name, inplace=True)
        # dict: param_file name -> column dict (the column with the name "bo_"+param_file name)
        # column dict: index column element -> cell value #index column is type str
        bo_param2model_param_dic[param_name] = param_df.to_dict()['bo_' + param_name]
        
        # bo_param_list is a list of every "bo_" column in all the param files of param_names
        # print("bo_param_list", bo_param_list)
    
#    env = SinEnvironment(bo_param2model_param_dic=bo_param2model_param_dic, 
#                         result_filename=RESULT_FILENAME,
#                         output_dir=OUTPUT_DIR,
#                         reload=RELOAD)
    
    env = OneDimGaussianEnvironment(bo_param2model_param_dic=bo_param2model_param_dic, 
                                    result_filename=RESULT_FILENAME,
                                    output_dir=OUTPUT_DIR,
                                    reload=RELOAD)

#    agent = GMRF_BO(bo_param_list, env, GAMMA=GAMMA, GAMMA0=GAMMA0, GAMMA_Y=GAMMA_Y, ALPHA=ALPHA,
#                     is_edge_normalized=IS_EDGE_NORMALIZED, 
#                     gt_available=True, 
#                     n_early_stopping=N_EARLY_STOPPING,
#                     burnin=BURNIN,
#                     normalize_output=NORMALIZE_OUTPUT, 
#                     update_hyperparam_func=UPDATE_HYPERPARAM_FUNC,
#                     initial_k=INITIAL_K, 
#                     initial_theta=INITIAL_THETA, 
#                     acquisition_func=ACQUISITION_FUNC,
#                     acquisition_param_dic=ACQUISITION_PARAM_DIC)

    agent = GP_BO(bo_param_list, env,
                   gt_available=True, 
                   my_kernel=kernel, 
                   burnin=BURNIN,
                   normalize_output=NORMALIZE_OUTPUT, 
                   acquisition_func=ACQUISITION_FUNC,
                   acquisition_param_dic=ACQUISITION_PARAM_DIC)

    nIter = 100
    for i in range(nIter):
        flg = agent.learn(drop=True if i<nIter-1 else False)
        agent.plot(output_dir=OUTPUT_DIR)
        agent.save_mu_sigma_csv()
        if flg == False:
            print("Early Stopping!!!")
            print("bestX =", agent.bestX)
            print("bestT =", agent.bestT)   
            break
    plot_1dim(agent.point_info_manager.T_seq, 'reward.png')
    subprocess.call(["./convert_pngs2gif.sh demo_%s_iter_%d_eps_%f.gif"%(ACQUISITION_FUNC, nIter, ACQUISITION_PARAM_DIC["eps"])], shell=True)
    os.system("mv ./output/*.gif ./")

if __name__ == '__main__':
    for ac in ["ucb", "pi", "ei", "en", "ts"]:
        test(ac)

