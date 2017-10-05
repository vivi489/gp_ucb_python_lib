# coding: utf-8
import os, shutil
import random

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm

from gphypo.env import BasicEnvironment
from gphypo.gmrf_bo import GMRF_BO
from gphypo.util import mkdir_if_not_exist, plot_1dim


# from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
# from tqdm import tqdm


def flip(p):
    return (random.random() < p)


class ClickOneDimGaussianEnvironment(BasicEnvironment):
    def __init__(self, bo_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(bo_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x, calc_gt=False, n_exp=1):
        #print("run_model: n_exp=", n_exp)
        prob = norm.pdf(x, loc=-2, scale=0.3) + norm.pdf(x, loc=3, scale=0.7) + norm.pdf(x, loc=0, scale=1.5)
        prob /= 3.0
        if calc_gt:
            #print("truth: ", logit(prob))
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

reload = False
# reload = True
N_EARLY_STOPPING = 1000

ALPHA = ndim ** 2  # prior:
GAMMA = 10 ** (-2) * 2 * ndim
GAMMA0 = 0.01 * GAMMA
GAMMA_Y = 10 ** (-2) # weight of adjacency

GAMMA0 = 0.01 * GAMMA
IS_EDGE_NORMALIZED = True

BURNIN = False
UPDATE_HYPERPARAM_FUNC = None #'pairwise_sampling'

INITIAL_K = 10
INITIAL_THETA = 10

ACQUISITION_PARAM_DIC = {
    'beta': 5, #for "ucb"
    'eps': 0.20, #for "greedy"
    "par": 0.01, 
}

parameter_dir = os.path.join('param_dir', 'csv_files')

#ACQUISITION_FUNC = 'ucb'

def singleTest(ACQUISITION_FUNC, trialCount):
    print("%s: trial %d"%(ACQUISITION_FUNC, trialCount))
    OUTPUT_DIR = os.path.join(os.getcwd(), 'output_%s_clicks'%ACQUISITION_FUNC)
    
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    RESULT_FILENAME = os.path.join(OUTPUT_DIR, "gaussian_result_1dim_clicks_%s_trialCount_%d.csv"%(ACQUISITION_FUNC, trialCount))
    # ##################
    
    mu_sigma_csv_path = './mu2ratio_%s/mu_sigma.csv'%ACQUISITION_FUNC
    ratio_csv_out_path = './mu2ratio_%s/ratios.csv'%ACQUISITION_FUNC
    #point_path = './mu2ratio_%s/point_info.csv'%ACQUISITION_FUNC
    N_TOTAL_EXP = 10000
    
    print('GAMMA: ', GAMMA)
    print('GAMMA_Y: ', GAMMA_Y)
    print('GAMMA0:', GAMMA0)
    
    mkdir_if_not_exist(OUTPUT_DIR)
    
    param_names = sorted([x.replace('.csv', '') for x in os.listdir(parameter_dir)])
    
    # print (param_names)
    
    bo_param2model_param_dic = {}
    bo_param_list = []
    for param_name in param_names:
        param_df = pd.read_csv(os.path.join(parameter_dir, param_name + '.csv'), dtype=str)
        bo_param_list.append(param_df[param_name].values)#appending a column
        param_df.set_index(param_name, inplace=True)
        bo_param2model_param_dic[param_name] = param_df.to_dict()['bo_' + param_name]
    
    env = ClickOneDimGaussianEnvironment(bo_param2model_param_dic=bo_param2model_param_dic, result_filename=RESULT_FILENAME,
                                         output_dir=OUTPUT_DIR,
                                         reload=reload)
    
    agent = GMRF_BO(bo_param_list, env, GAMMA=GAMMA, GAMMA0=GAMMA0, GAMMA_Y=GAMMA_Y, ALPHA=ALPHA,
                    is_edge_normalized=IS_EDGE_NORMALIZED, gt_available=True, n_early_stopping=N_EARLY_STOPPING,
                    burnin=BURNIN,
                    normalize_output=NORMALIZE_OUTPUT, update_hyperparam_func=UPDATE_HYPERPARAM_FUNC,
                    initial_k=INITIAL_K, initial_theta=INITIAL_THETA, acquisition_func=ACQUISITION_FUNC,
                    acquisition_param_dic=ACQUISITION_PARAM_DIC, n_ctr=N_TOTAL_EXP)
    
    
    agent.save_mu_sigma_csv(outfn=mu_sigma_csv_path)
    # agent.plot_click_distribution(output_dir)
    # agent.learn_from_clicks()

    nIter = 100
    for i in range(nIter):
        try:
            flg = agent.learn_from_clicks(mu2ratio_dir='./mu2ratio_%s'%ACQUISITION_FUNC, 
                                          mu_sigma_csv_path=mu_sigma_csv_path, 
                                          ratio_csv_out_path=ratio_csv_out_path)
            # agent.sample_randomly()
            #agent.plot_click_distribution(output_dir)
            #break
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

    #plot_1dim([agent.total_clicked_ratio_list, agent.randomly_total_clicked_ratio_list], 'total_clicked_ratio_list.png')
    #print(agent.total_clicked_ratio_list)
    #print(agent.randomly_total_clicked_ratio_list)

singleTest("ts", 0)
