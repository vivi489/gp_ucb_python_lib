# coding: utf-8
import os, shutil, random, sys

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import multivariate_normal

from gphypo.env import BasicEnvironment
from gphypo.gmrf_bo import GMRF_BO
from gphypo.util import mkdir_if_not_exist, plot_1dim

def flip(p):
    return (random.random() < p)


class ClickTwoDimGaussianEnvironment(BasicEnvironment):
    def __init__(self, bo_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(bo_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x, calc_gt=False, n_exp=1):
        mean1 = [-3, 3]
        cov1 = [[0.5, 0], [0, 0.5]]
        
        mean2 = [1, 1]
        cov2 = [[2.5, 0], [0, 2.5]]
        
        mean3 = [3, -3]
        cov3 = [[0.25, 0], [0, 0.25]]

        assert x.ndim in [1, 2]

        prob = multivariate_normal.pdf(x, mean=mean1, cov=cov1) + multivariate_normal.pdf(x, mean=mean2, cov=cov2) \
               + multivariate_normal.pdf(x, mean=mean3, cov=cov3)
        prob /= 3.0

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
N_EARLY_STOPPING = 1000


ALPHA = ndim ** 2  # prior:
GAMMA = 10 ** (-2) * 2 * ndim
GAMMA0 = 0.01 * GAMMA
GAMMA_Y = 10 ** (-2) # weight of adjacency

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

def singleTest(ACQUISITION_FUNC, trialCount):
    print("%s: trial %d"%(ACQUISITION_FUNC, trialCount))
    OUTPUT_DIR = os.path.join(os.getcwd(), 'output_%s_clicks'%ACQUISITION_FUNC)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    RESULT_FILENAME = os.path.join(OUTPUT_DIR, "gaussian_result_2dim_clicks_%s_trialCount_%d.csv"%(ACQUISITION_FUNC, trialCount))

    mu_sigma_csv_path = './mu2ratio_%s/mu_sigma.csv'%ACQUISITION_FUNC
    ratio_csv_out_path = './mu2ratio_%s/ratios.csv'%ACQUISITION_FUNC
    N_TOTAL_EXP = 100000

    print('GAMMA: ', GAMMA)
    print('GAMMA_Y: ', GAMMA_Y)
    print('GAMMA0:', GAMMA0)

    MU2RATIO_DIR = './mu2ratio_%s'%ACQUISITION_FUNC
    mkdir_if_not_exist(OUTPUT_DIR)
    mkdir_if_not_exist(MU2RATIO_DIR)
    mkdir_if_not_exist("./eval")

    param_names = sorted([x.replace('.csv', '') for x in os.listdir(parameter_dir)])


    bo_param2model_param_dic = {}
    
    bo_param_list = []
    for param_name in param_names:
        param_df = pd.read_csv(os.path.join(parameter_dir, param_name + '.csv'), dtype=str)
        bo_param_list.append(param_df[param_name].values)
    
        param_df.set_index(param_name, inplace=True)
    
        bo_param2model_param_dic[param_name] = param_df.to_dict()['bo_' + param_name]
    
    env = ClickTwoDimGaussianEnvironment(bo_param2model_param_dic=bo_param2model_param_dic, 
                                         result_filename=RESULT_FILENAME,
                                         output_dir=OUTPUT_DIR,
                                         reload=reload)
    
    agent = GMRF_BO(bo_param_list, env, GAMMA=GAMMA, GAMMA0=GAMMA0, GAMMA_Y=GAMMA_Y, ALPHA=ALPHA,
                    is_edge_normalized=IS_EDGE_NORMALIZED, gt_available=True, n_early_stopping=N_EARLY_STOPPING,
                    burnin=BURNIN,
                    normalize_output=NORMALIZE_OUTPUT, update_hyperparam_func=UPDATE_HYPERPARAM_FUNC,
                    initial_k=INITIAL_K, initial_theta=INITIAL_THETA, acquisition_func=ACQUISITION_FUNC,
                    acquisition_param_dic=ACQUISITION_PARAM_DIC, n_ctr=N_TOTAL_EXP)


    #agent.plot_click_distribution(output_dir)
    agent.save_mu_sigma_csv(outfn=mu_sigma_csv_path)

# agent.learn_from_clicks()
    nIter = 100
    for i in range(nIter):
        try:
            flg = agent.learn_from_clicks(mu2ratio_dir=MU2RATIO_DIR, 
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
            exit(0)
    os.system("mv %s/*.csv ./eval/"%OUTPUT_DIR)
    
def main(argv):
    nTrials = 30
    for trial in range(nTrials):
        singleTest(argv[0], trial)
    
if __name__ == "__main__":
    main(sys.argv[1:])
    
