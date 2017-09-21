# coding: utf-8
import os, shutil, sys

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from multiprocessing import Process, Pool
from sksparse.cholmod import cholesky


from gphypo.gmrf_bo import GMRF_BO
from gphypo.env import BasicEnvironment
from gphypo.util import mkdir_if_not_exist


# from tqdm import tqdm_notebook as tqdm

class FourDimGaussianEnvironment(BasicEnvironment):
    def __init__(self, bo_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(bo_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x, calc_gt=False, n_exp=1):
        x = np.array(x)

        mean1 = [3, 3, 2, 2]
        cov1 = np.eye(4) * 0.7
        mean2 = [-3, -2, -3, -2]
        cov2 = np.eye(4) * 0.3

        assert x.ndim in [1, 2]

        y = multivariate_normal.pdf(x, mean=mean1, cov=cov1) + multivariate_normal.pdf(x, mean=mean2, cov=cov2)
        return y


########################
ndim = 4  # TODO: 4 dimension does not work...\

BETA = 5  ## sqrt(BETA) controls the ratio between ucb and mean

NORMALIZE_OUTPUT = None
MEAN, STD = 0, 1

reload = False
# reload = True

N_EARLY_STOPPING = 1000

# ALPHA = ndim ** 2  # prior:
ALPHA = ndim   # prior:

GAMMA = 10 ** (-2) * 2 * ndim
GAMMA0 = 0.01 * GAMMA
GAMMA_Y = 10 ** (-2)  # weight of adjacen

INITIAL_K = 10
INITIAL_THETA = 10

IS_EDGE_NORMALIZED = True

BURNIN = True

UPDATE_HYPERPARAM_FUNC = 'pairwise_sampling'  # None

ACQUISITION_PARAM_DIC = {
    'beta': 5, 
    'eps': 0.3,
    "par": 0.01,
    "tsFactor": 2.0
}

# kernel = Matern(2.5)
PARAMETER_DIR = os.path.join('param_dir', 'csv_files')

def singleTest(ACQUISITION_FUNC, trialCount):
    print("%s: trial %d"%(ACQUISITION_FUNC, trialCount))
    OUTPUT_DIR = os.path.join(os.getcwd(), 'output_%s'%ACQUISITION_FUNC)
    
    ########################
    
    ### temporary ###
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    ##################
    RESULT_FILENAME = os.path.join(OUTPUT_DIR, "gaussian_result_4dim_%s_iterCount_%d.csv"%(ACQUISITION_FUNC, trialCount))
    
    print('GAMMA: ', GAMMA)
    print('GAMMA_Y: ', GAMMA_Y)
    print('GAMMA0:', GAMMA0)
    
    mkdir_if_not_exist(OUTPUT_DIR)
    
    param_names = sorted([x.replace('.csv', '') for x in os.listdir(PARAMETER_DIR)])
    
    bo_param2model_param_dic = {}
    
    bo_param_list = []
    
    for param_name in param_names:
        param_df = pd.read_csv(os.path.join(PARAMETER_DIR, param_name + '.csv'), dtype=str)
        bo_param_list.append(param_df[param_name].values)
    
        param_df.set_index(param_name, inplace=True)
    
        bo_param2model_param_dic[param_name] = param_df.to_dict()['gp_' + param_name]
    
    env = FourDimGaussianEnvironment(bo_param2model_param_dic=bo_param2model_param_dic, 
                                        result_filename=RESULT_FILENAME,
                                        output_dir=OUTPUT_DIR,
                                        reload=False)
    
    agent = GMRF_BO(bo_param_list, env, GAMMA=GAMMA, GAMMA0=GAMMA0, GAMMA_Y=GAMMA_Y, ALPHA=ALPHA,
                     is_edge_normalized=IS_EDGE_NORMALIZED, 
                     gt_available=True, 
                     n_early_stopping=N_EARLY_STOPPING,
                     burnin=BURNIN,
                     normalize_output=NORMALIZE_OUTPUT, 
                     update_hyperparam_func=UPDATE_HYPERPARAM_FUNC,
                     initial_k=INITIAL_K, 
                     initial_theta=INITIAL_THETA, 
                     acquisition_func=ACQUISITION_FUNC,
                     acquisition_param_dic=ACQUISITION_PARAM_DIC)
    
    nIter = 500
    for i in range(nIter):
        flg = agent.learn(drop=True if i<nIter-1 else False)
        if not flg:
            print("Early Stopping!!!")
            print(agent.bestX)
            print(agent.bestT)
            break
    os.system("mv %s/*.csv ./eval/"%OUTPUT_DIR)

def testForTrials(acFunc, nIter):
    trialCount = 0
    while trialCount < nIter:
        #np.random.seed(int(time.time()))
        singleTest(acFunc, trialCount)
        trialCount += 1   
        
    
        
if __name__ == '__main__':
#    for ac in :#["ucb", "pi", "ei", "en", "ts"]:
#        iterCount = 0
#        while iterCount < 50:
#            test(ac, iterCount)
#            iterCount += 1
    mkdir_if_not_exist(os.path.join(os.getcwd(), "eval"))
    acFuncs = sys.argv[1]
    nTrials = 30
#    jobs = []
#    for acFuncs, nTrial in zip(acFuncs, nTrials):
#        #testForTrials(acFuncs, nTrial)
#        p = Process(target=testForTrials, args=(acFuncs, nTrial,))
#        jobs.append(p)
#        p.start()
#    
#    for p in jobs:
#        p.join()
    testForTrials(acFuncs, nTrials)

