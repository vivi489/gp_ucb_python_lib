# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os, matplotlib
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal



def get_all_fx(acFunc, nTrials, eval_csv_dir):
    list_fx = []
    for i in range(nTrials):
        eval_csv_path = os.path.join(eval_csv_dir, "gaussian_result_4dim_%s_trialCount_%d.csv"%(acFunc, i))
        list_fx.append(np.array(pd.read_csv(eval_csv_path, index_col=None)["output"]))
    return np.array(list_fx) #fx_matrix

def computeRunningAvgRegret(fx_matrix, gTruthValues):
    #fx_matrix has one iter. fx on each row
    maxMask = np.zeros_like(fx_matrix) + gTruthValues.max()
    fx_matrix = maxMask - fx_matrix
    avgRegretsPerStep = fx_matrix.mean(axis=0)
    runningAvg = []
    for k in range(len(avgRegretsPerStep)):
        curAvg = 0 if len(runningAvg)==0 else runningAvg[-1]
        N = len(runningAvg) + 1
        runningAvg.append(curAvg * (N-1) / N + avgRegretsPerStep[k] / N)
    return runningAvg
    

def run_grid(W, X, Y, Z):
    mean1 = [3, 3, 2, 2]
    cov1 = np.eye(4) * 0.7
    mean2 = [-3, -2, -3, -2]
    cov2 = np.eye(4) * 0.3

    XRange = np.array(np.meshgrid(W, X, Y, Z))
    XRange = XRange.reshape(XRange.shape[0], -1).T
    
    y = multivariate_normal.pdf(XRange, mean=mean1, cov=cov1) + multivariate_normal.pdf(XRange, mean=mean2, cov=cov2)
    return y



acFuncs = ["ucb", "pi", "ei", "greedy", "ts"]
nTrials = 30
eval_csv_dir = "./eval"

font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

if __name__ == '__main__':
    runningAvgRegret = {}
    colors = {"ucb":"black", "pi":"orange", "ei":"blue", "ts":"red", "greedy":"green"}
    gTruthValues = run_grid(*(np.arange(-3, 4, 1) for _ in range(4)))
    
    for acFunc in acFuncs:
        runningAvgRegret[acFunc] = computeRunningAvgRegret(get_all_fx(acFunc, nTrials, eval_csv_dir), gTruthValues)
    
    handles = []
    plt.figure(figsize=(15, 10))
    for k, v in runningAvgRegret.items():
        handle, = plt.plot(v, color=colors[k], label=k)
        handles.append(handle)
    plt.legend(handles, acFuncs)
    plt.savefig("eval.eps", format='eps')
    plt.close()


