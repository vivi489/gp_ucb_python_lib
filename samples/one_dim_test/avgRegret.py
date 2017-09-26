# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os, matplotlib
import matplotlib.pyplot as plt

from scipy.stats import norm


def get_all_fx(acFunc, nTrials, eval_csv_dir):
    list_fx = []
    for i in range(nTrials):
        eval_csv_path = os.path.join(eval_csv_dir, "gaussian_result_1dim_%s_trialCount_%d.csv"%(acFunc, i))
        list_fx.append(np.array(pd.read_csv(eval_csv_path, index_col=None)["output"]))
    return np.array(list_fx) #fx_matrix

def computeRunningAvgRegret(fx_matrix, gTruthFunc, XRange):
    #fx_matrix has one iter. fx on each row
    maxMask = np.zeros_like(fx_matrix) + max(gTruthFunc(XRange))
    fx_matrix = maxMask - fx_matrix
    avgRegretsPerStep = fx_matrix.mean(axis=0)
    runningAvg = []
    for k in range(len(avgRegretsPerStep)):
        curAvg = 0 if len(runningAvg)==0 else runningAvg[-1]
        N = len(runningAvg) + 1
        runningAvg.append(curAvg * (N-1) / N + avgRegretsPerStep[k] / N)
    return runningAvg
    
    
acFuncs = ["ucb", "pi", "ei", "ts", "greedy"]
nTrials = 20
eval_csv_path = "./eval"

font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

if __name__ == '__main__':
    runningAvgRegret = {}
    colors = {"ucb":"black", "pi":"orange", "ei":"blue", "ts":"red", "greedy":"green"}
    
    gTruthFunc = lambda x: norm.pdf(x, loc=-3, scale=0.15) + norm.pdf(x, loc=3, scale=0.7) + norm.pdf(x, loc=0, scale=1.5)
    XRange = np.linspace(-5.0, 5.0, 1000)
    
    for acFunc in acFuncs:
        runningAvgRegret[acFunc] = computeRunningAvgRegret(get_all_fx(acFunc, nTrials, eval_csv_path), gTruthFunc, XRange)

    handles = []
    plt.figure(figsize=(15, 10))
    for k, v in runningAvgRegret.items():
        handle, = plt.plot(v, color=colors[k], label=k)
        handles.append(handle)
    plt.legend(handles, acFuncs)
    plt.savefig("eval.eps", format='eps')
    plt.close()








