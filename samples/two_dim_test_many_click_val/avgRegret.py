import pandas as pd
import numpy as np
import os
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
import matplotlib

def computeRegretOneIter(df, truthPdf, X):
    optimalClickCount = df["n_exp"].sum() * truthPdf(X).max()
    #print("optimalClickCount = ", optimalClickCount)
    realClickCount = df["output"].sum()
    #print("realClickCount = ", realClickCount)
    return optimalClickCount - realClickCount

def computerRegret(df, truthPdf, X):
    lenX = X.shape[0] * X.shape[1]
    assert (df.shape[0] % lenX) == 0, "result dataframe has an incorrect size"
    curIndex = 0
    runningAvg = []
    while curIndex + lenX <= df.shape[0]:
        regret = computeRegretOneIter(df.iloc[curIndex: curIndex+lenX], truthPdf, X)
        curAvg = 0 if len(runningAvg)==0 else runningAvg[-1]
        N = len(runningAvg) + 1
        runningAvg.append(curAvg * (N-1) / N + regret / N)
        curIndex += lenX
    return runningAvg

def pdf(x):
    mean1 = [-3, 3]
    cov1 = [[0.5, 0], [0, 0.5]]
    
    mean2 = [1, 1]
    cov2 = [[2.5, 0], [0, 2.5]]
    
    mean3 = [3, -3]
    cov3 = [[0.25, 0], [0, 0.25]]
    prob = multivariate_normal.pdf(x, mean=mean1, cov=cov1) + multivariate_normal.pdf(x, mean=mean2, cov=cov2) \
               + multivariate_normal.pdf(x, mean=mean3, cov=cov3)
    prob /= 3.0
    return prob


nTrials = 4
acFuncs = ["ucb", "ts", "greedy", "ei", "pi"]

font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

if __name__ == '__main__':
    runningAvgRegret = {}
    colors = {"ucb":"black", "pi":"brown", "ei":"blue", "ts":"red", "greedy":"green"}
    labels = {"ucb":"GP-UCB", "pi":"PI", "ei":"EI", "ts":"TS", "greedy":"EPS"}
    x, y = np.array(np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1)))
    X = np.empty(x.shape + (2,))
    X[:, :, 0] = x; X[:, :, 1] = y
    
    for acFunc in acFuncs:
        listAvgRegretsAllTrials = []
        for i in range(nTrials):
            eval_csv_path = os.path.join("./eval", "gaussian_result_2dim_clicks_%s_trialCount_%d.csv"%(acFunc, i))
            df = pd.DataFrame.from_csv(eval_csv_path, index_col=None)
            y = computerRegret(df, pdf, X)
            listAvgRegretsAllTrials.append(y)
        runningAvgRegret[acFunc] = np.array(listAvgRegretsAllTrials).mean(axis=0)

    handles = []
    plt.figure(figsize=(15, 10))
    for k, v in runningAvgRegret.items():
        handle, = plt.plot(v, color=colors[k], label=k)
        handles.append(handle)
    plt.xlabel("Iteration Count")
    plt.ylabel("Average Regret")
    plt.yscale("log")
    plt.legend(handles, [labels[k] for k in acFuncs])
    plt.savefig("eval.eps", format='eps')
    plt.close()


    
    
    
    
    
    
    
    
    
    
    
    
    
