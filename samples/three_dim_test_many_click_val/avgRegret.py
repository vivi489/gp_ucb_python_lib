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
    lenX = X.shape[0] * X.shape[1] * X.shape[2]
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
    mean1 = [3, 3, 3]
    cov1 = np.eye(3) * 0.75
    mean2 = [-2, -2, -2]
    cov2 = np.eye(3) * 0.75
    mean3 = [0, 0, 0]
    cov3 = np.eye(3) * 1.0
    prob = multivariate_normal.pdf(x, mean=mean1, cov=cov1) + multivariate_normal.pdf(x, mean=mean2, cov=cov2) \
               + multivariate_normal.pdf(x, mean=mean3, cov=cov3)
    return prob * 3


nTrials = 3
acFuncs = ["ucb", "ts", "greedy", "ei"] #["ucb", "ts", "greedy", "ei", "pi"]

font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

if __name__ == '__main__':
    runningAvgRegret = {}
    colors = {"ucb":"black", "pi":"brown", "ei":"blue", "ts":"red", "greedy":"green"}
    
    x, y, z = np.array(np.meshgrid(*(np.arange(-5, 5.0, 2) for _ in range(3))))
    X = np.empty(x.shape + (3,))
    X[:, :, :, 0] = x; X[:, :, :, 1] = y; X[:, :, :, 2] = z
    
    for acFunc in acFuncs:
        listAvgRegretsAllTrials = []
        for i in range(nTrials):
            eval_csv_path = os.path.join("./eval", "gaussian_result_3dim_clicks_%s_trialCount_%d.csv"%(acFunc, i))
            df = pd.DataFrame.from_csv(eval_csv_path, index_col=None)
            y = computerRegret(df, pdf, X)
            listAvgRegretsAllTrials.append(y)
        runningAvgRegret[acFunc] = np.array(listAvgRegretsAllTrials).mean(axis=0)

    handles = []
    plt.figure(figsize=(15, 10))
    for k, v in runningAvgRegret.items():
        handle, = plt.plot(v, color=colors[k], label=k)
        handles.append(handle)
    plt.legend(handles, acFuncs)
    plt.savefig("eval.eps", format='eps')
    plt.close() 
    
    
    
    
    
    
    
    
    
    
