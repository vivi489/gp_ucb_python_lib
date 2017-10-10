import pandas as pd
import numpy as np
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib

def computeRegretOneIter(df, truthPdf):
    optimalClickCount = df["n_exp"].sum() * truthPdf(df["bo_x"]).max()
    realClickCount = df["output"].sum()
    return optimalClickCount - realClickCount

def computerRegret(df, truthPdf, lenX):
    assert (df.shape[0] % lenX) == 0, "result dataframe has an incorrect size"
    curIndex = 0
    runningAvg = []
    while curIndex + lenX <= df.shape[0]:
        regret = computeRegretOneIter(df.iloc[curIndex: curIndex+lenX], truthPdf)
        curAvg = 0 if len(runningAvg)==0 else runningAvg[-1]
        N = len(runningAvg) + 1
        runningAvg.append(curAvg * (N-1) / N + regret / N)
        curIndex += lenX
    return runningAvg

def pdf(x):
    prob = norm.pdf(x, loc=-3, scale=0.15) + norm.pdf(x, loc=3, scale=0.7) + norm.pdf(x, loc=-1, scale=1.5)
    return prob / 3.0



nTrials = 1
acFuncs = ["pi", "ts"] #["ucb", "ts", "greedy", "ei", "pi"]

font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

if __name__ == '__main__':
    runningAvgRegret = {}
    colors = {"ucb":"black", "pi":"brown", "ei":"blue", "ts":"red", "greedy":"green"}
    
    x = np.arange(-5.0, 5.1, 0.1)
    for acFunc in acFuncs:
        listAvgRegretsAllTrials = []
        for i in range(0, nTrials):
            eval_csv_path = os.path.join("./eval", "gaussian_result_1dim_clicks_%s_trialCount_%d.csv"%(acFunc, i))
            df = pd.DataFrame.from_csv(eval_csv_path, index_col=None)
            y = computerRegret(df, pdf, len(x))
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


    
    
    
    
    
    
    
    
    
    
    
    
    
