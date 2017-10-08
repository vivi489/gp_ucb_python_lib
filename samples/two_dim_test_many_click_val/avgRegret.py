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

def computerRegret(df, truthPdf, X):
    assert df.shape[0] % len(X) == 0, "result dataframe has an incorrect size"
    curIndex = 0
    runningAvg = []
    while curIndex + len(X) <= df.shape[0]:
        regret = computeRegretOneIter(df.iloc[curIndex: curIndex+len(X)], truthPdf)
        curAvg = 0 if len(runningAvg)==0 else runningAvg[-1]
        N = len(runningAvg) + 1
        runningAvg.append(curAvg * (N-1) / N + regret / N)
        curIndex += len(X)
    return runningAvg

def pdf(x):
    prob = norm.pdf(x, loc=-2, scale=0.3) + norm.pdf(x, loc=3, scale=0.7) + norm.pdf(x, loc=0, scale=1.5)
    return prob / 3.0



nTrials = 30
acFuncs = ["ucb",  "ts", "greedy"]

font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

if __name__ == '__main__':
    runningAvgRegret = {}
    colors = {"ucb":"black", "pi":"orange", "ei":"blue", "ts":"red", "greedy":"green"}
    X = np.arange(-5.0, 5.1, 0.1)
    
    for acFunc in acFuncs:
        listRegretsAllTrials = []
        eval_csv_dir = "./eval"
        for i in range(nTrials):
            eval_csv_path = os.path.join(eval_csv_dir, "gaussian_result_1dim_clicks_%s_trialCount_%d.csv"%(acFunc, i))
            df = pd.DataFrame.from_csv(eval_csv_path, index_col=None)
            y = computerRegret(df, pdf, X)
            listRegretsAllTrials.append(y)
        print(len(listRegretsAllTrials))
        listRegretsAllTrials = np.array(listRegretsAllTrials).mean(axis=0)
        runningAvgRegret[acFunc] = listRegretsAllTrials
    handles = []
    plt.figure(figsize=(15, 10))
    for k, v in runningAvgRegret.items():
        handle, = plt.plot(v, color=colors[k], label=k)
        handles.append(handle)
    plt.legend(handles, acFuncs)
    plt.savefig("eval.eps", format='eps')
    plt.close()


    
    
    
    
    
    
    
    
    
    
    
    
    