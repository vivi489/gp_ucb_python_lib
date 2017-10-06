import pandas as pd
import numpy as np
import os
from scipy.stats import norm
import matplotlib.pyplot as plt






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
    
    for acFunc in acFuncs:
        listRegretsAllTrials = []
        for i in range(nTrials):
            eval_csv_dir = "./output_%s_clicks"%acFunc
            eval_csv_path = os.path.join(eval_csv_dir, "gaussian_result_1dim_%s_trialCount_%d.csv"%(acFunc, i))
            listRegretsAllTrials.append(np.array(pd.read_csv(eval_csv_path, index_col=None)["output"]))
            df = pd.DataFrame.from_csv(os.path.join(eval_csv_path, "gaussian_result_1dim_clicks_%s_trialCount_0.csv"%acFunc), index_col=None)
            y = computeRegretOneIter(df, pdf, x.shape[0])
            listRegretsAllTrials.append(y)
            
        runningAvgRegret[acFunc] = computeRunningAvgRegret(get_all_fx(acFunc, nTrials, eval_csv_path), gTruthFunc, XRange)


    
    
    
    
    
    
    
    
    
    
    
    
    