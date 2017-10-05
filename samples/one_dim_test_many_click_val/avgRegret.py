import pandas as pd
import numpy as np
import os
from scipy.stats import norm
import matplotlib.pyplot as plt



ACQUISITION_FUNC = "ts"
EVAL_CSV_DIR = "./output_%s_clicks"%ACQUISITION_FUNC




df = pd.DataFrame.from_csv(os.path.join(EVAL_CSV_DIR, "gaussian_result_1dim_clicks_%s_trialCount_0.csv"%ACQUISITION_FUNC), index_col=None)

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

x = np.array(range(np.arange(-5, 5.1, 0.1).shape[0]))
y = computerRegret(df, pdf, x.shape[0])

nIter = 100
plt.plot(y[:nIter])
plt.title("1d 10000-click regret")
plt.show()
