# coding: utf-8
import subprocess
from multiprocessing import Process, Pool


        
if __name__ == '__main__':
    acFuncs = ["ucb", "en", "ts"]
    jobs = []
    for acFunc in acFuncs:
        p = subprocess.Popen(["python", "./test40.py", acFunc], stderr=None)
        jobs.append(p)
    for p in jobs: p.wait()
