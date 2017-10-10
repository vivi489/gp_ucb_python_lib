import subprocess
from multiprocessing import Process, Pool


        
if __name__ == '__main__':
    acFuncs = ["ei", "ucb", "greedy", "ts", "pi"]
    jobs = []
    for acFunc in acFuncs:
        p = subprocess.Popen(["python", "./test20.py", acFunc], stderr=None)
        jobs.append(p)
    for p in jobs: p.wait()
