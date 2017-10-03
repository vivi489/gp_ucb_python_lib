import subprocess
from multiprocessing import Process, Pool


        
if __name__ == '__main__':
    acFuncs = ["ucb", "pi", "ei", "greedy", "ts"]
    jobs = []
    for acFunc in acFuncs:
        p = subprocess.Popen(["python", "./test30.py", acFunc], stderr=None)
        jobs.append(p)
    for p in jobs: p.wait()
