import subprocess
from multiprocessing import Process, Pool


        
if __name__ == '__main__':
    acFuncs = ["ucb", "ts", "greedy", "ei", "pi"]
    jobs = []
    for acFunc in acFuncs:
        p = subprocess.Popen(["python", "./test30_many_click_val.py", acFunc], stderr=None)
        jobs.append(p)
    for p in jobs: p.wait()
