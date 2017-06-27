# coding: utf-8
import os

import numpy as np
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm

from env import GaussianEnvironment
from gpucb import GPUCB

### Setting Needed ###
my_hyper_param_dic = {
    "x": np.arange(-5, 5, 0.5),
    "y": np.arange(-5, 5, 0.5)
}

output_dir = 'output_sample'
result_filename = 'sample_result.csv'

output_model_filename = os.path.join(output_dir, 'learned_agent.pkl')

n_iter = 100
########################

my_sorted_keys = sorted(my_hyper_param_dic.keys())
env = GaussianEnvironment(my_sorted_keys, result_filename, output_dir)
agent = GPUCB(np.meshgrid(*[my_hyper_param_dic[k] for k in my_sorted_keys]), env, 36)

for i in tqdm(range(n_iter)):
    try:
        agent.learn()
        agent.plot(output_dir)

    except KeyboardInterrupt:
        print("Learnig process was forced to stop!")

        ## Unfotunately, this cause "RecursionError: maximum recursion depth exceeded"
        # with open(output_model_filename, mode='wb') as f:
        #     dill.dump(agent, f)
        #     print("%s was saved." % output_model_filename)

        break


## Unfotunately, this cause "RecursionError: maximum recursion depth exceeded"
# with open(output_model_filename, mode='wb') as f:
#     dill.dump(agent, f)
#     print("%s was saved." % output_model_filename)
