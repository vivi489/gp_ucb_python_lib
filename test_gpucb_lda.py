# coding: utf-8
import os
import pickle

import numpy as np
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

from env import LDA_Environment
from gpucb import GPUCB

original_config_dict = {
    "filename_training": "./input/input.txt",
    "filename_result": "./output/output.txt",
    "filename_testing": "./input/input.txt",
    "filename_model": "./output/model",
    "pathname_dump": "./dump/",
    "HAS_ID": 0,
    "ALPHA": 1,
    "FEATURE_NUM": 1,
    "CLUSTER_NUM": 8,
    "DATA_SIZE": 0,
    "DOC_SIZE": 0,
    "THREAD_NUM": 2,
    "start": 0,
    "end": 1000,
    "TYPE_LIST": [
        "Categorical:0,1,2,3,4,5,6,7,8,9;1"
    ]
}

### Setting Needed ###
my_hyper_param_dic = {
    "alpha": np.arange(-2, 2.01, 0.2),
    "beta": np.arange(-2, 2.01, 0.2),
    "n_cluster": np.arange(5, 20.1).astype(int)
}

output_dir = 'output'
result_filename = 'lda_result.csv'

# reload_model_filename = None
reload_model_filename = os.path.join(output_dir, 'learned_agent.pkl')

output_model_filename = os.path.join(output_dir, 'learned_agent.pkl')

n_iter = 100
########################


if reload_model_filename is not None:
    with open(reload_model_filename, mode='rb') as f:
        agent = pickle.load(f)

else:
    if os.path.exists(result_filename):
        os.remove(result_filename)

    my_sorted_keys = sorted(my_hyper_param_dic.keys())
    env = LDA_Environment(my_sorted_keys, result_filename, output_dir, original_config_dict, lda_n_iter=400)
    agent = GPUCB(np.meshgrid(*[my_hyper_param_dic[k] for k in my_sorted_keys]), env, 36)

for i in tqdm(range(n_iter)):
    try:
        agent.learn()


    # When you stop this process, model will be saved automatically.
    except KeyboardInterrupt:
        print("Learnig process was forced to stop!")
        with open(output_model_filename, mode='wb') as f:
            pickle.dump(agent, f)
            print("%s was saved." % output_model_filename)

        break
