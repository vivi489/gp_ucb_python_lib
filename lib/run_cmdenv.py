# coding: utf-8
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

from gpucb import GPUCB
from util import mkdir_if_not_exist

parser = argparse.ArgumentParser(description='Draw a loss figure')
parser.add_argument('-i', '--input', type=str, default='parameter_gp.json', help='parameter_gp filename')

args = parser.parse_args()

gp_paramter_filename = args.input

with open(gp_paramter_filename, 'r') as f:
    gp_parameter_dic = json.load(f)

full_path = gp_parameter_dic['my_env_path']
env_path, env_filename = os.path.split(full_path)

sys.path.append(env_path)

import_env_str = "from %s import MyEnvironment" % env_filename.replace('.py', '')
exec(import_env_str)

output_dir = gp_parameter_dic['output_dir']
mkdir_if_not_exist(output_dir)
with open(os.path.join(output_dir, 'parameter_gp_output.json'), 'w') as f:
    json.dump(gp_parameter_dic, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

result_filename = gp_parameter_dic['result_filename']
parameter_dir = gp_parameter_dic['parameter_dir']
template_cmdline_filename = gp_parameter_dic['template_cmdline_filename']
template_parameter_filename = gp_parameter_dic['template_parameter_filename']
reload = gp_parameter_dic['reload']
n_iter = gp_parameter_dic['n_iter']
beta = gp_parameter_dic['beta']
noise = gp_parameter_dic['noise']

param_names = sorted([x.replace('.csv', '') for x in os.listdir(parameter_dir)])

gp_param2model_param_dic = {}

gp_param_list = []
for param_name in param_names:
    param_df = pd.read_csv(os.path.join(parameter_dir, param_name + '.csv'))
    gp_param_list.append(param_df[param_name].values)

    param_df.set_index(param_name, inplace=True)

    gp_param2model_param_dic[param_name] = param_df.to_dict()['gp_' + param_name]

env = MyEnvironment(gp_param2model_param_dic=gp_param2model_param_dic,
                    template_cmdline_filename=template_cmdline_filename,
                    template_paramter_filename=template_parameter_filename,
                    result_filename=result_filename,
                    output_dir=output_dir,
                    reload=reload)

agent = GPUCB(np.meshgrid(*gp_param_list), env, beta=beta, noise=noise)

for i in tqdm(range(n_iter)):
    try:
        agent.learn()
        agent.plot(output_dir=output_dir)

    except KeyboardInterrupt:
        print("Learnig process was forced to stop!")
        break
