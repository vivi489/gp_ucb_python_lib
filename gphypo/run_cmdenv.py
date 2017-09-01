# coding: utf-8
import json
import os
from string import Template

import numpy as np
import pandas as pd
from tqdm import tqdm

from gphypo.egmrf_ucb import EGMRF_UCB
from .gpucb import GPUCB
from .util import mkdir_if_not_exist


def run_gp(gp_paramter_filename, MyEnvironment):
    with open(gp_paramter_filename, 'r') as f:
        gp_parameter_dic = json.load(f)

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

    cmdline_paramfile_str = ''
    with open(template_cmdline_filename) as f:
        cmdline_paramfile_str += f.read()

    with open(template_parameter_filename) as f:
        cmdline_paramfile_str += f.read()

    template = Template(cmdline_paramfile_str)

    param_names = sorted([x.replace('.csv', '') for x in os.listdir(parameter_dir)])

    for param_name in param_names:
        replaced = template.safe_substitute({param_name: "HOGEHOGE"})
        if cmdline_paramfile_str == replaced:
            param_names.remove(param_name)
            print(
                "%s exist in your param dir. But not exist in your cmdline or parameter file, so %s.csv is ignored" % (
                    param_name, param_name))

        else:
            print("%s is one hyper-paramter!" % param_name)

    gp_param2model_param_dic = {}

    gp_param_list = []
    for param_name in param_names:
        param_df = pd.read_csv(os.path.join(parameter_dir, param_name + '.csv'), dtype=str)
        gp_param_list.append(param_df[param_name].values)

        param_df.set_index(param_name, inplace=True)

        gp_param2model_param_dic[param_name] = param_df.to_dict()['transformed_' + param_name]

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


def run_gmrf(gmrf_paramter_filename, MyEnvironment):
    with open(gmrf_paramter_filename, 'r') as f:
        gmrf_parameter_dic = json.load(f)

    output_dir = gmrf_parameter_dic['output_dir']
    mkdir_if_not_exist(output_dir)
    with open(os.path.join(output_dir, 'parameter_gmrf_output.json'), 'w') as f:
        json.dump(gmrf_parameter_dic, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    result_filename = gmrf_parameter_dic['result_filename']
    parameter_dir = gmrf_parameter_dic['parameter_dir']
    template_cmdline_filename = gmrf_parameter_dic['template_cmdline_filename']
    template_parameter_filename = gmrf_parameter_dic['template_parameter_filename']
    reload = gmrf_parameter_dic['reload']
    n_iter = gmrf_parameter_dic['n_iter']
    n_exp = gmrf_parameter_dic['n_exp']
    beta = gmrf_parameter_dic['beta']
    ALPHA = gmrf_parameter_dic['ALPHA']
    GAMMA = gmrf_parameter_dic['GAMMA']
    GAMMA0 = gmrf_parameter_dic['GAMMA0']
    GAMMA_Y = gmrf_parameter_dic['reload']
    IS_EDGE_NORMALIZED = gmrf_parameter_dic['IS_EDGE_NORMALIZED']
    UPDATE_HYPERPARAM_FUNC = gmrf_parameter_dic['UPDATE_HYPERPARAM_FUNC']
    N_EARLY_STOPPING = gmrf_parameter_dic['N_EARLY_STOPPING']
    BURNIN = gmrf_parameter_dic['BURNIN']
    NORMALIZE_OUTPUT = gmrf_parameter_dic['NORMALIZE_OUTPUT']

    cmdline_paramfile_str = ''
    with open(template_cmdline_filename) as f:
        cmdline_paramfile_str += f.read()

    with open(template_parameter_filename) as f:
        cmdline_paramfile_str += f.read()

    template = Template(cmdline_paramfile_str)

    param_names = sorted([x.replace('.csv', '') for x in os.listdir(parameter_dir)])

    for param_name in param_names:
        replaced = template.safe_substitute({param_name: "HOGEHOGE"})
        if cmdline_paramfile_str == replaced:
            param_names.remove(param_name)
            print(
                "%s exist in your param dir. But not exist in your cmdline or parameter file, so %s.csv is ignored" % (
                    param_name, param_name))

        else:
            print("%s is one hyper-paramter!" % param_name)

    gmrf_param2model_param_dic = {}

    gmrf_param_list = []
    for param_name in param_names:
        param_df = pd.read_csv(os.path.join(parameter_dir, param_name + '.csv'), dtype=str)
        gmrf_param_list.append(param_df[param_name].values)

        param_df.set_index(param_name, inplace=True)

        gmrf_param2model_param_dic[param_name] = param_df.to_dict()['transformed_' + param_name]

    env = MyEnvironment(gp_param2model_param_dic=gmrf_param2model_param_dic,
                        template_cmdline_filename=template_cmdline_filename,
                        template_paramter_filename=template_parameter_filename,
                        result_filename=result_filename,
                        output_dir=output_dir,
                        reload=reload)

    agent = EGMRF_UCB(np.meshgrid(*gmrf_param_list), env, GAMMA=GAMMA, GAMMA0=GAMMA0, GAMMA_Y=GAMMA_Y, ALPHA=ALPHA,
                      BETA=beta,
                      is_edge_normalized=IS_EDGE_NORMALIZED, n_early_stopping=N_EARLY_STOPPING,
                      burnin=BURNIN,
                      normalize_output=NORMALIZE_OUTPUT, update_hyperparam_func=UPDATE_HYPERPARAM_FUNC, n_exp=n_exp)

    for i in tqdm(range(n_iter)):
        try:
            agent.learn()
            agent.plot(output_dir=output_dir)

        except KeyboardInterrupt:
            print("Learnig process was forced to stop!")
            break
