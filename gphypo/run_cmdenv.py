# coding: utf-8
import json
import os
from string import Template

import pandas as pd
from tqdm import tqdm

from .gmrf_bo import GMRF_BO
from .gp_bo import GP_BO
from .util import mkdir_if_not_exist


def run_gp(bo_paramter_filename, MyEnvironment):
    with open(bo_paramter_filename, 'r') as f:
        bo_parameter_dic = json.load(f)

    output_dir = bo_parameter_dic['output_dir']
    mkdir_if_not_exist(output_dir)
    with open(os.path.join(output_dir, 'parameter_bo_output.json'), 'w') as f:
        json.dump(bo_parameter_dic, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    result_filename = bo_parameter_dic['result_filename']
    parameter_dir = bo_parameter_dic['parameter_dir']
    template_cmdline_filename = bo_parameter_dic['template_cmdline_filename']
    template_parameter_filename = bo_parameter_dic['template_parameter_filename']
    reload = bo_parameter_dic['reload']
    n_iter = bo_parameter_dic['n_iter']
    noise = bo_parameter_dic['noise']

    kernel = None  # TODO
    ACQUISITION_FUNC = bo_parameter_dic['ACQUISITION_FUNC']
    ACQUISITION_PARAM_DIC = bo_parameter_dic['ACQUISITION_PARAM_DIC']
    NORMALIZE_OUTPUT = bo_parameter_dic['NORMALIZE_OUTPUT']  # TODO
    BURNIN = bo_parameter_dic['BURNIN']  # TODO

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

    bo_param2model_param_dic = {}

    bo_param_list = []
    for param_name in param_names:
        param_df = pd.read_csv(os.path.join(parameter_dir, param_name + '.csv'), dtype=str)
        bo_param_list.append(param_df[param_name].values)

        param_df.set_index(param_name, inplace=True)

        bo_param2model_param_dic[param_name] = param_df.to_dict()['transformed_' + param_name]

    env = MyEnvironment(bo_param2model_param_dic=bo_param2model_param_dic,
                        template_cmdline_filename=template_cmdline_filename,
                        template_paramter_filename=template_parameter_filename,
                        result_filename=result_filename,
                        output_dir=output_dir,
                        reload=reload)

    agent = GP_BO(bo_param_list, env, gt_available=False, my_kernel=kernel, burnin=BURNIN,
                  normalize_output=NORMALIZE_OUTPUT, acquisition_func=ACQUISITION_FUNC,
                  acquisition_param_dic=ACQUISITION_PARAM_DIC, noise=noise)

    for i in tqdm(range(n_iter)):
        try:
            agent.learn()
            agent.plot(output_dir=output_dir)

        except KeyboardInterrupt:
            print("Learnig process was forced to stop!")
            break


def run_gmrf(bo_paramter_filename, MyEnvironment):
    with open(bo_paramter_filename, 'r') as f:
        bo_parameter_dic = json.load(f)

    output_dir = bo_parameter_dic['output_dir']
    mkdir_if_not_exist(output_dir)
    with open(os.path.join(output_dir, 'parameter_bo_output.json'), 'w') as f:
        json.dump(bo_parameter_dic, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    result_filename = bo_parameter_dic['result_filename']
    parameter_dir = bo_parameter_dic['parameter_dir']
    template_cmdline_filename = bo_parameter_dic['template_cmdline_filename']
    template_parameter_filename = bo_parameter_dic['template_parameter_filename']
    reload = bo_parameter_dic['reload']
    n_iter = bo_parameter_dic['n_iter']
    n_exp = bo_parameter_dic['n_exp']
    beta = bo_parameter_dic['beta']
    ALPHA = bo_parameter_dic['ALPHA']
    GAMMA = bo_parameter_dic['GAMMA']
    GAMMA0 = bo_parameter_dic['GAMMA0']
    GAMMA_Y = bo_parameter_dic['reload']
    IS_EDGE_NORMALIZED = bo_parameter_dic['IS_EDGE_NORMALIZED']
    UPDATE_HYPERPARAM_FUNC = bo_parameter_dic['UPDATE_HYPERPARAM_FUNC']
    N_EARLY_STOPPING = bo_parameter_dic['N_EARLY_STOPPING']
    BURNIN = bo_parameter_dic['BURNIN']
    NORMALIZE_OUTPUT = bo_parameter_dic['NORMALIZE_OUTPUT']
    ACQUISITION_FUNC = 'ucb'  # TODO
    ACQUISITION_PARAM_DIC = {'beta': beta}  # TODO

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

    bo_param2model_param_dic = {}

    bo_param_list = []
    for param_name in param_names:
        param_df = pd.read_csv(os.path.join(parameter_dir, param_name + '.csv'), dtype=str)
        bo_param_list.append(param_df[param_name].values)

        param_df.set_index(param_name, inplace=True)

        bo_param2model_param_dic[param_name] = param_df.to_dict()['transformed_' + param_name]

    env = MyEnvironment(bo_param2model_param_dic=bo_param2model_param_dic,
                        template_cmdline_filename=template_cmdline_filename,
                        template_paramter_filename=template_parameter_filename,
                        result_filename=result_filename,
                        output_dir=output_dir,
                        reload=reload)

    agent = GMRF_BO(bo_param_list, env, GAMMA=GAMMA, GAMMA0=GAMMA0, GAMMA_Y=GAMMA_Y, ALPHA=ALPHA,
                    is_edge_normalized=IS_EDGE_NORMALIZED, gt_available=False, n_early_stopping=N_EARLY_STOPPING,
                    burnin=BURNIN,
                    normalize_output=NORMALIZE_OUTPUT, update_hyperparam_func=UPDATE_HYPERPARAM_FUNC,
                    acquisition_func=ACQUISITION_FUNC,
                    acquisition_param_dic=ACQUISITION_PARAM_DIC)

    for i in tqdm(range(n_iter)):
        try:
            agent.learn()
            agent.plot(output_dir=output_dir)

        except KeyboardInterrupt:
            print("Learnig process was forced to stop!")
            break
