# coding: utf-8
import copy
import json
import os
import subprocess
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from util import mkdir_if_not_exist


class BasicEnvironment(object):
    __metaclass__ = ABCMeta

    def __init__(self, parameter_filename="param.csv", result_filename='result.csv', output_dir='output',
                 reload=False):
        self.result_filename = result_filename
        self.reload = reload

        self.parameter_df = pd.read_csv(parameter_filename)

        self.gp_param_names = sorted([x for x in self.parameter_df.columns.tolist() if x.find('gp_') >= 0])
        self.param_names = sorted([x for x in self.parameter_df.columns.tolist() if x.find('gp_') < 0])

        self.parameter_df = self.parameter_df[self.gp_param_names + self.param_names]

        if os.path.exists(result_filename):
            if reload:
                print(result_filename + " will be loaded!!")
            else:
                msg = "Oops! %s has already existed... Please change the filename or set reload flag to be true!" % result_filename
                raise AttributeError(msg)

        else:
            if reload:
                msg = "Oops! Reload flag is true, but %s does not exist..." % result_filename
                raise AttributeError(msg)
            else:
                with open(result_filename, 'w') as f:
                    columns = self.gp_param_names + self.param_names + ['output']
                    f.write(','.join(columns) + os.linesep)

                print(result_filename + " is created!")

        self.result_df = pd.read_csv(result_filename)
        mkdir_if_not_exist(output_dir)
        self.output_dir = output_dir

    @abstractmethod
    def run_model(self, n_model, idx):
        pass

    def sample(self, x, idx, get_ground_truth=False):
        if get_ground_truth:
            result = self.run_model(0, )
            return result

        n_model = self.result_df.shape[0] + 1

        prefix_msg = 'No.%04d model started!  ' % n_model
        pair_msg = ', '.join(['{}: {}'.format(k, v) for k, v in zip(self.param_names, x)])
        print(prefix_msg + pair_msg)

        result = self.run_model(n_model, idx)

        self.result_df.loc[len(self.result_df)] = self.parameter_df.iloc[idx, :].values.tolist() + [result]
        self.result_df.to_csv(self.result_filename, index=False)

        msg = 'No.%04d model finished! Result was %f' % (n_model, result)
        print(msg)

        return result


class GaussianEnvironment(BasicEnvironment):
    def __init__(self, parameter_filename, result_filename, output_dir, reload):
        super().__init__(parameter_filename, result_filename, output_dir, reload)

    def run_model(self, model_number, idx):
        x = self.parameter_df[self.gp_param_names].iloc[idx, :].as_matrix()

        mean1 = [3, 3]
        cov1 = [[2, 0], [0, 2]]

        mean2 = [-2, -2]
        cov2 = [[1, 0], [0, 1]]

        mean3 = [3, -3]
        cov3 = [[0.6, 0], [0, 0.6]]

        if x.ndim == 1:
            pass
        elif x.ndim == 2:
            x = x.T
        else:
            return "OOPS"

        y = multivariate_normal.pdf(x, mean=mean1, cov=cov1) + multivariate_normal.pdf(x, mean=mean2, cov=cov2) \
            + multivariate_normal.pdf(x, mean=mean3, cov=cov3)

        return y


class LDA_Environment(BasicEnvironment):
    def __init__(self, parameter_filename="lda_param.csv", result_filename='result.csv',
                 output_dir='output', reload=False):
        super().__init__(parameter_filename, result_filename, output_dir, reload=reload)

        self.default_config_dic = {
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
            "end": 10,
            "TYPE_LIST": [
                "Categorical:0,1,2,3,4,5,6,7,8,9;1"
            ]
        }

    def set_my_config(self, model_number, series):
        config = copy.deepcopy(self.default_config_dic)

        config['filename_result'] = './output/output%04d.txt' % model_number
        config['filename_model'] = './output/model%04d' % model_number
        config["pathname_dump"] = "./dump/dump%04d/" % model_number

        if 'alpha' in series.keys():
            config['ALPHA'] = series.alpha
        if 'beta' in series.keys():
            config['TYPE_LIST'][0] = "Categorical:0,1,2,3,4,5,6,7,8,9;%f" % series.beta
        if 'n_cluster' in series.keys():
            config['CLUSTER_NUM'] = series.n_cluster

        mkdir_if_not_exist(config["pathname_dump"])

        return config

    def run_model(self, model_number, idx):
        series = self.parameter_df.iloc[idx, :]
        conf = self.set_my_config(model_number, series)
        conf_fn = os.path.join(self.output_dir, 'conf%04d.json' % model_number)

        with open(conf_fn, "w") as f:
            json.dump(conf, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

        cmd = "java -jar IndependentMixtureModelMCMC.jar training %s" % conf_fn

        subprocess.call(cmd, shell=True)

        loglikelihood = np.loadtxt(conf['pathname_dump'] + 'loglikelihood.dmp')[-1]

        return loglikelihood
