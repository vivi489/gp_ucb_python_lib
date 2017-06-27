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

    def __init__(self, sorted_keys, result_filename='result.csv', output_dir='output'):
        self.result_filename = result_filename

        if os.path.exists(result_filename):
            msg = "Oops! %s has already existed... Please change the filename or set reload flag to be true!" % result_filename
            raise AttributeError(msg)

        self.hyper_param_names = sorted_keys

        with open(result_filename, 'w') as f:
            columns = self.hyper_param_names + ['output']
            f.write(','.join(columns) + os.linesep)

        self.df = pd.read_csv(result_filename)

        mkdir_if_not_exist(output_dir)
        self.output_dir = output_dir

    @abstractmethod
    def run_model(self, *args, **kargs):
        pass

    def preprocess_param(self, x):
        return np.array(x)

    def sample(self, x, get_ground_truth=False):
        if get_ground_truth:
            x = self.preprocess_param(x)
            result = self.run_model(0, *x)
            return result

        self.df = pd.read_csv(self.result_filename)
        n_model = self.df.shape[0] + 1

        x = self.preprocess_param(x)

        prefix_msg = 'No.%04d model started!  ' % n_model
        pair_msg = ', '.join(['{}: {}'.format(k, v) for k, v in zip(self.hyper_param_names, x)])
        print(prefix_msg + pair_msg)

        result = self.run_model(n_model, *x)

        self.df.loc[len(self.df)] = list(x) + [result]
        self.df.to_csv(self.result_filename, index=False)

        msg = 'No.%04d model finished! Result was %f' % (n_model, result)
        print(msg)

        return result


class GaussianEnvironment(BasicEnvironment):
    def __init__(self, sorted_keys, result_filename, output_dir):
        super().__init__(sorted_keys, result_filename, output_dir)

    def run_model(self, model_number, *x):
        x = np.array(x)

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
    def __init__(self, sorted_keys, lda_output_filename, output_dir, original_config, default_n_cluster=8,
                 lda_n_iter=400):
        super().__init__(sorted_keys, lda_output_filename, output_dir)

        self.original_config = original_config
        self.default_n_cluster = default_n_cluster
        self.lda_n_iter = lda_n_iter

    def preprocess_param(self, x):
        # key order must be alpha, beta, n_cluster
        y = np.ones_like(x)
        y[:2] = 10 ** x[:2]
        y[2:] = x[2:]
        return y

    def set_my_config(self, model_number, alpha, beta, n_cluster):
        config = copy.deepcopy(self.original_config)

        config['filename_result'] = './output/output%04d.txt' % model_number
        config['filename_model'] = './output/model%04d' % model_number
        config["pathname_dump"] = "./dump/dump%04d/" % model_number

        config['ALPHA'] = alpha
        config['TYPE_LIST'][0] = "Categorical:0,1,2,3,4,5,6,7,8,9;%f" % beta
        config['CLUSTER_NUM'] = n_cluster
        config['end'] = self.lda_n_iter

        mkdir_if_not_exist(config["pathname_dump"])

        return config

    def run_lda(self, model_number, alpha, beta, n_cluster):
        conf = self.set_my_config(model_number, alpha, beta, n_cluster)
        conf_fn = os.path.join(self.output_dir, 'conf%04d.json' % model_number)

        with open(conf_fn, "w") as f:
            json.dump(conf, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

        cmd = "java -jar IndependentMixtureModelMCMC.jar training %s" % conf_fn

        subprocess.call(cmd, shell=True)

        loglikelihood = np.loadtxt(conf['pathname_dump'] + 'loglikelihood.dmp')[-1]

        return loglikelihood

    def run_model(self, model_number, alpha, beta, n_cluster):
        res = self.run_lda(model_number, alpha, beta, n_cluster)
        return res
