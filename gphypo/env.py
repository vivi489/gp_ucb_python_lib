# coding: utf-8
import json
import os
import subprocess
from abc import ABCMeta, abstractmethod
from string import Template

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from .util import mkdir_if_not_exist


class BasicEnvironment(object):
    __metaclass__ = ABCMeta

    def __init__(self, gp_param2model_param_dic, result_filename='result.csv', output_dir='output',
                 reload=False):
        self.result_filename = result_filename
        self.reload = reload

        self.param_names = sorted(gp_param2model_param_dic.keys())
        self.gp_param_names = ['gp_' + x for x in sorted(gp_param2model_param_dic.keys())]
        self.gp_param2model_param_dic = gp_param2model_param_dic

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

    def preprocess_x(self, x):
        x = np.array(x)

        assert x.ndim == 1
        assert len(x) == len(
            self.gp_param2model_param_dic), "oops! len(x)=%d, len(self.gp_param2model_param_dic)=%d" % (
            len(x), len(self.gp_param2model_param_dic))

        res = np.zeros_like(x)
        for i, (key, gp2model) in enumerate(self.gp_param2model_param_dic.items()):
            res[i] = gp2model[x[i]]

        return res

    @abstractmethod
    def run_model(self, n_model, x):
        pass

    def sample(self, x, get_ground_truth=False):
        if get_ground_truth:
            result = self.run_model(0, )
            return result

        n_model = self.result_df.shape[0] + 1

        processed_x = self.preprocess_x(x)

        prefix_msg = 'No.%04d model started!  ' % n_model
        pair_msg = ', '.join(['{}: {}'.format(k, v) for k, v in zip(self.param_names, processed_x)])
        print(prefix_msg + pair_msg)

        result = self.run_model(n_model, processed_x)

        self.result_df.loc[len(self.result_df)] = list(x) + list(processed_x) + [result]

        self.result_df.to_csv(self.result_filename, index=False)

        msg = 'No.%04d model finished! Result was %f' % (n_model, result)
        print(msg)

        return result


class GaussianEnvironment(BasicEnvironment):
    def __init__(self, gp_param2model_param_dic, result_filename, output_dir, reload):
        super().__init__(gp_param2model_param_dic, result_filename, output_dir, reload)

    def run_model(self, model_number, x):

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


class Cmdline_Environment(BasicEnvironment):
    def __init__(self, gp_param2model_param_dic, template_cmdline_filename, template_paramter_filename=None,
                 result_filename='result.csv',
                 output_dir='output', reload=False, ):
        super().__init__(gp_param2model_param_dic, result_filename, output_dir, reload=reload)

        with open(template_cmdline_filename) as f:
            self.template_cmdline = Template(f.read())

        if template_paramter_filename:
            with open(template_paramter_filename) as f:
                self.template_paramter = Template(f.read())
        else:
            self.template_paramter = None

    @abstractmethod
    def get_result(self):
        pass

    def run_model(self, model_number, x):
        my_param_dic = {k: one_x for k, one_x in zip(self.param_names, list(x))}
        my_param_dic['model_number'] = "%04d" % model_number

        # rewrite your_model_parameter.json below
        if self.template_paramter is not None:
            # self.conf = self.set_my_config(my_param_dic)
            self.parameter_dic = json.loads(
                self.template_paramter.substitute(my_param_dic))  ## TODO: should support yaml, etc

            if "pathname_dump" in self.parameter_dic.keys():
                mkdir_if_not_exist(self.parameter_dic["pathname_dump"])  ## TODO: only for LDA

            conf_fn = os.path.join(self.output_dir, 'conf%04d.json' % model_number)

            with open(conf_fn, "w") as f:
                json.dump(self.parameter_dic, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

            my_param_dic['param_file'] = conf_fn

        # rewrite your cmdline below
        cmd = self.template_cmdline.substitute(my_param_dic)

        subprocess.call(cmd, shell=True)

        loglikelihood = self.get_result()

        return loglikelihood
