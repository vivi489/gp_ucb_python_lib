import numpy as np

from env import Cmdline_Environment


class MyEnvironment(Cmdline_Environment):
    def get_result(self):
        return np.loadtxt(self.parameter_dic['pathname_dump'] + 'loglikelihood.dmp')[-1]
