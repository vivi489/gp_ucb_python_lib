import sys

import numpy as np

sys.path.append("/Users/ar-kohei.watanabe/Codes/gp_ucb_python_lib/lib")
from env import Cmdline_Environment


class MyEnvironment(Cmdline_Environment):
    def get_result(self):
        return np.loadtxt(self.parameter_dic['pathname_dump'] + 'loglikelihood.dmp')[-1]
