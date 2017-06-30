import re
import sys

sys.path.append("/Users/ar-kohei.watanabe/Codes/gp_ucb_python_lib/lib")
from env import Cmdline_Environment


class MyEnvironment(Cmdline_Environment):
    def get_result(self):
        with open(self.parameter_dic['filename_result']) as f:
            line = f.readline()

        m = re.search(r"Accuracy *= *(\d+[\.]\d+)%", line)
        if m is None:
            m = re.search(r"Accuracy *= *(\d+)%", line)

        res = float(m.group(1))
        return res * 0.01
