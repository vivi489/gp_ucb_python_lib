import argparse

import matplotlib
import pandas as pd

matplotlib.use('Agg')

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Draw a loss figure')
parser.add_argument('result_filename', type=str, help='your result csv filename that must have "output" column')
parser.add_argument('-o', '--output', type=str, default='loss.png', help='the output figure filename')

args = parser.parse_args()

res = pd.read_csv(args.result_filename)

plt.plot(res['output'])
plt.savefig(args.output)
