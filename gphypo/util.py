import os
from matplotlib import pyplot as plt

def mkdir_if_not_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    else:
        pass


def plot_1dim(aggregated_list, output_filename='loss.png'):
    if type(aggregated_list) is not list:
        aggregated_list = [aggregated_list]

    for one_list in aggregated_list:
        plt.plot(one_list)
    plt.savefig(output_filename)
    plt.close()

