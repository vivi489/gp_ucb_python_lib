import os
from matplotlib import pyplot as plt

def mkdir_if_not_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    else:
        pass


def plot_loss(loss_list, output_filename='loss.png'):
    plt.plot(loss_list)
    plt.savefig(output_filename)