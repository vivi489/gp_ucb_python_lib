# coding: utf-8
import os

import numpy as np
from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

from util import mkdir_if_not_exist


class GPUCB(object):
    def __init__(self, meshgrid, environment, beta=100., noise=False):
        '''
        meshgrid: Output from np.methgrid.
        e.g. np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1)) for 2D space
        with |x_i| < 1 constraint. s
        environment: Environment class which is equipped with sample() method to
        return observed value.
        beta (optional): Hyper-parameter to tune the exploration-exploitation
        balance. If beta is large, it emphasizes the variance of the unexplored
        solution solution (i.e. larger curiosity)
        '''
        self.meshgrid = np.array(meshgrid)
        self.environment = environment
        self.beta = beta

        self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T
        self.mu = np.array([0. for _ in range(self.X_grid.shape[0])])
        self.sigma = np.array([0.5 for _ in range(self.X_grid.shape[0])])

        if self.X_grid.shape[1] == 2:
            nrow, ncol = self.meshgrid.shape[1:]
            self.z = self.environment.sample(self.X_grid.T, get_ground_truth=True).reshape(nrow, ncol)

        self.X = []
        self.T = []
        self.kernel_list = []

        # Instanciate a Gaussian Process model
        my_kernel = C(1, constant_value_bounds="fixed") * RBF(2,
                                                              length_scale_bounds="fixed")  # works well, but not so sharp
        #     my_kernel = Matern(nu=2.5) # good
        if noise:
            my_kernel += WhiteKernel(1e-1)

        self.gp = GaussianProcessRegressor(kernel=my_kernel)

    def argmax_ucb(self):
        ucb = np.argmax(self.mu + self.sigma * np.sqrt(self.beta))
        return ucb

    def learn(self):
        grid_idx = self.argmax_ucb()
        self.sample(self.X_grid[grid_idx])

        self.gp.fit(self.X, self.T)
        self.kernel_list.append(self.gp.kernel_)
        self.mu, self.sigma = self.gp.predict(self.X_grid, return_std=True)

    def sample(self, x):
        t = self.environment.sample(x)
        self.X.append(x)
        self.T.append(t)

    def plot(self, output_dir):

        if self.X_grid.shape[1] != 2:
            print("plotting only supports X_grid that consisted of 2 dimentions.")
            return

        fig = plt.figure()
        ax = Axes3D(fig)

        ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                          self.mu.reshape(self.meshgrid[0].shape), alpha=0.5, color='g')

        ucb_mesh = self.mu + self.sigma * np.sqrt(self.beta)
        ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                          ucb_mesh.reshape(self.meshgrid[0].shape), alpha=0.5, color='y')

        #     ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1], self.z, alpha=0.3, color='b')


        ax.scatter([x[0] for x in self.X], [x[1] for x in self.X], self.T, c='r',
                   marker='o', alpha=1.0)

        out_fn = os.path.join(output_dir, 'res_%04d.png' % len(self.X))
        mkdir_if_not_exist(output_dir)

        plt.savefig(out_fn)
        plt.close()
