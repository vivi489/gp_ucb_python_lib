# coding: utf-8
import os

import matplotlib
import numpy as np

# matplotlib.use('Agg')

from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

from .util import mkdir_if_not_exist
from .helper import unique_rows


class GPUCB(object):
    def __init__(self, meshgrid, environment, beta=100., noise=True, gt_available=False, my_kernel=None, burnin=0):
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

        self.environment = environment
        self.beta = beta

        self.meshgrid = np.array(meshgrid)
        self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T

        self.mu = np.array([0. for _ in range(self.X_grid.shape[0])])
        self.sigma = np.array([0.5 for _ in range(self.X_grid.shape[0])])

        if self.X_grid.shape[1] == 2 and gt_available:
            nrow, ncol = self.meshgrid.shape[1:]
            self.z = self.environment.sample(self.X_grid.T, get_ground_truth=True).reshape(nrow, ncol)
        elif self.X_grid.shape[1] == 1 and gt_available:
            self.z = self.environment.sample(self.X_grid.flatten(),
                                             get_ground_truth=True)  # TODO: check if this works correctly
        else:
            self.z = None
        self.gt_available = gt_available

        if my_kernel is None:
            # Instanciate a Gaussian Process model
            my_kernel = C(1, constant_value_bounds="fixed") * RBF(2,
                                                                  length_scale_bounds="fixed")  # works well, but not so sharp
            #     my_kernel = Matern(nu=2.5) # good
            if noise:
                my_kernel += WhiteKernel(1e-1)
        else:
            my_kernel = my_kernel

        self.gp = GaussianProcessRegressor(kernel=my_kernel, n_restarts_optimizer=25)

        self.X = []
        self.T = []

        if environment.reload:
            X = environment.result_df[environment.gp_param_names].as_matrix()
            T = environment.result_df['output'].as_matrix()

            self.X = [np.array(x) for x in X.tolist()]
            self.T = T.tolist()

            npX = np.array(self.X)
            npT = np.array(self.T)

            ur = unique_rows(npX)

            self.gp.fit(npX[ur], npT[ur])

            # self.gp.fit(self.X, self.T)
            self.mu, self.sigma = self.gp.predict(self.X_grid, return_std=True)
            print("Reloading model succeeded!")


        if burnin > 0 and not environment.reload:
            for _ in range(burnin):
                n_points = self.X_grid.shape[0]
                rand_idx = np.random.randint(n_points)
                x = self.X_grid[rand_idx]
                self.sample(x)

    def argmax_ucb(self):
        ucb = np.argmax(self.mu + self.sigma * np.sqrt(self.beta))
        return ucb

    def learn(self):
        grid_idx = self.argmax_ucb()
        self.sample(self.X_grid[grid_idx])

        npX = np.array(self.X)
        npT = np.array(self.T)

        ur = unique_rows(npX)

        self.gp.fit(npX[ur], npT[ur])

        # self.gp.fit(self.X, self.T)

        self.mu, self.sigma = self.gp.predict(self.X_grid, return_std=True)

    def sample(self, x):
        t = self.environment.sample(x)
        self.X.append(x)
        self.T.append(t)

    def plot(self, output_dir):
        def plot2d():
            fig = plt.figure()
            ax = Axes3D(fig)

            ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                              self.mu.reshape(self.meshgrid[0].shape), alpha=0.5, color='g')

            ucb_score = self.mu + self.sigma * np.sqrt(self.beta)
            ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                              ucb_score.reshape(self.meshgrid[0].shape), alpha=0.5, color='y')

            if self.gt_available:
                ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1], self.z, alpha=0.3, color='b')

            ax.scatter([x[0] for x in self.X], [x[1] for x in self.X], self.T, c='r',
                       marker='o', alpha=0.5)

            ax.scatter(self.X[-1][0], self.X[-1][1], self.T[-1], c='m', s=50, marker='o', alpha=1.0)

            out_fn = os.path.join(output_dir, 'res_%04d.png' % len(self.X))
            mkdir_if_not_exist(output_dir)

            plt.savefig(out_fn)
            plt.close()

        def plot1d():
            plt.plot(self.meshgrid[0], self.mu.flatten(), color='g')
            ucb_score = self.mu.flatten() + self.sigma.flatten() * np.sqrt(self.beta)
            plt.plot(self.meshgrid[0], ucb_score.reshape(self.meshgrid[0].shape), color='y')

            # plt.plot(self.meshgrid[0], ucb_score.flatten(), color='y')

            if self.gt_available:
                plt.plot(self.meshgrid[0], self.z, alpha=0.3, color='b')

            plt.scatter(self.X, self.T, c='r', s=10, marker='o', alpha=1.0)
            plt.scatter(self.X[-1], self.T[-1], c='m', s=50, marker='o', alpha=1.0)

            out_fn = os.path.join(output_dir, 'res_%04d.png' % len(self.X))
            mkdir_if_not_exist(output_dir)

            plt.savefig(out_fn)
            plt.close()

        if self.X_grid.shape[1] == 2:
            plot2d()
            return

        elif self.X_grid.shape[1] == 1:
            plot1d()
            return

        else:
            print("Sorry... Plotting only supports 1 dim or 2 dim.")
            return
