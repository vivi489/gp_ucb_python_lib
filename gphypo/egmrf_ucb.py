# coding: utf-8
import os

import matplotlib

matplotlib.use('Agg')

from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from .util import mkdir_if_not_exist
import numpy as np
from scipy.spatial.distance import pdist, squareform


def create_normalized_meshgrid(meshgrid):
    mesh_shape = meshgrid[0].shape
    fixed_shape = mesh_shape[:2][::-1] + mesh_shape[2:]

    mesh_width_list = []
    mesh_range_list = []
    for i, n_coorinate in enumerate(fixed_shape):
        original_min = meshgrid[i].min()
        original_max = meshgrid[i].max()

        width = (original_max - original_min) / (n_coorinate - 1)
        mesh_width_list.append(width)

        mesh_range_list.append((original_min, original_max + width))

    normalized_width = min(mesh_width_list)

    normalized_mesh_list = []
    for mesh_width, mesh_range in zip(mesh_width_list, mesh_range_list):
        ratio = normalized_width / mesh_width

        mesh = np.arange(mesh_range[0] * ratio, mesh_range[1] * ratio, normalized_width)
        normalized_mesh_list.append(mesh)

    return normalized_width, np.meshgrid(*normalized_mesh_list)


class EGMRF_UCB(object):
    def __init__(self, meshgrid, environment, GAMMA=0.01, GAMMA0=0.01, Y_GAMMA=0.01, ALPHA=0.1, BETA=16, noise=True,
                 gt_available=False):
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
        self.GAMMA = GAMMA
        self.GAMMA0 = GAMMA0
        self.ALPHA = ALPHA
        self.BETA = BETA

        self.environment = environment

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

        self.X = []
        self.T = []

        if environment.reload:
            X = environment.result_df[environment.gp_param_names].as_matrix()
            T = environment.result_df['output'].as_matrix()

            self.X = [np.array(x) for x in X.tolist()]
            self.T = T.tolist()
            print("Reloaded csv.")

        normalized_width, normalized_meshgrid = create_normalized_meshgrid(list(self.meshgrid))
        normalized_meshgrid = np.array(normalized_meshgrid)
        normalized_X_grid = normalized_meshgrid.reshape(normalized_meshgrid.shape[0], -1).T

        tau0 = pdist(normalized_X_grid, metric='sqeuclidean')
        tau0[tau0 == normalized_width ** 2] = -Y_GAMMA
        tau0[tau0 != -Y_GAMMA] = 0

        self.tau0 = squareform(tau0)

        self.update()

    def update(self):
        r_grid = []
        np_T = np.array(self.T)
        for X in self.X_grid:
            idx = tuple(X)
            tgt_idxes = [i for i, X in enumerate(self.X) if tuple(X) == idx]
            r_grid.append(np_T[tgt_idxes])

        n_grid = np.array([len(r) for r in r_grid])

        gammma_tilda = n_grid * self.GAMMA + self.GAMMA0

        tau1 = - np.diag(gammma_tilda)
        tau2 = np.zeros_like(tau1)
        tmp1 = np.concatenate([self.tau0, tau1])
        tmp2 = np.concatenate([tau1, tau2])
        all_tau = np.concatenate([tmp1, tmp2], axis=1)
        diag = np.sum(all_tau, axis=1)
        all_tau[np.diag_indices_from(all_tau)] = -diag
        tau = all_tau[:self.tau0.shape[0], :self.tau0.shape[1]]

        cov = np.linalg.inv(tau)  # TODO: should use cholesky like "L = cholesky(tau)"

        mu_tilda = np.array([r.sum() * self.GAMMA + self.ALPHA * self.GAMMA0 for r in r_grid]) / gammma_tilda

        self.mu = cov.dot(-tau1).dot(mu_tilda)
        self.sigma = np.sqrt(cov[np.diag_indices_from(cov)])

    def argmax_ucb(self):
        ucb = np.argmax(self.mu + self.sigma * np.sqrt(self.BETA))
        return ucb

    def learn(self):
        grid_idx = self.argmax_ucb()
        self.sample(self.X_grid[grid_idx])
        self.update()

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

            ucb_score = self.mu + self.sigma * np.sqrt(self.BETA)
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
            ucb_score = self.mu + self.sigma * np.sqrt(self.BETA)
            plt.plot(self.meshgrid[0], ucb_score.reshape(self.meshgrid[0].shape), color='y')

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
