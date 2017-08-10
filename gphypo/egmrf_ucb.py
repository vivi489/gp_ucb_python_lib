# coding: utf-8
import os
import random
import warnings
from itertools import combinations
from operator import itemgetter

import numpy as np
import scipy
from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fmin_l_bfgs_b
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform

from gphypo import normalization
from gphypo.transform_val import transform_click_val2real_val
from .util import mkdir_if_not_exist


def adj_metric(u, v):
    '''
    give this function to scipy.spatial.distance.dist
    :param u:
    :param v:
    :return: 1 if (u, v) is adj else 0
    '''
    if np.abs(u - v).sum() == 1:
        return 1
    else:
        return 0


def mat_flatten(x):
    return np.array(x).flatten()


def create_normalized_X_grid(meshgrid):
    def create_normalized_meshgrid(meshgrid):
        mesh_shape = meshgrid[0].shape
        fixed_shape = mesh_shape[:2][::-1] + mesh_shape[2:]
        new_mesh = np.meshgrid(*[np.arange(shape) for shape in fixed_shape])
        return new_mesh

    normalized_meshgrid = create_normalized_meshgrid(list(meshgrid))
    normalized_meshgrid = np.array(normalized_meshgrid)
    normalized_X_grid = normalized_meshgrid.reshape(normalized_meshgrid.shape[0], -1).T
    return normalized_X_grid


def create_adjacent_matrix(meshgrid):
    normalized_X_grid = create_normalized_X_grid(meshgrid)
    dist = pdist(normalized_X_grid, metric=adj_metric)  ## TODO
    tau0 = np.zeros_like(dist)

    tau0[dist == 1 ** 2] = -1
    # tau0 = squareform(tau0) ## Memory Consuming

    n_x = normalized_X_grid.shape[0]
    edge_idxes = np.where(dist == 1)[0]
    edge_coordinates = np.array(list(combinations(range(n_x), 2)))[edge_idxes]

    data = - np.ones(edge_coordinates.shape[0])
    tau0 = scipy.sparse.csc_matrix((data, (edge_coordinates[:, 0], edge_coordinates[:, 1])), shape=(n_x, n_x))
    tau0 += scipy.sparse.csc_matrix((data, (edge_coordinates[:, 1], edge_coordinates[:, 0])), shape=(n_x, n_x))

    return tau0


class EGMRF_UCB(object):
    def __init__(self, meshgrid, environment, GAMMA=0.01, GAMMA0=0.01, GAMMA_Y=0.01, ALPHA=0.1, BETA=16,
                 burnin=0, is_edge_normalized=False, noise=True, n_early_stopping=20,
                 gt_available=False, normalize_output=False, optimizer="fmin_l_bfgs_b", update_hyperparam=False,
                 update_only_gamma_y=False, initial_k=1, initial_theta=1, pairwise_sampling=False):
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
        self.GAMMA_Y = GAMMA_Y
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.environment = environment
        self.optimizer = optimizer
        self.update_hyperparam = update_hyperparam

        self.update_only_gamma_y = update_only_gamma_y
        self.k = initial_k
        self.theta = initial_theta

        self.normalize_output = normalize_output

        self.ndim = len(meshgrid)

        self.meshgrid = np.array(meshgrid)
        self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T
        self.normalized_X_grid = create_normalized_X_grid(self.meshgrid)

        self.mu = np.array([0. for _ in range(self.X_grid.shape[0])])
        self.sigma = np.array([0.5 for _ in range(self.X_grid.shape[0])])

        self.gt_available = gt_available

        if self.X_grid.shape[1] == 2 and gt_available:
            nrow, ncol = self.meshgrid.shape[1:]
            self.z = self.environment.sample(self.X_grid.T, get_ground_truth=True).reshape(nrow, ncol)
        elif self.X_grid.shape[1] == 1 and gt_available:
            self.z = self.environment.sample(self.X_grid.flatten(),
                                             get_ground_truth=True)  # TODO: check if this works correctly
        else:
            self.z = None

        self.X = []
        self.T = np.array([])
        self.Treal = np.array([])

        self.bestX = None
        self.bestT = -np.inf
        self.cnt_since_bestT = 0
        self.n_early_stopping = n_early_stopping

        self.pairwise_sampling = pairwise_sampling
        if pairwise_sampling:
            self.paiwise_var_list = np.array([])

        if burnin > 0 and not environment.reload:
            for _ in range(burnin):
                n_points = self.X_grid.shape[0]
                rand_idx = np.random.randint(n_points)
                x = self.X_grid[rand_idx]
                self.sample(x)

            print('%d burins has finised!' % burnin)
            self.bestX = self.X[np.argmax(self.T)]
            self.bestT = self.T.max()

            if self.update_hyperparam:
                self.ALPHA = self.T.mean()
                diff = self.T.std()
                print("mean: %f, std: %f" % (self.T.mean(), self.T.std()))

                self.GAMMA_Y = 2 / ((diff) ** 2)
                self.GAMMA = 2 * self.ndim * self.GAMMA_Y  # 0.01 200  # weight of obserbed data
                self.GAMMA0 = 0.01 * self.GAMMA  # 0.01  # weight of alpha
                print("GAMMA_Y: %f, GAMMA: %f, GAMMA0: %f" % (self.GAMMA_Y, self.GAMMA, self.GAMMA0))

        if environment.reload:
            X = environment.result_df[environment.gp_param_names].as_matrix()
            T = environment.result_df['output'].as_matrix()

            self.X = [np.array(x) for x in X.tolist()]
            self.T = np.array(T)
            print("Reloaded csv.")

        tau0 = create_adjacent_matrix(self.meshgrid)

        # edge_idxes = np.count_nonzero(tau0, axis=1) < self.ndim * 2
        # print('edge num is ' + str(edge_idxes.sum()))

        # row_sum_list = -tau0.sum(axis=1, keepdims=True)
        row_sum_list = - mat_flatten(tau0.sum(axis=0))
        self.diff_list = row_sum_list.max() - row_sum_list

        # TODO this normalization should be reconsidered
        if is_edge_normalized:
            weight_arr = self.ndim * 2 / row_sum_list
            print(weight_arr)

            tau0 *= weight_arr[:, np.newaxis]
            tau0 *= weight_arr

        self.baseTau0 = tau0
        # print (tau0.toarray())
        self.update()

    def calc_true_mean_std(self):
        assert self.gt_available == True
        print(self.X_grid)
        print(self.X_grid.shape)
        sampled_y = np.array(self.environment.sample(self.X_grid.T, get_ground_truth=True))

        return sampled_y.mean(), sampled_y.std()

    def get_r_grid(self):
        r_grid = []

        for X in self.X_grid:
            idx = tuple(X)
            tgt_idxes = [i for i, X in enumerate(self.X) if tuple(X) == idx]
            r_grid.append(self.T[tgt_idxes])

        return np.array(r_grid)

    def calc_tau(self, theta, return_all=False):
        gamma = theta[0]
        gamma_y = theta[1]
        gamma0 = 0.001 * gamma  # TODO hard coding

        r_grid = self.get_r_grid()
        n_grid = np.array([len(r) for r in r_grid])

        # gamma_tilda = n_grid * gamma + gamma0  # Normal
        gamma_tilda = n_grid * gamma + gamma0 + gamma_y * self.diff_list.flatten()  # corner will be treated as center nodes

        mu_tilda = np.array([r.sum() * gamma + self.ALPHA * gamma0 for r in r_grid]) / gamma_tilda

        # tau1 = - np.diag(gamma_tilda)
        # tau2 = np.zeros_like(tau1)

        tau1 = -sp.sparse.diags(gamma_tilda)

        tau0 = self.baseTau0 * self.GAMMA_Y

        # tmp1 = np.concatenate([tau0, tau1])
        # tmp2 = np.concatenate([tau1, tau2])
        # all_tau = np.concatenate([tmp1, tmp2], axis=1)
        # diag = np.sum(all_tau, axis=1)

        # all_tau[np.diag_indices_from(all_tau)] = -diag
        # tau = all_tau[:tau0.shape[0], :tau0.shape[1]]

        tau0 -= sp.sparse.diags(- gamma_tilda + mat_flatten(tau0.sum(axis=0)))
        tau = tau0
        # print (tau.toarray())

        if return_all:
            return tau, -tau1, mu_tilda, gamma_tilda, r_grid, n_grid

        return tau, -tau1, mu_tilda, gamma_tilda

    def update(self, n_start_opt_hyper_param=5):
        theta = [self.GAMMA, self.GAMMA_Y]
        A, B, mu_tilda, gamma_tilda = self.calc_tau(theta)

        # cov = np.linalg.inv(A)  # TODO: should use cholesky like "L = cholesky(tau)"
        cov = sp.sparse.linalg.inv(A)

        # self.mu = cov.dot(B).dot(mu_tilda)
        # self.sigma = np.sqrt(cov[np.diag_indices_from(cov)])

        self.mu = mat_flatten(cov.dot(B).dot(mu_tilda))
        self.sigma = mat_flatten(np.sqrt(cov[np.diag_indices_from(cov)]))
        # print('-' * 100)
        # print (self.mu)
        # print (self.sigma)
        # print('-' * 100)

        if self.update_only_gamma_y and len(self.T) > n_start_opt_hyper_param:
            self.update_gammaY()

        if self.update_hyperparam and len(self.T) > n_start_opt_hyper_param:
            # Update hyper-paramter of GMRF below
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta)

            # optima = [(self._constrained_optimization(obj_func, theta))] # Use when gradient can be calculated
            optima = [(self._constrained_optimization(lambda x: obj_func(x, eval_gradient=False)
                                                      , theta,
                                                      gradExist=False))]  # Use when gradient cannot be calculated

            # Select result from run with minimal (negative) log-marginal likelihood
            lml_values = list(map(itemgetter(1), optima))

            self.GAMMA, self.GAMMA_Y = optima[np.argmin(lml_values)][0]

            self.GAMMA0 = 0.001 * self.GAMMA  # TODO Hard coding

            print("GAMMA: %s, GAMMA_Y: %s" % (self.GAMMA, self.GAMMA_Y))

            self.log_marginal_likelihood_value_ = -np.min(lml_values)
            print("log_marginal_likelihood: %s" % self.log_marginal_likelihood_value_)

    def argmax_ucb(self):
        ucb = np.argmax(self.mu + self.sigma * np.sqrt(self.BETA))
        return ucb

    def learn(self):
        grid_idx = self.argmax_ucb()
        obserbed_val = self.sample(self.X_grid[grid_idx])
        if obserbed_val is None:
            return False

        if self.pairwise_sampling:
            # print (self.baseTau0.toarray())
            # adj_idxes = np.where(self.baseTau0[grid_idx] != 0)[0]
            adj_idxes = sp.sparse.find(self.baseTau0[grid_idx] != 0)[1]

            adj_idx = random.choice(adj_idxes)
            obserbed_val2 = self.sample(self.X_grid[adj_idx])
            if obserbed_val2 is None:
                return False
            self.paiwise_var_list = np.append(self.paiwise_var_list, (obserbed_val2 - obserbed_val) ** 2)

        if len(self.X) > 200:  # TODO Hard coding
            self.update_only_gamma_y = False
            self.pairwise_sampling = False

        self.update()
        return True

    def learn_from_click(self, n_exp=10):
        grid_idx = self.argmax_ucb()
        continue_flg = self.sample_from_click(self.X_grid[grid_idx], n_exp)
        if not continue_flg:
            return False

        self.update()
        return True

    def sample_from_click(self, x, n_exp):
        n1 = self.environment.sample(x, n_exp=n_exp)
        n0 = n_exp - n1
        t = transform_click_val2real_val(n0, n1)

        # TODO change the structure self.X and self.Treal (should not use list because of the problem of consuming-memory when the obserbation is many 01 values)
        for _ in range(n_exp):
            self.X.append(x)
            self.Treal = np.append(self.Treal, t)

        if len(self.Treal) == 1:
            self.T = np.zeros(1)
            self.t_mean = 0
            self.t_std = 1

        else:
            if self.normalize_output:
                # Normalize output to have zero mean and unit standard deviation
                self.T, self.t_mean, self.t_std = normalization.zero_mean_unit_var_normalization(self.Treal)
                # self.T, self.t_mean, self.t_std = normalization.zero_one_normalization(self.Treal)

            else:
                self.T = np.copy(self.Treal)

        if t <= self.bestT:
            self.cnt_since_bestT += 1
        else:
            self.bestT = t
            self.bestX = x
            self.cnt_since_bestT = 0

        if self.cnt_since_bestT > self.n_early_stopping:
            return False

        return True

    def sample(self, x):
        t = self.environment.sample(x)
        self.X.append(x)

        self.Treal = np.append(self.Treal, t)

        if len(self.Treal) == 1:
            self.T = np.zeros(1)
            self.t_mean = 0
            self.t_std = 1

        else:
            if self.normalize_output:
                # Normalize output to have zero mean and unit standard deviation
                self.T, self.t_mean, self.t_std = normalization.zero_mean_unit_var_normalization(self.Treal)
                # self.T, self.t_mean, self.t_std = normalization.zero_one_normalization(self.Treal)

            else:
                self.T = np.copy(self.Treal)

        if t <= self.bestT:
            self.cnt_since_bestT += 1
        else:
            self.bestT = t
            self.bestX = x
            self.cnt_since_bestT = 0

        if self.cnt_since_bestT > self.n_early_stopping:
            return None

        return self.T[-1]

    def update_hyper_params_by_pairwise_sampling(self):
        var = self.paiwise_var_list.sum() / len(self.paiwise_var_list)

        # self.GAMMA_Y = 1 / var / self.ndim
        self.GAMMA_Y = 1 / var

        # self.GAMMA_Y = 1 / var
        print("New GammaY: %s" % self.GAMMA_Y)
        self.GAMMA = self.GAMMA_Y * 2 * self.ndim
        print("New Gamma: %s" % self.GAMMA)
        self.GAMMA0 = self.GAMMA * 0.01

    # TODO this method does not work well...
    def update_gammaY(self):

        if self.pairwise_sampling:
            self.update_hyper_params_by_pairwise_sampling()
            return

        r_grid = self.get_r_grid()

        mean_grid = np.array([r_list.mean() for r_list in r_grid])

        unobserbed_idxes = np.isnan(mean_grid)
        obserbed_idxes = (np.isnan(mean_grid) == False)

        y_hat = mean_grid[obserbed_idxes]

        n_obserbed = len(y_hat)

        new_X_grid = np.concatenate([self.normalized_X_grid[obserbed_idxes], self.normalized_X_grid[unobserbed_idxes]],
                                    axis=0)
        # dist = pdist(new_X_grid, metric='sqeuclidean')  # TODO heavy calculation
        dist = pdist(new_X_grid, metric=adj_metric)

        A = np.zeros_like(dist)
        A[dist == 1 ** 2] = -1
        A = squareform(A)
        diag_arr = - A.sum(axis=0)
        A[np.diag_indices_from(A)] = diag_arr

        A00 = A[:n_obserbed, :n_obserbed]
        A01 = A[n_obserbed:, :n_obserbed]
        A11 = A[n_obserbed:, n_obserbed:]

        Lambda = A00 - (A01.T).dot(np.linalg.inv(A11).dot(A01))

        print(Lambda.shape, A.shape)

        t = y_hat.dot(Lambda.dot(y_hat[:, np.newaxis]))[0] / 2

        self.k += 1 / 2
        self.theta = self.theta / (t * self.theta + 1)
        self.GAMMA_Y = (self.k - 1) * self.theta
        print("New GammaY: %s" % self.GAMMA_Y)
        print("t: %s" % t)
        print('k: %s, theta: %s' % (self.k, self.theta))

        log_likelihood = (y_hat.dot(Lambda.dot(y_hat[:, np.newaxis])) - np.log(np.linalg.det(Lambda) + 0.00001))[0]
        print('log_likelihood: %s' % log_likelihood)

    # TODO this method does not work well...
    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        """Returns log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        A, B, mu_tilda, gamma_tilda, r_grid, n_grid = self.calc_tau(theta=theta, return_all=True)

        # print ( A, B, mu_tilda, gamma_tilda, r_grid, n_grid)
        gamma = theta[0]
        gamma_y = theta[1]

        A1 = np.copy(self.baseTau0)
        A1[np.diag_indices_from(A1)] = 2 * self.ndim

        # A1 = (A - B) / gamma_y
        A2 = np.diag(n_grid)

        B_inv = np.linalg.inv(B)
        A_inv = np.linalg.inv(A)

        M_lambda = B - B.dot(np.linalg.inv(A).dot(B))

        # log_likelihood = mu_tilda.dot(M_lambda).dot(
        #     mu_tilda[:, np.newaxis]) - np.nan_to_num(np.log(np.linalg.det(M_lambda)))
        log_likelihood = mu_tilda.dot(M_lambda).dot(mu_tilda[:, np.newaxis])

        # log_likelihood = -log_likelihood
        print('log_likelihood: %f' % log_likelihood)

        Lambda = B - B.dot(A_inv.dot(B))
        Lambda_inv = np.linalg.inv(Lambda)

        M_gamma = A2 - A2.dot(2 * gamma * A_inv - (gamma ** 2) * A_inv.dot(A2).dot(A_inv)).dot(A2)

        mean_r_grid = np.nan_to_num(np.array([r.sum() for r in r_grid]) / n_grid)

        log_likelihood_grad_gamma = mean_r_grid.dot(M_gamma.dot(mean_r_grid[:, np.newaxis]))[0]

        tmpMat = A2.dot(A_inv).dot(A1).dot(A_inv).dot(A2)

        log_likelihood_grad_gamma_y = (gamma ** 2) * (
            np.trace(Lambda_inv.dot(tmpMat) + mean_r_grid.dot(tmpMat.dot(mean_r_grid[:, np.newaxis]))[0]))

        log_likelihood_gradient = np.array([log_likelihood_grad_gamma, log_likelihood_grad_gamma_y])
        print('log_likelihood_grad: %s' % log_likelihood_gradient)

        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

    def _constrained_optimization(self, obj_func, initial_theta, bounds=[(1e-5, 1e5), (1e-5, 1e5)], gradExist=True):

        if gradExist:
            if self.optimizer == "fmin_l_bfgs_b":
                theta_opt, func_min, convergence_dict = \
                    fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)

                if convergence_dict["warnflag"] != 0:
                    warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                                  " state: %s" % convergence_dict)
                    # pass
            elif callable(self.optimizer):
                theta_opt, func_min = \
                    self.optimizer(obj_func, initial_theta, bounds=bounds)
            else:
                raise ValueError("Unknown optimizer %s." % self.optimizer)
        else:
            theta_opt, func_min, convergence_dict = \
                fmin_l_bfgs_b(obj_func, initial_theta, approx_grad=True, bounds=bounds)

        return theta_opt, func_min

    def plot(self, output_dir):
        def plot2d():
            fig = plt.figure()
            ax = Axes3D(fig)
            ucb_score = self.mu + self.sigma * np.sqrt(self.BETA)
            if self.normalize_output:
                unnormalized_mu = normalization.zero_mean_unit_var_unnormalization(self.mu.flatten(), self.t_mean,
                                                                                   self.t_std)
                ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                                  unnormalized_mu.reshape(self.meshgrid[0].shape), alpha=0.5, color='g')

                ucb_score = normalization.zero_mean_unit_var_unnormalization(ucb_score, self.t_mean, self.t_std)

            else:
                ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                                  self.mu.reshape(self.meshgrid[0].shape), alpha=0.5, color='g')

            ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                              ucb_score.reshape(self.meshgrid[0].shape), alpha=0.5, color='y')

            if self.gt_available:
                ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1], self.z, alpha=0.3, color='b')

            ax.scatter([x[0] for x in self.X], [x[1] for x in self.X], self.Treal, c='r', marker='o', alpha=0.5)

            if self.pairwise_sampling:
                ax.scatter(self.X[-1][0], self.X[-1][1], self.Treal[-1], c='m', s=100, marker='o', alpha=1.0)
                ax.scatter(self.X[-2][0], self.X[-2][1], self.Treal[-2], c='m', s=50, marker='o', alpha=1.0)
            else:
                ax.scatter(self.X[-1][0], self.X[-1][1], self.Treal[-1], c='m', s=50, marker='o', alpha=1.0)

            out_fn = os.path.join(output_dir, 'res_%04d.png' % len(self.X))
            mkdir_if_not_exist(output_dir)

            plt.savefig(out_fn)
            plt.close()

        def plot1d():

            ucb_score = self.mu + self.sigma * np.sqrt(self.BETA)

            if self.normalize_output:
                unnormalized_mu = normalization.zero_mean_unit_var_unnormalization(self.mu.flatten(), self.t_mean,
                                                                                   self.t_std)
                plt.plot(self.meshgrid[0], unnormalized_mu, color='g')

                ucb_score = normalization.zero_mean_unit_var_unnormalization(ucb_score, self.t_mean, self.t_std)
            else:
                plt.plot(self.meshgrid[0], self.mu.flatten(), color='g')

            plt.plot(self.meshgrid[0], ucb_score.reshape(self.meshgrid[0].shape), color='y')

            if self.gt_available:
                plt.plot(self.meshgrid[0], self.z, alpha=0.3, color='b')

            plt.scatter(self.X, self.Treal, c='r', s=10, marker='o', alpha=1.0)

            if self.pairwise_sampling:
                plt.scatter(self.X[-2], self.Treal[-2], c='m', s=100, marker='o', alpha=1.0)
                plt.scatter(self.X[-1], self.Treal[-1], c='m', s=50, marker='o', alpha=1.0)
            else:
                plt.scatter(self.X[-1], self.Treal[-1], c='m', s=50, marker='o', alpha=1.0)

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
