# coding: utf-8
import os
import random
import warnings

import numpy as np
import scipy
from matplotlib import cm
from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fmin_l_bfgs_b
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from sksparse.cholmod import cholesky

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


def create_adjacency_mat_using_pdist(dim_list):
    meshgrid = np.array(np.meshgrid(*[np.arange(ndim) for ndim in dim_list]))
    X_grid = meshgrid.reshape(meshgrid.shape[0], -1).T
    dist = pdist(X_grid, adj_metric)
    return squareform(dist)


def create_adjacency_mat(dim_list, calc_sparse=True):
    K_func = lambda x: scipy.sparse.csc_matrix(x) if calc_sparse else x

    xp = scipy.sparse if calc_sparse else np
    adj1d_list = [create_adjacency_mat_using_pdist([ndim]) for ndim in dim_list]
    if len(dim_list) == 1:
        return K_func(adj1d_list[0])

    K = xp.kron(adj1d_list[1], xp.eye(dim_list[0])) + xp.kron(xp.eye(dim_list[1]), adj1d_list[0])

    prod = dim_list[0] * dim_list[1]

    for i in range(2, len(dim_list)):
        K = xp.kron(K, xp.eye(dim_list[i])) + xp.kron(xp.eye(prod), adj1d_list[i])
        prod *= dim_list[i]

    return K_func(K)


class EGMRF_UCB(object):
    def __init__(self, meshgrid, environment, GAMMA=0.01, GAMMA0=0.01, GAMMA_Y=0.01, ALPHA=0.1, BETA=16,
                 burnin=0, is_edge_normalized=False, noise=True, n_early_stopping=None,
                 gt_available=False, normalize_output="zero_mean_unit_var", optimizer="fmin_l_bfgs_b",
                 update_hyperparam=False,
                 update_only_gamma_y=False, initial_k=1, initial_theta=1, does_pairwise_sampling=False):
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

        assert normalize_output in [None, "zero_mean_unit_var", "zero_one"]
        self.normalize_output = normalize_output
        if normalize_output == "zero_mean_unit_var":
            self.t_mean = 0
            self.t_std = 1
        elif normalize_output == "zero_one":
            self.t_lower = 0
            self.t_upper = 1

        self.ndim = len(meshgrid)

        self.meshgrid = np.array(meshgrid)

        mesh_shape = self.meshgrid[0].shape
        self.ndim_list = mesh_shape[:2][::-1] + mesh_shape[2:]

        self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T
        self.normalized_X_grid = create_normalized_X_grid(self.meshgrid)

        self.mu = np.array([0. for _ in range(self.X_grid.shape[0])])
        self.sigma = np.array([0.5 for _ in range(self.X_grid.shape[0])])

        self.gt_available = gt_available
        self.learn_cnt = 0

        # Calculate ground truth (self.z)
        if self.ndim == 1 and gt_available:
            self.z = self.environment.sample(self.X_grid.flatten(), get_ground_truth=True)
        elif self.ndim == 2 and gt_available:
            nrow, ncol = self.meshgrid.shape[1:]
            self.z = self.environment.sample(self.X_grid.T, get_ground_truth=True).reshape(nrow, ncol)
        elif self.ndim == 3 and gt_available:
            self.z = self.environment.sample(self.X_grid.T, get_ground_truth=True)
        else:
            self.z = None

        self.X = []
        self.T = np.array([])
        self.Treal = np.array([])

        self.bestX = None
        self.bestT = -np.inf
        self.cnt_since_bestT = 0
        if n_early_stopping:
            self.n_early_stopping = n_early_stopping
        else:
            self.n_early_stopping = np.inf

        self.does_pairwise_sampling = does_pairwise_sampling

        if environment.reload:
            X = environment.result_df[environment.gp_param_names].as_matrix()
            T = environment.result_df['output'].as_matrix()

            self.X = [np.array(x) for x in X.tolist()]
            self.T = np.array(T)
            print("Reloaded csv.")

        tau0 = - create_adjacency_mat(self.ndim_list)

        # edge_idxes = np.count_nonzero(tau0, axis=1) < self.ndim * 2
        # print('edge num is ' + str(edge_idxes.sum()))

        # row_sum_list = -tau0.sum(axis=1, keepdims=True)
        row_sum_list = - mat_flatten(tau0.sum(axis=0))
        self.diff_list = row_sum_list.max() - row_sum_list
        print(np.count_nonzero(self.diff_list))

        print(tau0.toarray())
        self.is_edge_normalized = is_edge_normalized

        # if is_edge_normalized:
        #     weight_arr = np.matrix(self.ndim * 2 / row_sum_list)
        #     weight_mat = weight_arr.T.dot(weight_arr)
        #     # weight_mat = np.sqrt(weight_arr.T.dot(weight_arr))
        #
        #     if type(tau0) == scipy.sparse.csc.csc_matrix:
        #         tau0 = tau0.multiply(weight_mat)
        #     else:
        #         tau0 *= weight_mat  # O.K. for dense mat but N.G. for sparse mat
        #
        #     tau0 = scipy.sparse.csc_matrix(tau0)

        self.baseTau0 = tau0
        print(tau0.toarray())

        if burnin > 0 and not environment.reload:
            if self.does_pairwise_sampling:
                for _ in range(burnin):
                    n_points = self.X_grid.shape[0]
                    rand_idx = np.random.randint(n_points)
                    self.sample(self.X_grid[rand_idx])
                    adj_idx = self.get_pairwise_idx(rand_idx)
                    self.sample(self.X_grid[adj_idx])

            else:
                for _ in range(burnin):
                    n_points = self.X_grid.shape[0]
                    rand_idx = np.random.randint(n_points)
                    x = self.X_grid[rand_idx]
                    self.sample(x)

            print('%d burins has finised!' % burnin)
            # print(self.X, self.T)
            self.bestX = self.X[np.argmax(self.Treal)]
            self.bestT = self.Treal.max()

            if self.update_hyperparam:
                self.ALPHA = self.T.mean()
                diff = self.T.std()
                print("mean: %f, std: %f" % (self.T.mean(), self.T.std()))

                # TODO: this is not correct
                self.GAMMA_Y = 2 / ((diff) ** 2)
                self.GAMMA = 2 * self.ndim * self.GAMMA_Y  # 0.01 200  # weight of obserbed data
                self.GAMMA0 = 0.01 * self.GAMMA  # 0.01  # weight of alpha
                print("GAMMA_Y: %f, GAMMA: %f, GAMMA0: %f" % (self.GAMMA_Y, self.GAMMA, self.GAMMA0))

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
        gamma0 = 0.01 * gamma  # TODO hard coding

        r_grid = self.get_r_grid()
        n_grid = np.array([len(r) for r in r_grid])

        if self.is_edge_normalized:
            gamma_tilda = n_grid * gamma + gamma0 + gamma_y * self.diff_list.flatten()  # corner will be treated as center nodes
        else:
            gamma_tilda = n_grid * gamma + gamma0  # Normal

        mu_tilda = np.array([r.sum() * gamma + self.ALPHA * gamma0 for r in r_grid]) / gamma_tilda

        tau1 = -scipy.sparse.diags(gamma_tilda)
        tau0 = self.baseTau0 * self.GAMMA_Y

        tau0 += scipy.sparse.diags(gamma_tilda - mat_flatten(tau0.sum(axis=0)))

        if return_all:
            return tau0, -tau1, mu_tilda, gamma_tilda, r_grid, n_grid

        return tau0, -tau1, mu_tilda, gamma_tilda

    def update(self, n_start_opt_hyper_param=0):
        theta = [self.GAMMA, self.GAMMA_Y]
        A, B, mu_tilda, gamma_tilda = self.calc_tau(theta)

        # start = time.time()
        # cov = scipy.sparse.linalg.inv(A) # This is slow

        factor = cholesky(A)
        cov = scipy.sparse.csc_matrix(factor(np.eye(A.shape[0])))
        # end = time.time() - start
        # print("one calc: %s sec" % end)

        self.mu = mat_flatten(cov.dot(B).dot(mu_tilda))
        self.sigma = mat_flatten(np.sqrt(cov[np.diag_indices_from(cov)]))

        if self.update_only_gamma_y and len(self.T) > n_start_opt_hyper_param:
            self.update_gammaY()

            # if self.update_hyperparam and len(self.T) > n_start_opt_hyper_param:
            #     # Update hyper-paramter of GMRF below
            #     def obj_func(theta, eval_gradient=True):
            #         if eval_gradient:
            #             lml, grad = self.log_marginal_likelihood(
            #                 theta, eval_gradient=True)
            #             return -lml, -grad
            #         else:
            #             return -self.log_marginal_likelihood(theta)
            #
            #     # optima = [(self._constrained_optimization(obj_func, theta))] # Use when gradient can be calculated
            #     optima = [(self._constrained_optimization(lambda x: obj_func(x, eval_gradient=False)
            #                                               , theta,
            #                                               gradExist=False))]  # Use when gradient cannot be calculated
            #
            #     # Select result from run with minimal (negative) log-marginal likelihood
            #     lml_values = list(map(itemgetter(1), optima))
            #
            #     self.GAMMA, self.GAMMA_Y = optima[np.argmin(lml_values)][0]
            #
            #     self.GAMMA0 = 0.001 * self.GAMMA  # TODO Hard coding
            #
            #     print("GAMMA: %s, GAMMA_Y: %s" % (self.GAMMA, self.GAMMA_Y))
            #
            #     self.log_marginal_likelihood_value_ = -np.min(lml_values)
            #     print("log_marginal_likelihood: %s" % self.log_marginal_likelihood_value_)

    def get_ei(self, par=0.00001):
        if len(self.T) == 0:
            z = (1 - self.mu - par) / self.sigma
        else:
            # z = (max(self.T) - self.mu - par) / self.sigma
            z = (self.mu - max(self.T) - par) / self.sigma
        # z = (self.mu - eta - par) / self.sigma
        f = self.sigma * (z * norm.cdf(z) + norm.pdf(z))
        # print (f)
        return f

    def get_pi(self, par=0.00001):
        inc_val = 0
        if len(self.T) > 0:
            inc_val = max(self.T)

        z = - (inc_val - self.mu - par) / self.sigma
        return norm.cdf(z)

    def get_ucb(self):
        # def get_beta(self):
        #     delta = .5 # in (0, 1)
        #
        #     return 2 * np.log(d_size * (t * t) * (Math.PI * Math.PI) / (6 * delta));
        # self.BETA *= 0.99

        # d_size = self.X_grid.shape[0]
        # t = self.learn_cnt + 1
        # delta = 0.9  # must be in (0, 1)
        # self.BETA = 2 * np.log(d_size * ((t * np.pi) ** 2) / (6 * delta))
        print("New BETA: %s" % self.BETA)
        return self.mu + self.sigma * np.sqrt(self.BETA)

    def get_pairwise_idx(self, idx):
        # adj_idxes = np.where(self.baseTau0[grid_idx] != 0)[0]
        adj_idxes = scipy.sparse.find(self.baseTau0[idx] != 0)[1]
        print(adj_idxes)
        return random.choice(adj_idxes)

    def learn(self):
        grid_idx = np.argmax(self.get_ucb())
        # grid_idx = np.argmax(self.get_ei())
        # grid_idx = np.argmax(self.get_pi())

        obserbed_val = self.sample(self.X_grid[grid_idx])
        if obserbed_val is None:
            return False

        if self.does_pairwise_sampling:
            adj_idx = self.get_pairwise_idx(grid_idx)
            obserbed_val2 = self.sample(self.X_grid[adj_idx])

            if obserbed_val2 is None:
                return False

        # if len(self.X) > 200:  # TODO Hard coding
        #     self.update_only_gamma_y = False
        #     self.does_pairwise_sampling = False

        self.update()
        self.learn_cnt += 1
        return True

    def learn_from_click(self, n_exp=10):
        grid_idx = self.argmax_ucb()
        continue_flg = self.sample_from_click(self.X_grid[grid_idx], n_exp)
        if not continue_flg:
            return False

        self.update()
        self.learn_cnt += 1
        return True

    def sample_from_click(self, x, n_exp):
        n1 = self.environment.sample(x, n_exp=n_exp)
        n0 = n_exp - n1
        t = transform_click_val2real_val(n0, n1)

        # TODO change the structure self.X and self.Treal (should not use list because of the problem of consuming-memory when the obserbation is many 01 values)
        for _ in range(n_exp):
            self.X.append(x)
            self.Treal = np.append(self.Treal, t)

        if self.normalize_output == "zero_mean_unit_var":
            # Normalize output to have zero mean and unit standard deviation
            # self.T, self.t_mean, self.t_std = normalization.zero_mean_unit_var_normalization(self.Treal)
            tmp, self.t_mean, self.t_std = normalization.zero_mean_unit_var_normalization(np.unique(self.Treal))
            self.T = (self.Treal - self.t_mean) / self.t_std


        elif self.normalize_output == "zero_one":
            self.T, self.t_lower, self.t_upper = normalization.zero_one_normalization(self.Treal)
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
        # print (self.Treal)

        if self.normalize_output == "zero_mean_unit_var":
            # Normalize output to have zero mean and unit standard deviation
            # self.T, self.t_mean, self.t_std = normalization.zero_mean_unit_var_normalization(self.Treal) ## This is a mistake
            tmp, self.t_mean, self.t_std = normalization.zero_mean_unit_var_normalization(np.unique(self.Treal))
            self.T = (self.Treal - self.t_mean) / self.t_std

        elif self.normalize_output == "zero_one":
            self.T, self.t_lower, self.t_upper = normalization.zero_one_normalization(self.Treal)
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

    def update_hyper_params_by_does_pairwise_sampling(self):
        # paiwise_var_list = np.array([(t2 - t1) ** 2 for t1, t2 in zip(self.T[::2], self.T[1::2])]) ## This is a mistake
        paiwise_var_list = np.unique(np.array([(t2 - t1) ** 2 for t1, t2 in zip(self.T[::2], self.T[1::2])]))

        # paiwise_var_list = np.append(paiwise_var_list, [3]*self.ndim)
        # print (paiwise_var_list)

        print("len(paiwise_var_list) = %s" % len(paiwise_var_list))

        var = paiwise_var_list.mean()

        # self.GAMMA_Y = 1 / var / (self.ndim ** 3)
        self.GAMMA_Y = 1 / var / (self.ndim ** 2)
        self.GAMMA = self.GAMMA_Y * (2 * self.ndim)
        self.GAMMA0 = self.GAMMA * 0.01

        print("New GammaY: %s" % self.GAMMA_Y)
        print("New Gamma: %s" % self.GAMMA)

        # self.ALPHA = np.unique(self.T).mean()
        # print("New ALPHA: %s" % self.ALPHA)

    # TODO this method does not work well...
    def update_gammaY(self):

        if self.does_pairwise_sampling:
            self.update_hyper_params_by_does_pairwise_sampling()
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
        def plot3d():
            fig = plt.figure()
            ax = Axes3D(fig)

            if self.gt_available:
                c_true, lower, upper = normalization.zero_one_normalization(self.z)
                c_true = cm.bwr(c_true * 255)
                ax.scatter([x[0] for x in self.X_grid], [x[1] for x in self.X_grid], [x[2] for x in self.X_grid],
                           c=c_true, marker='o',
                           alpha=0.5, s=5)

                c = cm.bwr(normalization.zero_one_normalization(self.Treal, self.z.min(), self.z.max())[0] * 255)
            else:
                c = cm.bwr(normalization.zero_one_normalization(self.Treal)[0] * 255)

            ax.scatter([x[0] for x in self.X], [x[1] for x in self.X], [x[2] for x in self.X], c='y', marker='o',
                       alpha=0.5)

            if self.does_pairwise_sampling:
                ax.scatter(self.X[-1][0], self.X[-1][1], self.X[-1][2], c='m', s=50, marker='o', alpha=1.0)
                ax.scatter(self.X[-2][0], self.X[-2][1], self.X[-2][2], c='m', s=100, marker='o', alpha=1.0)
            else:
                ax.scatter(self.X[-1][0], self.X[-1][1], self.X[-1][2], c='m', s=50, marker='o', alpha=1.0)

        def plot2d():

            fig = plt.figure()
            ax = Axes3D(fig)
            ucb_score = self.mu + self.sigma * np.sqrt(self.BETA)

            if self.normalize_output == "zero_mean_unit_var":
                unnormalized_mu = normalization.zero_mean_unit_var_unnormalization(self.mu.flatten(), self.t_mean,
                                                                                   self.t_std)
                ucb_score = normalization.zero_mean_unit_var_unnormalization(ucb_score, self.t_mean, self.t_std)
                ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                                  unnormalized_mu.reshape(self.meshgrid[0].shape), alpha=0.5, color='g')
            elif self.normalize_output == "zero_one":
                unnormalized_mu = normalization.zero_one_unnormalization(self.mu.flatten(), self.t_lower, self.t_upper)
                ucb_score = normalization.zero_one_unnormalization(ucb_score, self.t_lower, self.t_upper)
                ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                                  unnormalized_mu.reshape(self.meshgrid[0].shape), alpha=0.5, color='g')
            else:
                ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                                  self.mu.reshape(self.meshgrid[0].shape), alpha=0.5, color='g')

            ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                              ucb_score.reshape(self.meshgrid[0].shape), alpha=0.5, color='y')

            if self.gt_available:
                ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1], self.z, alpha=0.3, color='b')

            ax.scatter([x[0] for x in self.X], [x[1] for x in self.X], self.Treal, c='r', marker='o', alpha=0.5)

            if self.does_pairwise_sampling:
                ax.scatter(self.X[-1][0], self.X[-1][1], self.Treal[-1], c='m', s=50, marker='o', alpha=1.0)
                ax.scatter(self.X[-2][0], self.X[-2][1], self.Treal[-2], c='m', s=100, marker='o', alpha=1.0)
            else:
                ax.scatter(self.X[-1][0], self.X[-1][1], self.Treal[-1], c='m', s=50, marker='o', alpha=1.0)

        def plot1d():
            ucb_score = self.get_ucb()
            ei_score = self.get_ei()
            if self.normalize_output == "zero_mean_unit_var":
                unnormalized_mu = normalization.zero_mean_unit_var_unnormalization(self.mu.flatten(), self.t_mean,
                                                                                   self.t_std)
                ucb_score = normalization.zero_mean_unit_var_unnormalization(ucb_score, self.t_mean, self.t_std)
                ei_score = normalization.zero_mean_unit_var_unnormalization(ei_score, self.t_mean, self.t_std)
                plt.plot(self.meshgrid[0], unnormalized_mu, color='g')

            elif self.normalize_output == "zero_one":
                unnormalized_mu = normalization.zero_one_unnormalization(self.mu.flatten(), self.t_lower, self.t_upper)
                ucb_score = normalization.zero_one_unnormalization(ucb_score, self.t_lower, self.t_upper)
                ei_score = normalization.zero_one_unnormalization(ei_score, self.t_lower, self.t_upper)
                plt.plot(self.meshgrid[0], unnormalized_mu, color='g')
            else:
                plt.plot(self.meshgrid[0], self.mu.flatten(), color='g')

            plt.plot(self.meshgrid[0], ucb_score.reshape(self.meshgrid[0].shape), color='y')
            # plt.plot(self.meshgrid[0], ei_score.reshape(self.meshgrid[0].shape), color='c')

            if self.gt_available:
                plt.plot(self.meshgrid[0], self.z, alpha=0.3, color='b')

            plt.scatter(self.X, self.Treal, c='r', s=10, marker='o', alpha=1.0)

            if self.does_pairwise_sampling:
                plt.scatter(self.X[-2], self.Treal[-2], c='m', s=100, marker='o', alpha=1.0)
                plt.scatter(self.X[-1], self.Treal[-1], c='m', s=50, marker='o', alpha=1.0)
            else:
                plt.scatter(self.X[-1], self.Treal[-1], c='m', s=50, marker='o', alpha=1.0)

        if self.ndim in [1, 2, 3]:
            exec("plot{}d()".format(self.ndim))
            out_fn = os.path.join(output_dir, 'res_%04d.png' % len(self.X))
            mkdir_if_not_exist(output_dir)

            plt.savefig(out_fn)
            plt.close()
            return


        else:
            print("Sorry... Plotting only supports 1 dim or 2 dim.")
            return
