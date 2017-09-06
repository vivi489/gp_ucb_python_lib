# coding: utf-8
import itertools
import os
import random
import warnings
from operator import itemgetter

import numpy as np
import pandas as pd
import scipy
from gphypo import normalization
from gphypo.acquisition_func import UCB, EI, PI
from gphypo.point_info import PointInfoManager
from gphypo.transform_val import transform_click_val2real_val
from gphypo.util import mkdir_if_not_exist
from matplotlib import cm, gridspec
from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fmin_l_bfgs_b
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform
from sksparse.cholmod import cholesky


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
    def __init__(self, bo_param_list, environment, GAMMA=0.01, GAMMA0=0.01, GAMMA_Y=0.01, ALPHA=0.1, BETA=16,
                 burnin=False, is_edge_normalized=False, noise=True, n_early_stopping=None,
                 gt_available=False, normalize_output="zero_mean_unit_var", optimizer="fmin_l_bfgs_b",
                 update_hyperparam_func='pairwise_sampling', initial_k=1, initial_theta=1,
                 acquisition_func='ucb', n_stop_pairwise_sampling=np.inf, n_exp=1):

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

        self.bo_param_list = bo_param_list
        meshgrid = np.meshgrid(*bo_param_list)
        self.n_stop_pairwise_sampling = n_stop_pairwise_sampling
        self.GAMMA = GAMMA
        self.GAMMA0 = GAMMA0
        self.GAMMA_Y = GAMMA_Y
        self.ALPHA = ALPHA

        self.environment = environment
        self.optimizer = optimizer
        self.n_exp = n_exp

        self.does_pairwise_sampling = False
        assert update_hyperparam_func in [None, "pairwise_sampling", "simple_loglikelihood", "loglikelihood"]
        if update_hyperparam_func == 'pairwise_sampling':
            self.update_hyperparam = self.update_hyper_params_by_adj_idxes
            self.does_pairwise_sampling = True
        elif update_hyperparam_func == 'simple_loglikelihood':
            self.update_hyperparam = self.update_hyperparams_by_simple_loglikelihood
        elif update_hyperparam_func == 'loglikelihood':
            self.update_hyperparam = self.update_hyperparams_by_loglikelihood
        else:
            self.update_hyperparam = None

        self.k = initial_k
        self.theta = initial_theta

        assert normalize_output in [None, "zero_mean_unit_var", "zero_one"]
        self.normalize_output = normalize_output
        self.ndim = len(meshgrid)

        self.meshgrid = np.array(meshgrid)

        mesh_shape = self.meshgrid[0].shape
        self.ndim_list = mesh_shape[:2][::-1] + mesh_shape[2:]

        self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T

        self.X_grid2idx_dic = {tuple(x): i for i, x in enumerate(self.X_grid)}
        self.n_points = self.X_grid.shape[0]


        assert acquisition_func in ["ucb", "ei", "pi"]
        self.acquisition_func_name = acquisition_func
        if acquisition_func == 'ucb':
            param_dic = {'beta': BETA}
            self.acquisition_func = UCB(param_dic, d_size=self.X_grid.shape[0])
        elif acquisition_func == 'ei':
            param_dic = {'par': 0.01}
            self.acquisition_func = EI(param_dic)
        else:
            param_dic = {'par': 0.01}
            self.acquisition_func = PI(param_dic)

        self.gt_available = gt_available

        # Calculate ground truth (self.z)
        if self.ndim == 1 and gt_available:
            self.z = self.environment.sample(self.X_grid, get_ground_truth=True)
        elif self.ndim == 2 and gt_available:
            nrow, ncol = self.meshgrid.shape[1:]
            self.z = self.environment.sample(self.X_grid, get_ground_truth=True).reshape(nrow, ncol)
        elif self.ndim == 3 and gt_available:
            self.z = self.environment.sample(self.X_grid, get_ground_truth=True)
        else:
            self.z = None

        self.point_info_manager = PointInfoManager(self.X_grid, self.normalize_output)

        self.bestX = None
        self.bestT = -np.inf
        self.cnt_since_bestT = 0

        if n_early_stopping:
            self.n_early_stopping = n_early_stopping
        else:
            self.n_early_stopping = np.inf

        tau0 = - create_adjacency_mat(self.ndim_list)

        row_sum_list = - mat_flatten(tau0.sum(axis=0))
        self.diff_list = row_sum_list.max() - row_sum_list
        print("Edge Num: %s" % np.count_nonzero(self.diff_list))

        print(tau0.toarray())
        self.is_edge_normalized = is_edge_normalized

        # # This changes the edge weight. This seems to be not so effective(Result does not change)
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
        #     print(tau0.toarray())

        self.baseTau0 = tau0

        row_idxes, col_idxes, _ = scipy.sparse.find(self.baseTau0 != 0)
        self.adj_pair_list = [(r, c) for r, c in zip(row_idxes, col_idxes)]

        # Reload past results if exits
        if environment.reload:
            for key, row in environment.result_df.iterrows():
                x = row[environment.bo_param_names].as_matrix()
                t = float(row['output'])
                print(x, t, row.n_exp)
                n_exp = int(float(row.n_exp))
                if n_exp > 1:
                    n1 = t
                    n0 = n_exp - n1
                    t = transform_click_val2real_val(n0, n1)
                    if type(t) == list or type(t) == np.ndarray:
                        t = t[0]
                    self.point_info_manager.update(x, {t: n_exp})
                else:
                    self.point_info_manager.update(x, {t: 1})

            print(self.point_info_manager.get_T())
            print("Reloaded csv.")

            if self.update_hyperparam is not None:
                self.update_hyperparam()

        # Grid-based BurnIn
        if burnin and not environment.reload:
            burnin = 2 ** self.ndim
            quarter_point_list = [[bo_param[len(bo_param) // 4], bo_param[len(bo_param) // 4 * 3]] for bo_param in
                                  bo_param_list]

            for coodinate in itertools.product(*quarter_point_list):
                coodinate = np.array(coodinate)
                self.sample(coodinate)
                if update_hyperparam_func == 'pairwise_sampling':
                    adj_idx = self.get_pairwise_idx(self.X_grid2idx_dic[tuple(coodinate)])
                    self.sample(self.X_grid[adj_idx])

            print('%d burins has finised!' % burnin)

            X_seq, T_seq = self.point_info_manager.X_seq, self.point_info_manager.T_seq
            self.bestT = np.max(T_seq)
            self.bestX = X_seq[np.argmax(T_seq)]

            if self.update_hyperparam is not None:
                self.update_hyperparam()

            self.does_pairwise_sampling = False

        self.update()

    def calc_true_mean_std(self):
        assert self.gt_available == True
        print(self.X_grid)
        print(self.X_grid.shape)
        sampled_y = np.array(self.environment.sample(self.X_grid.T, get_ground_truth=True))

        return sampled_y.mean(), sampled_y.std()

    def calc_tau(self):
        gamma = self.GAMMA
        gamma_y = self.GAMMA_Y
        gamma0 = self.GAMMA0

        n_grid = self.point_info_manager.get_n_grid()
        sum_grid = self.point_info_manager.get_sum_grid()

        if self.is_edge_normalized:
            # corner will be treated as center nodes. This is explained as A -> A' in PPTX file.
            gamma_tilda = n_grid * gamma + gamma0 + gamma_y * self.diff_list.flatten()
        else:
            gamma_tilda = n_grid * gamma + gamma0  # Normal

        # print ('gamma_tilda: {}'.format(gamma_tilda) )
        mu_tilda = np.array([s * gamma + self.ALPHA * gamma0 for s in sum_grid]) / gamma_tilda

        tau1 = -scipy.sparse.diags(gamma_tilda)
        tau0 = self.baseTau0 * self.GAMMA_Y

        tau0 += scipy.sparse.diags(gamma_tilda - mat_flatten(tau0.sum(axis=0)))

        return tau0, -tau1, mu_tilda, gamma_tilda

    def get_adj_idxes(self, idx):
        return scipy.sparse.find(self.baseTau0[idx] != 0)[1]

    def get_pairwise_idx(self, idx):
        adj_idxes = scipy.sparse.find(self.baseTau0[idx] != 0)[1]
        return random.choice(adj_idxes)

    def learn(self):
        T = self.point_info_manager.get_T()
        grid_idx = np.argmax(self.acquisition_func.compute(self.mu, self.sigma, T))

        continue_flg = self.sample(self.X_grid[grid_idx], self.n_exp)

        if not continue_flg:
            return False

        if self.does_pairwise_sampling:
            adj_idx = self.get_pairwise_idx(grid_idx)
            continue_flg = self.sample(self.X_grid[adj_idx], self.n_exp)

            if not continue_flg:
                return False

        if self.point_info_manager.update_cnt > self.n_stop_pairwise_sampling:
            self.does_pairwise_sampling = False

        self.update()
        return True

    # TODO
    def learn_from_clicks(self, ratio_csv_fn='ratio.csv', n_total_exp=10000):

        ratio_df = pd.read_csv(ratio_csv_fn)

        for key, row in ratio_df.iterrows():
            x = row[self.environment.bo_param_names].as_matrix()
            t = float(row['output'])
            print(x, t, row.n_exp)
            n_exp = int(float(row.ratio) * n_total_exp)
            assert n_exp > 1
            n1 = t
            n0 = n_exp - n1
            t = transform_click_val2real_val(n0, n1)
            if type(t) == list or type(t) == np.ndarray:
                t = t[0]
            self.point_info_manager.update(x, {t: n_exp})

        print("Loaded csv.")

        self.update()
        return True

    def sample(self, x, n_exp=1):
        if n_exp > 2:
            n1 = self.environment.sample(x, n_exp=self.n_exp)
            n0 = self.n_exp - n1
            t = transform_click_val2real_val(n0, n1)
            if type(t) == list or type(t) == np.ndarray:
                t = t[0]

            self.point_info_manager.update(x, {t: self.n_exp})
        else:
            t = self.environment.sample(x)
            if type(t) == list or type(t) == np.ndarray:
                t = t[0]

            self.point_info_manager.update(x, {t: 1})

        if t <= self.bestT:
            self.cnt_since_bestT += 1
        else:
            self.bestT = t
            self.bestX = x
            self.cnt_since_bestT = 0

        if self.cnt_since_bestT > self.n_early_stopping:
            return False

        return True

    def update(self, n_start_opt_hyper_param=0):
        A, B, mu_tilda, gamma_tilda = self.calc_tau()

        # start = time.time()
        # cov = scipy.sparse.linalg.inv(A) # This is slow

        factor = cholesky(A)
        cov = scipy.sparse.csc_matrix(factor(np.eye(A.shape[0])))
        # end = time.time() - start
        # print("one calc: %s sec" % end)

        self.mu = mat_flatten(cov.dot(B).dot(mu_tilda))
        self.sigma = mat_flatten(np.sqrt(cov[np.diag_indices_from(cov)]))

        # Update hyperparamters
        if self.update_hyperparam is not None and self.point_info_manager.update_cnt > n_start_opt_hyper_param:
            self.update_hyperparam()

    def update_hyper_params_by_adj_idxes(self):
        T = self.point_info_manager.get_T()
        var_list = []

        for i, j in self.adj_pair_list:
            t_i, t_j = T[i], T[j]
            if (t_i is not None) and (t_j is not None):
                var_list.append((t_i - t_j) ** 2)

        var_list = np.unique(var_list)
        var = var_list.mean()

        self.GAMMA_Y = 1 / var / (self.ndim ** 2)

        self.GAMMA = self.GAMMA_Y * (2 * self.ndim)

        self.GAMMA0 = self.GAMMA * 0.01

        print("New GammaY: %s" % self.GAMMA_Y)
        print("New Gamma: %s" % self.GAMMA)

        self.ALPHA = np.mean(self.point_info_manager.get_T(excludes_none=True))
        print("New ALPHA: %s" % self.ALPHA)

    # TODO this method does not work well and caliculation is heavy...
    def update_hyperparams_by_simple_loglikelihood(self):
        mean_grid = self.point_info_manager.get_T()

        unobserbed_idxes = np.isnan(mean_grid)
        obserbed_idxes = (np.isnan(mean_grid) == False)

        y_hat = mean_grid[obserbed_idxes]

        n_obserbed = len(y_hat)

        new_X_grid = np.concatenate([self.normalized_X_grid[obserbed_idxes], self.normalized_X_grid[unobserbed_idxes]],
                                    axis=0)
        dist = pdist(new_X_grid, metric=adj_metric)  # TODO heavy calculation

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
        self.GAMMA = self.GAMMA_Y * (2 * self.ndim)
        self.GAMMA0 = self.GAMMA * 0.01

        if self.normalize_output is None:
            self.ALPHA = np.mean(self.point_info_manager.get_T(excludes_none=True))
            print("New ALPHA: %s" % self.ALPHA)

        print("New GammaY: %s" % self.GAMMA_Y)
        print("New Gamma: %s" % self.GAMMA)
        print("t: %s" % t)
        print('k: %s, theta: %s' % (self.k, self.theta))

        log_likelihood = (y_hat.dot(Lambda.dot(y_hat[:, np.newaxis])) - np.log(np.linalg.det(Lambda) + 0.00001))[0]
        print('log_likelihood: %s' % log_likelihood)

    def update_hyperparams_by_loglikelihood(self):
        # Update hyper-paramter of GMRF below
        def obj_func(eval_gradient=True):
            if eval_gradient:
                lml, grad = self.log_marginal_likelihood(eval_gradient=True)
                return -lml, -grad
            else:
                return -self.log_marginal_likelihood()

        # optima = [(self._constrained_optimization(obj_func, theta))] # Use when gradient can be calculated
        optima = [(self._constrained_optimization(lambda x: obj_func(x, eval_gradient=False),
                                                  gradExist=False))]  # Use when gradient cannot be calculated

        # Select result from run with minimal (negative) log-marginal likelihood
        lml_values = list(map(itemgetter(1), optima))

        self.GAMMA, self.GAMMA_Y = optima[np.argmin(lml_values)][0]
        self.GAMMA0 = 0.01 * self.GAMMA

        print("New GammaY: %s" % self.GAMMA_Y)
        print("New Gamma: %s" % self.GAMMA)

        self.ALPHA = self.point_info_manager.get_real_T(excludes_none=True).mean()
        print("New ALPHA: %s" % self.ALPHA)

        self.log_marginal_likelihood_value_ = -np.min(lml_values)
        print("log_marginal_likelihood: %s" % self.log_marginal_likelihood_value_)

    # TODO this method does not work well...
    def log_marginal_likelihood(self, eval_gradient=False):
        """Returns log-marginal likelihood of theta for training data.(theta is hyper-paramters)

        Parameters
        ----------
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
        A, B, mu_tilda, gamma_tilda, r_grid, n_grid = self.calc_tau(return_all=True)

        gamma = self.GAMMA
        gamma_y = self.GAMMA_Y

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

    def _constrained_optimization(self, obj_func, bounds=[(1e-5, 1e5), (1e-5, 1e5)], gradExist=True):
        initial_theta = [self.GAMMA, self.GAMMA_Y]
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

    def save_mu_sigma_csv(self, outfn="mu_sigma.csv", point_info_fn='point_info.csv'):
        # df = pd.DataFrame(self.X_grid, columns=self.environment.bo_param_names)
        # df['mu'] = self.mu
        # df['sigma'] = self.sigma
        # df.to_csv(outfn, index=False)

        df = pd.DataFrame({
            'mu': self.mu,
            'sigma': self.sigma
        })
        df.index.name = 'point_id'
        df.to_csv(outfn)

        point_info_df = pd.DataFrame(self.X_grid, columns=self.environment.bo_param_names)
        point_info_df.index.name = 'point_id'
        point_info_df.to_csv(point_info_fn)

        print('%s was saved!' % outfn)

    def plot(self, output_dir):
        def plot3d():
            fig = plt.figure()
            ax = Axes3D(fig)

            X_seq, T_seq = self.point_info_manager.X_seq, self.point_info_manager.T_seq
            if self.gt_available:
                c_true, lower, upper = normalization.zero_one_normalization(self.z)
                c_true = cm.bwr(c_true * 255)
                ax.scatter([x[0] for x in self.X_grid.astype(float)], [x[1] for x in self.X_grid.astype(float)],
                           [x[2] for x in self.X_grid.astype(float)],
                           c=c_true, marker='o',
                           alpha=0.5, s=5)
                c = cm.bwr(normalization.zero_one_normalization(T_seq, self.z.min(), self.z.max())[0] * 255)

            else:
                c = cm.bwr(normalization.zero_one_normalization(T_seq)[0] * 255)

            ax.scatter([x[0] for x in X_seq], [x[1] for x in X_seq], [x[2] for x in X_seq], c='y', marker='o',
                       alpha=0.5)

            if self.does_pairwise_sampling:
                ax.scatter(X_seq[-1][0], X_seq[-1][1], X_seq[-1][2], c='m', s=50, marker='o', alpha=1.0)
                ax.scatter(X_seq[-2][0], X_seq[-2][1], X_seq[-2][2], c='m', s=100, marker='o', alpha=1.0)
            else:
                ax.scatter(X_seq[-1][0], X_seq[-1][1], X_seq[-1][2], c='m', s=50, marker='o', alpha=1.0)

        def plot2d():
            acq_score = self.acquisition_func.compute(self.mu, self.sigma, self.point_info_manager.get_T())
            mu = self.mu.flatten()
            X, T = self.point_info_manager.get_observed_XT_pair()
            X_seq, T_seq = self.point_info_manager.X_seq, self.point_info_manager.T_seq

            if self.normalize_output:
                mu = self.point_info_manager.get_unnormalized_value_list(mu)
                acq_score = self.point_info_manager.get_unnormalized_value_list(acq_score)

            if self.acquisition_func_name == 'ucb':
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.plot_wireframe(self.meshgrid[0].astype(float), self.meshgrid[1].astype(float),
                                  mu.reshape(self.meshgrid[0].shape), alpha=0.5,
                                  color='g')

                ax.plot_wireframe(self.meshgrid[0].astype(float), self.meshgrid[1].astype(float),
                                  acq_score.reshape(self.meshgrid[0].shape), alpha=0.5, color='y')

                if self.gt_available:
                    ax.plot_wireframe(self.meshgrid[0].astype(float), self.meshgrid[1].astype(float), self.z, alpha=0.3,
                                      color='b')

                ax.scatter([x[0] for x in X], [x[1] for x in X], T, c='r', marker='o', alpha=0.5)

                if self.does_pairwise_sampling:
                    ax.scatter(X_seq[-2][0], X_seq[-2][1], T_seq[-2], c='m', s=100, marker='o', alpha=1.0)

                ax.scatter(X_seq[-1][0], X_seq[-1][1], T_seq[-1], c='m', s=50, marker='o', alpha=1.0)
            else:
                fig = plt.figure(figsize=(6, 10))
                fig.subplots_adjust(right=0.8)

                upper = fig.add_subplot(2, 1, 1, projection='3d')
                lower = fig.add_subplot(2, 1, 2, projection='3d')

                upper.plot_wireframe(self.meshgrid[0].astype(float), self.meshgrid[1].astype(float),
                                     mu.reshape(self.meshgrid[0].shape), alpha=0.5,
                                     color='g')

                lower.plot_wireframe(self.meshgrid[0].astype(float), self.meshgrid[1].astype(float),
                                     acq_score.reshape(self.meshgrid[0].shape), alpha=0.5, color='y')

                if self.gt_available:
                    upper.plot_wireframe(self.meshgrid[0].astype(float), self.meshgrid[1].astype(float), self.z,
                                         alpha=0.3,
                                         color='b')

                upper.scatter([x[0] for x in X], [x[1] for x in X], T, c='r', marker='o', alpha=0.5)

                if self.does_pairwise_sampling:
                    upper.scatter(X_seq[-2][0], X_seq[-2][1], T_seq[-2], c='m', s=100, marker='o', alpha=1.0)

                upper.scatter(X_seq[-1][0], X_seq[-1][1], T_seq[-1], c='m', s=50, marker='o', alpha=1.0)

                # upper.set_zlabel('f(x)', fontdict={'size': 18})
                # lower.set_zlabel(self.acquisition_func_name.upper(), fontdict={'size': 18})

        def plot1d():
            acq_score = self.acquisition_func.compute(self.mu, self.sigma, self.point_info_manager.get_T())

            mu = self.mu.flatten()
            if self.normalize_output:
                mu = self.point_info_manager.get_unnormalized_value_list(mu)
                acq_score = self.point_info_manager.get_unnormalized_value_list(acq_score)

            X, T = self.point_info_manager.get_observed_XT_pair()
            X_seq, T_seq = self.point_info_manager.X_seq, self.point_info_manager.T_seq

            if self.acquisition_func_name == 'ucb':
                plt.plot(self.meshgrid[0].astype(float), mu, color='g')
                plt.plot(self.meshgrid[0].astype(float), acq_score, color='y')
                plt.plot(self.meshgrid[0], self.z, alpha=0.3, color='b')
                plt.scatter(X, T, c='r', s=10, marker='o', alpha=1.0)
                if self.does_pairwise_sampling:
                    plt.scatter(X_seq[-2], T_seq[-2], c='m', s=100, marker='o', alpha=1.0)
                plt.scatter(X_seq[-1], T_seq[-1], c='m', s=50, marker='o', alpha=1.0)

            else:
                fig = plt.figure()
                fig.subplots_adjust(left=0.15)

                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
                upper = plt.subplot(gs[0])
                lower = plt.subplot(gs[1])

                upper.plot(self.meshgrid[0].astype(float), mu, color='g')
                lower.plot(self.meshgrid[0].astype(float), acq_score, color='y')

                if self.gt_available:
                    upper.plot(self.meshgrid[0], self.z, alpha=0.3, color='b')

                upper.scatter(X, T, c='r', s=10, marker='o', alpha=1.0)

                if self.does_pairwise_sampling:
                    upper.scatter(X_seq[-2], T_seq[-2], c='m', s=100, marker='o', alpha=1.0)

                upper.scatter(X_seq[-1], T_seq[-1], c='m', s=50, marker='o', alpha=1.0)

                upper.set_ylabel('f(x)', fontdict={'size': 18})
                lower.set_ylabel(self.acquisition_func_name.upper(), fontdict={'size': 18})

        if self.ndim in [1, 2, 3]:
            exec("plot{}d()".format(self.ndim))
            out_fn = os.path.join(output_dir, 'res_%04d.png' % self.point_info_manager.update_cnt)
            mkdir_if_not_exist(output_dir)
            plt.savefig(out_fn, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

        else:
            print("Sorry... Plotting only supports 1 dim or 2 dim.")
