import itertools
import random
import warnings
from operator import itemgetter

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import fmin_l_bfgs_b
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform
from sksparse.cholmod import cholesky

from gphypo.base_bo import BaseBO


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


class GMRF_BO(BaseBO):
    def __init__(self, bo_param_list, environment,
                 burnin=False, is_edge_normalized=False, n_early_stopping=None,
                 gt_available=False, normalize_output="zero_mean_unit_var", optimizer="fmin_l_bfgs_b",
                 update_hyperparam_func='pairwise_sampling',
                 acquisition_func='ucb', acquisition_param_dic={'beta': 5, 'pi': 0.01}, n_stop_pairwise_sampling=np.inf,
                 GAMMA=0.01, GAMMA0=0.01, GAMMA_Y=0.01, ALPHA=0.1, initial_k=1, initial_theta=1, n_ctr=None):

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

        super(GMRF_BO, self).__init__(bo_param_list, environment,
                                      n_early_stopping=n_early_stopping,
                                      gt_available=gt_available, normalize_output=normalize_output,
                                      acquisition_func=acquisition_func, acquisition_param_dic=acquisition_param_dic,
                                      optimizer=optimizer, n_ctr=n_ctr)

        self.n_stop_pairwise_sampling = n_stop_pairwise_sampling
        self.GAMMA = GAMMA
        self.GAMMA0 = GAMMA0
        self.GAMMA_Y = GAMMA_Y
        self.ALPHA = ALPHA

        self.does_pairwise_sampling = False
        assert update_hyperparam_func in [None, "pairwise_sampling", "simple_loglikelihood", "loglikelihood"] #TODO "simple_loglikelihood", "loglikelihood" does not work (2017/9/7)
        if update_hyperparam_func == 'pairwise_sampling':
            self.update_hyperparam = self.update_hyper_params_by_adj_idxes
            self.does_pairwise_sampling = True
        elif update_hyperparam_func == 'simple_loglikelihood':
            self.update_hyperparam = self.update_hyperparams_by_simple_loglikelihood
            self.normalized_X_grid = create_normalized_X_grid(self.meshgrid)
        elif update_hyperparam_func == 'loglikelihood':
            self.update_hyperparam = self.update_hyperparams_by_loglikelihood
        else:
            self.update_hyperparam = None

        tau0 = - create_adjacency_mat(self.ndim_list)
        row_sum_list = - mat_flatten(tau0.sum(axis=0))
        self.diff_list = row_sum_list.max() - row_sum_list
        #print("Edge Num: %s" % np.count_nonzero(self.diff_list))
        #print(tau0.toarray())
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
        
        # Grid-based BurnIn
        if burnin and not environment.reload:
            if self.n_ctr:
                self.do_all_point_burnin()
            else:
                self.do_grid_based_burnin()

        if environment.reload or burnin:
            if self.update_hyperparam is not None:
                self.update_hyperparam()
        self.update()
        self.total_clicked_ratio_list = []
        
        
    def do_all_point_burnin(self):
        n_sample_per_point = int(self.n_ctr / self.n_points)
        for i in range(self.n_points):
            self.sample(self.X_grid[i], n_sample_per_point)


    def do_grid_based_burnin(self):
        burnin = 2 ** self.ndim
        quarter_point_list = [[bo_param[len(bo_param) // 4], bo_param[len(bo_param) // 4 * 3]] for bo_param in
                              self.bo_param_list]
        #print("quarter_point_list:", quarter_point_list)
        for coodinate in itertools.product(*quarter_point_list):
            coodinate = np.array(coodinate)
            self.sample(coodinate, self.n_ctr)
            if self.does_pairwise_sampling:
                adj_idx = self.get_pairwise_idx(self.X_grid2idx_dic[tuple(coodinate)])
                self.sample(self.X_grid[adj_idx], self.n_ctr)

        X_seq, T_seq = self.point_info_manager.X_seq, self.point_info_manager.T_seq
        self.bestT = np.max(T_seq)
        self.bestX = X_seq[np.argmax(T_seq)]

        self.does_pairwise_sampling = False
        #print('%d burins has finised!' % burnin)

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

        #print ('gamma_tilda: {}'.format(gamma_tilda) )
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
        T = self.point_info_manager.get_T(excludes_none=True) # T: the means of all the points; either normalized or not
        grid_idx = np.argmax(self.acquisition_func.compute(self.mu, self.sigma, T)) # this line is crucial
        continue_flg = self.sample(self.X_grid[grid_idx]) # self.sample alters the optimizer's point info manager which contains all the points
        if not continue_flg:
            return False
        #print("does_pairwise_sampling=", self.does_pairwise_sampling)
        if self.does_pairwise_sampling:
            adj_idx = self.get_pairwise_idx(grid_idx)
            continue_flg = self.sample(self.X_grid[adj_idx])
            if not continue_flg:
                return False

        if self.point_info_manager.update_cnt > self.n_stop_pairwise_sampling:
            self.does_pairwise_sampling = False
        self.update()
        return True

    def learn_from_clicks(self, mu2ratio_dir=None, mu_sigma_csv_path=None,
                          ratio_csv_out_path=None):
        if self.acquisition_func.name == 'greedy':
            #print(self.acquisition_func.name)
            T = self.point_info_manager.get_T(excludes_none=True)
            #print("mu=", self.mu, "sigma=", self.sigma)
            clickProbDistribution = self.acquisition_func.compute(self.mu, self.sigma, T)
            clickProbDistribution -= clickProbDistribution.min()
            clickProbDistribution /= clickProbDistribution.sum()
            #print("clickProbDistribution", clickProbDistribution)
            df_ratio = pd.DataFrame(list(zip(range(len(self.mu)), clickProbDistribution)))
            #print("df_ratio\n", df_ratio)
            df_ratio.to_csv(ratio_csv_out_path, header=False, index=False)
        elif self.acquisition_func.name == 'ts':
            self.call_mu2ratio(mu2ratio_dir=mu2ratio_dir, mu_sigma_csv_path=mu_sigma_csv_path, ratio_csv_out_path=ratio_csv_out_path) #generate a ratio csv
        else:
            T = self.point_info_manager.get_T(excludes_none=True)
            #print("mu=", self.mu, "sigma=", self.sigma)
            clickProbDistribution = self.acquisition_func.compute(self.mu, self.sigma, T)
            index_max = np.argmax(clickProbDistribution)
            clickProbDistribution = np.ones(len(clickProbDistribution))
            clickProbDistribution[index_max] = self.n_ctr / 10.0
            clickProbDistribution /= clickProbDistribution.sum()
            df_ratio = pd.DataFrame(list(zip(range(len(self.mu)), clickProbDistribution)))
            #print("df_ratio\n", df_ratio)
            df_ratio.to_csv(ratio_csv_out_path, header=False, index=False)
            
        self.sample_using_ratio_csv(ratio_csv_out_path)
        #print("point_info_manager.T_mean, point_info_manager.T_std", self.point_info_manager.T_mean, self.point_info_manager.T_std)
        #print("before update mu=", self.mu, "sigma=", self.sigma)
        self.update()
        #print("after update mu=", self.mu, "sigma=", self.sigma)
        self.save_mu_sigma_csv(mu_sigma_csv_path)
        n_total_clicked = self.environment.result_df.output.values[-self.n_points:].astype(np.float64).sum()
        total_clicked_ratio = n_total_clicked / self.n_ctr
        self.total_clicked_ratio_list.append(total_clicked_ratio)
        return True

    def update(self, n_start_opt_hyper_param=0):
        A, B, mu_tilda, gamma_tilda = self.calc_tau()
        # start = time.time()
        # cov = scipy.sparse.linalg.inv(A) # This is slow
        #print("cholesky started")
        factor = cholesky(A)
        #print("cholesky finished")
        cov = scipy.sparse.csc_matrix(factor(np.eye(A.shape[0])))
        # end = time.time() - start
        # print("one calc: %s sec" % end)
        #print("A, B, mu_tilda, gamma_tilda ", A, B, mu_tilda, gamma_tilda)
        self.mu = mat_flatten(cov.dot(B).dot(mu_tilda))
        self.sigma = mat_flatten(np.sqrt(cov[np.diag_indices_from(cov)]))
        
        # Update hyperparamters
        if self.update_hyperparam is not None and self.point_info_manager.update_cnt > n_start_opt_hyper_param:
            self.update_hyperparam()

    def update_hyper_params_by_adj_idxes(self):
        T = self.point_info_manager.get_T(excludes_none=False)
        var_list = []
        for i, j in self.adj_pair_list:
            t_i, t_j = T[i], T[j]
            if (t_i is not None) and (t_j is not None):
                var_list.append((t_i - t_j) ** 2)

        var_list = np.unique(var_list)
        var = var_list.mean()
        if self.n_ctr:
            self.GAMMA_Y = 1 / var
            self.GAMMA = self.GAMMA_Y * (2 * self.ndim)
            self.GAMMA0 = self.GAMMA * 0.01
        else:
            self.GAMMA_Y = 1 / var / (self.ndim ** 2)
            self.GAMMA = self.GAMMA_Y * (2 * self.ndim)
            self.GAMMA0 = self.GAMMA * 0.01
        self.ALPHA = np.mean(self.point_info_manager.get_T(excludes_none=True))

        #print("New GammaY: %s" % self.GAMMA_Y)
        #print("New Gamma: %s" % self.GAMMA)
        #print("New ALPHA: %s" % self.ALPHA)

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

        #print(Lambda.shape, A.shape)

        t = y_hat.dot(Lambda.dot(y_hat[:, np.newaxis]))[0] / 2

        self.k += 1 / 2
        self.theta = self.theta / (t * self.theta + 1)
        self.GAMMA_Y = (self.k - 1) * self.theta
        self.GAMMA = self.GAMMA_Y * (2 * self.ndim)
        self.GAMMA0 = self.GAMMA * 0.01

        if self.normalize_output is None:
            self.ALPHA = np.mean(self.point_info_manager.get_T(excludes_none=True))
            #print("New ALPHA: %s" % self.ALPHA)

#        print("New GammaY: %s" % self.GAMMA_Y)
#        print("New Gamma: %s" % self.GAMMA)
#        print("t: %s" % t)
#        print('k: %s, theta: %s' % (self.k, self.theta))

        log_likelihood = (y_hat.dot(Lambda.dot(y_hat[:, np.newaxis])) - np.log(np.linalg.det(Lambda) + 0.00001))[0]
#        print('log_likelihood: %s' % log_likelihood)

    # TODO this method does not work well...
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

#        print("New GammaY: %s" % self.GAMMA_Y)
#        print("New Gamma: %s" % self.GAMMA)

        self.ALPHA = self.point_info_manager.get_real_T(excludes_none=True).mean()
#        print("New ALPHA: %s" % self.ALPHA)

        self.log_marginal_likelihood_value_ = -np.min(lml_values)
#        print("log_marginal_likelihood: %s" % self.log_marginal_likelihood_value_)

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
#        print('log_likelihood: %f' % log_likelihood)

        Lambda = B - B.dot(A_inv.dot(B))
        Lambda_inv = np.linalg.inv(Lambda)

        M_gamma = A2 - A2.dot(2 * gamma * A_inv - (gamma ** 2) * A_inv.dot(A2).dot(A_inv)).dot(A2)

        mean_r_grid = np.nan_to_num(np.array([r.sum() for r in r_grid]) / n_grid)

        log_likelihood_grad_gamma = mean_r_grid.dot(M_gamma.dot(mean_r_grid[:, np.newaxis]))[0]

        tmpMat = A2.dot(A_inv).dot(A1).dot(A_inv).dot(A2)

        log_likelihood_grad_gamma_y = (gamma ** 2) * (
            np.trace(Lambda_inv.dot(tmpMat) + mean_r_grid.dot(tmpMat.dot(mean_r_grid[:, np.newaxis]))[0]))

        log_likelihood_gradient = np.array([log_likelihood_grad_gamma, log_likelihood_grad_gamma_y])
#        print('log_likelihood_grad: %s' % log_likelihood_gradient)

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

            elif callable(self.optimizer):
                theta_opt, func_min = \
                    self.optimizer(obj_func, initial_theta, bounds=bounds)
            else:
                raise ValueError("Unknown optimizer %s." % self.optimizer)
        else:
            theta_opt, func_min, convergence_dict = \
                fmin_l_bfgs_b(obj_func, initial_theta, approx_grad=True, bounds=bounds)

        return theta_opt, func_min
