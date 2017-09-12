# coding: utf-8
import itertools

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

from gphypo.base_bo import BaseBO


class GP_BO(BaseBO):
    def __init__(self, bo_param_list, environment, noise=False, gt_available=False, my_kernel=None,
                 burnin=0, optimizer="fmin_l_bfgs_b", normalize_output=None, n_early_stopping=None,
                 acquisition_func='ucb', acquisition_param_dic={'beta': 5, 'pi': 0.01}):
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

        super(GP_BO, self).__init__(bo_param_list, environment,
                                    n_early_stopping=n_early_stopping,
                                    gt_available=gt_available, normalize_output=normalize_output,
                                    acquisition_func=acquisition_func, acquisition_param_dic=acquisition_param_dic,
                                    optimizer=optimizer)

        if my_kernel is None:
            # Instantiate a Gaussian Process model
            my_kernel = C(1, constant_value_bounds="fixed") * RBF(2, length_scale_bounds="fixed")  # works well, but not so sharp
            #     my_kernel = Matern(nu=2.5) # good
            if noise:
                my_kernel += WhiteKernel(1e-1)
        else:
            my_kernel = my_kernel

        self.gp = GaussianProcessRegressor(kernel=my_kernel, n_restarts_optimizer=25, optimizer=self.optimizer)

        # BurnIn
        if burnin and not environment.reload:
            self.do_grid_based_burnin()

    def do_grid_based_burnin(self):
        burnin = 2 ** self.ndim
        quarter_point_list = [[bo_param[len(bo_param) // 4], bo_param[len(bo_param) // 4 * 3]] for bo_param in
                              self.bo_param_list]

        adj_quarter_point_list = [[bo_param[len(bo_param) // 4 + 1], bo_param[len(bo_param) // 4 * 3 + 1]] for
                                  bo_param in
                                  self.bo_param_list]

        for coodinate, adj_coordinate in zip(itertools.product(*quarter_point_list),
                                             itertools.product(*adj_quarter_point_list)):
            coodinate = np.array(coodinate)
            self.sample(coodinate)
            self.sample(adj_coordinate)

        npX, npT = self.point_info_manager.get_observed_XT_pair(gets_real=True)
        npX = np.array(npX).astype(np.float64)
        npT = np.array(npT).astype(np.float64)

        self.gp.fit(npX, npT)
        self.mu, self.sigma = self.gp.predict(self.X_grid, return_std=True)
        print('%d burins has finished!' % burnin)

    def learn(self):
        T = self.point_info_manager.get_T(excludes_none=True)# T: the means of all the points; either normalized or not
        grid_idx = np.argmax(self.acquisition_func.compute(self.mu, self.sigma, T))# this line is crucial
        continue_flg = self.sample(self.X_grid[grid_idx])# self.sample alters the optimizer's point info manager which contains all the points

        if not continue_flg:
            return False

        npX, npT = self.point_info_manager.get_observed_XT_pair(gets_real=False)
        npX = np.array(npX).astype(np.float64)
        npT = np.array(npT).astype(np.float64)

        self.gp.fit(npX, npT)
        self.mu, self.sigma = self.gp.predict(self.X_grid, return_std=True)
        print(str(self.gp.kernel))
        return True
