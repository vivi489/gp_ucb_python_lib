import os
import subprocess
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from matplotlib import cm, gridspec
from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from gphypo import normalization
from gphypo.acquisition_func import *
from gphypo.point_info import PointInfoManager
from gphypo.transform_val import transform_click_val2real_val
from gphypo.util import mkdir_if_not_exist


def mat_flatten(x):
    return np.array(x).flatten()


class BaseBO(object):
    __metaclass__ = ABCMeta

    def __init__(self, bo_param_list, environment, n_early_stopping=None,
                 gt_available=False, normalize_output="zero_mean_unit_var",
                 acquisition_func='ucb', acquisition_param_dic={'beta': 5, 'pi': 0.01}, optimizer='fmin_l_bfgs_b',
                 does_pairwise_sampling=False, n_ctr=None):

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
        self.meshgrid = np.array(np.meshgrid(*bo_param_list)) #this line is crucial
        self.environment = environment
        self.optimizer = optimizer
        self.does_pairwise_sampling = does_pairwise_sampling

        assert normalize_output in [None, "zero_mean_unit_var", "zero_one"]
        
        self.normalize_output = normalize_output
        self.ndim_list = [len(bo_param) for bo_param in bo_param_list]
        self.ndim = len(self.meshgrid)

        self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T #this line is crucial
        self.X_grid2idx_dic = {tuple(x): i for i, x in enumerate(self.X_grid)}
        self.n_points = self.X_grid.shape[0]

        assert acquisition_func in ["ucb", "ei", "pi", "ts", "greedy"]
        self.acquisition_func_name = acquisition_func
        if acquisition_func == 'ucb':
            self.acquisition_func = UCB(acquisition_param_dic, d_size=self.X_grid.shape[0])
        elif acquisition_func == 'ei':
            self.acquisition_func = EI(acquisition_param_dic)
        elif acquisition_func == 'pi':
            self.acquisition_func = PI(acquisition_param_dic)
        elif acquisition_func == 'ts':
            self.acquisition_func = Thompson(acquisition_param_dic)
        else:
            self.acquisition_func = GreedyEps(acquisition_param_dic, type="theorem1", d_size=self.X_grid.shape[0])

        self.gt_available = gt_available
        self._set_z()
        self.point_info_manager = PointInfoManager(self.X_grid, self.normalize_output)
        self._set_early_stopping_params(n_early_stopping)

        self.mu = abs(np.random.randn(self.n_points))
        self.sigma = abs(np.random.randn(self.n_points)) * 2.5
        self.n_ctr = n_ctr
        self.randomly_total_clicked_ratio_list = []
        
        # Reload past results if exits
        if environment.reload:
            self.reload_data_from_environment_csv()

    def _set_z(self):
        # Calculate ground truth (self.z)
        if self.gt_available:
            if self.ndim == 1:
                self.z = self.environment.sample(self.X_grid, get_ground_truth=True)
            elif self.ndim == 2:
                nrow, ncol = self.meshgrid.shape[1:]
                self.z = self.environment.sample(self.X_grid, get_ground_truth=True).reshape(nrow, ncol)
            elif self.ndim == 3:
                self.z = self.environment.sample(self.X_grid, get_ground_truth=True)
        else:
            self.z = None

    def _set_early_stopping_params(self, n_early_stopping):
        self.bestX = None
        self.bestT = -np.inf
        self.cnt_since_bestT = 0

        if n_early_stopping:
            self.n_early_stopping = n_early_stopping
        else:
            self.n_early_stopping = np.inf

    def do_random_burnin(self, n_burnin):
        for _ in range(n_burnin):
            n_points = self.X_grid.shape[0]
            rand_idx = np.random.randint(n_points)
            x = self.X_grid[rand_idx]
            self.sample(x, n_exp=self.n_ctr)

    def reload_data_from_environment_csv(self):
        for key, row in self.environment.result_df.iterrows():
            x = row[self.environment.bo_param_names].as_matrix()
            t = float(row['output'])
            #print(x, t, row.n_exp)
            n_exp = round(float(row.n_exp))
            if n_exp > 1:
                n1 = t
                n0 = n_exp - n1
                t = transform_click_val2real_val(n0, n1)
                if type(t) == list or type(t) == np.ndarray:
                    t = t[0]
                self.point_info_manager.update(x, {t: n_exp})
            else:
                self.point_info_manager.update(x, {t: 1})

        #print("Finished reloading csv.")

    def calc_true_mean_std(self):
        assert self.gt_available == True
        #print(self.X_grid)
        #print(self.X_grid.shape)
        sampled_y = np.array(self.environment.sample(self.X_grid, get_ground_truth=True))

        return sampled_y.mean(), sampled_y.std()

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def learn_from_clicks(self, ratio_csv_fn='ratio.csv'):
        pass

    def call_mu2ratio(self, mu2ratio_dir='./mu2ratio', mu_sigma_csv_fn='./mu2ratio/mu_sigma.csv',
                      ratio_csv_fn='./mu2ratio/ratios.csv'):
        jar_fn = os.path.join(mu2ratio_dir, 'mu2ratio.jar')
        cmd = "java -jar {} {} {}".format(jar_fn, mu_sigma_csv_fn, ratio_csv_fn)
        subprocess.call(cmd, shell=True)
        #print('mu2ratio finished!!!')

    def sample_using_ratio_csv(self, ratio_csv_fn):
        ratio_df = pd.read_csv(ratio_csv_fn, index_col=0, names=['ratio'])
        #print("raw ratio_df:\n", ratio_df)
        ratio_df.index.name = 'point_id'
        ratio_df = ratio_df.join(self.point_info_df)
        ratio_df['n_exp'] = ratio_df['ratio'] * self.n_ctr
        ratio_df = ratio_df.sort_index()
        #ratio_df format: point_id(index name)   ratio   bo_x    n_exp
        #print(ratio_df)
        for key, row in ratio_df.iterrows():
            x = row[self.environment.bo_param_names].as_matrix()
            #print("sample_using_ratio_csv key=", key, "row.n_exp=", row.n_exp)
            continue_flg = self.sample(x, round(row.n_exp))

    # This method is just for comparison
    def sample_randomly(self):
        random_n_exp_list = np.random.uniform(0, 1, self.n_points)
        random_n_exp_list /= random_n_exp_list.sum()
        random_n_exp_list *= self.n_ctr

        clicked_list = []
        for i, (x, n_exp) in enumerate(zip(self.X_grid, random_n_exp_list)):
            clicked_list.append(self.environment.sample(x, n_exp=int(n_exp)))

        # print(clicked_list)
        n_total_clicked = float(sum(clicked_list))
        total_clicked_ratio = n_total_clicked / self.n_ctr
        self.randomly_total_clicked_ratio_list.append(total_clicked_ratio)

    # Generate the observation value
    def sample(self, x, n_exp=None, overwrite=True): # performs environment sampling on x #extremely IMPORTANT
        #print("base sample:n_exp=", n_exp)
        if n_exp is not None and n_exp > 1:
            n1 = self.environment.sample(x, n_exp=n_exp, overwrite=overwrite)  # Returns the number of clicks
            n0 = n_exp - n1  # Calculates the number of unclick
            t = transform_click_val2real_val(n0, n1)
            if type(t) == list or type(t) == np.ndarray:
                t = t[0]
            self.point_info_manager.update(x, {t: n_exp})
        else: #when n_exp==0, self.environment.sample(n_exp=1) by default, written to self.environment.result_df
            #print("base sample: n_exp<=1", n_exp)
            t = self.environment.sample(x, overwrite=overwrite)
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

    def save_mu_sigma_csv(self, outfn="mu_sigma.csv", point_info_fn='point_info.csv'):
        df = pd.DataFrame({
            'mu': self.mu,
            'sigma': self.sigma
        })
        df.index.name = 'point_id'
        df.to_csv(outfn)
        point_info_df = pd.DataFrame(self.X_grid, columns=self.environment.bo_param_names)
        point_info_df.index.name = 'point_id'
        point_info_df.to_csv(point_info_fn)

        self.point_info_df = point_info_df

        #print('%s was saved!' % outfn)

    # TODO fix this
    def plot_click_distribution(self, output_dir):
        def plot3d_click_distribution():
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

        def plot2d_click_distribution():
            acq_score = self.acquisition_func.compute(self.mu, self.sigma, self.point_info_manager.get_T(), drop=False)
            mu = self.mu.flatten()
            X, T = self.point_info_manager.get_observed_XT_pair(gets_real=True)
            X_seq, T_seq = self.point_info_manager.X_seq, self.point_info_manager.T_seq

            if self.normalize_output:
                mu = self.point_info_manager.get_unnormalized_value_list(mu)
                acq_score = self.point_info_manager.get_unnormalized_value_list(acq_score)

            if self.acquisition_func_name in ["ucb", "ts", "en"]:
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

                clicked_list = self.environment.result_df.output.values[-self.n_points:].astype(np.float64)
                n_total_clicked = clicked_list.sum()
                size_list = (clicked_list / n_total_clicked) * 1000
                ax.scatter([float(x[0]) for x in self.X_grid], [float(x[1]) for x in self.X_grid],
                           mu.reshape(self.meshgrid[0].shape),
                           s=size_list, alpha=0.5, color='r')

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

                clicked_list = self.environment.result_df.output.values[-self.n_points:].astype(np.float64)
                n_total_clicked = clicked_list.sum()
                size_list = (clicked_list / n_total_clicked) * 1000
                upper.scatter([float(x[0]) for x in self.X_grid], [float(x[1]) for x in self.X_grid],
                              mu.reshape(self.meshgrid[0].shape),
                              s=size_list, alpha=0.5, color='r')

                # upper.set_zlabel('f(x)', fontdict={'size': 18})
                # lower.set_zlabel(self.acquisition_func_name.upper(), fontdict={'size': 18})

        def plot1d_click_distribution():
            acq_score = self.acquisition_func.compute(self.mu, self.sigma, self.point_info_manager.get_T(), drop=False)

            mu = self.mu.flatten()
            if self.normalize_output:
                mu = self.point_info_manager.get_unnormalized_value_list(mu)
                acq_score = self.point_info_manager.get_unnormalized_value_list(acq_score)

            X, T = self.point_info_manager.get_observed_XT_pair(gets_real=True)
            X_seq, T_seq = self.point_info_manager.X_seq, self.point_info_manager.T_seq

            if self.acquisition_func_name in ["ucb", "ts", "en"]:
                plt.plot(self.meshgrid[0].astype(float), mu, color='g')
                plt.plot(self.meshgrid[0].astype(float), acq_score, color='y')
                plt.plot(self.meshgrid[0], self.z, alpha=0.3, color='b')

                clicked_list = self.environment.result_df.output.values[-self.n_points:].astype(np.float64)
                n_total_clicked = clicked_list.sum()
                size_list = (clicked_list / n_total_clicked) * 1000
                plt.scatter([float(x[0]) for x in self.X_grid],
                            mu.reshape(self.meshgrid[0].shape),
                            s=size_list, alpha=0.5, color='r')

                #print(clicked_list.shape)
                #print(np.argmax(clicked_list))

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

                clicked_list = self.environment.result_df.output.values[-self.n_points:].astype(np.float64)

                n_total_clicked = clicked_list.sum()
                size_list = (clicked_list / n_total_clicked) * 1000
                upper.scatter([float(x[0]) for x in self.X_grid],
                              mu.reshape(self.meshgrid[0].shape),
                              s=size_list, alpha=0.5, color='r')

                upper.set_ylabel('f(x)', fontdict={'size': 18})
                lower.set_ylabel(self.acquisition_func_name.upper(), fontdict={'size': 18})

        if self.ndim in [1, 2, 3]:
            exec("plot{}d_click_distribution()".format(self.ndim))
            out_fn = os.path.join(output_dir, 'res_%04d.png' % self.point_info_manager.update_cnt)
            mkdir_if_not_exist(output_dir)
            plt.savefig(out_fn, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

        else:
            print("Sorry... Plotting only supports 1, 2, or 3 dim.")

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
            acq_score = self.acquisition_func.compute(self.mu, self.sigma, self.point_info_manager.get_T(), drop=False)
            mu = self.mu.flatten()
            X, T = self.point_info_manager.get_observed_XT_pair(gets_real=True)
            X_seq, T_seq = self.point_info_manager.X_seq, self.point_info_manager.T_seq

            if self.normalize_output:
                mu = self.point_info_manager.get_unnormalized_value_list(mu)
                acq_score = self.point_info_manager.get_unnormalized_value_list(acq_score)

            if self.acquisition_func_name in ["ucb", "ts", "en"]:
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
            acq_score = self.acquisition_func.compute(self.mu, self.sigma, self.point_info_manager.get_T(), drop=False)

            mu = self.mu.flatten()
            if self.normalize_output:
                mu = self.point_info_manager.get_unnormalized_value_list(mu)
                acq_score = self.point_info_manager.get_unnormalized_value_list(acq_score)

            X, T = self.point_info_manager.get_observed_XT_pair(gets_real=True)
            X_seq, T_seq = self.point_info_manager.X_seq, self.point_info_manager.T_seq

            if self.acquisition_func_name in ["ucb", "ts", "en"]:
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
            print("Sorry... Plotting only supports 1, 2, or 3 dim.")
