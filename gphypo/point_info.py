from collections import Counter

import numpy as np

from gphypo import normalization


class PointInfo(object):
    def __init__(self, coordinate):
        self.coordinate = coordinate
        self.val2cnt = Counter()

    def update_val2cnt(self, new_val2cnt):
        self.val2cnt.update(new_val2cnt)

    def get_total_zero_mean_unit_var_normalized_val(self, mu, sigma):
        s = 0
        for val, cnt in self.val2cnt.items():
            val = float(val)
            normalized_val = (val - mu) / sigma
            s += normalized_val * cnt
        return s

    def get_total_zero_one_normalized_val(self, lower, upper):
        s = 0
        for val, cnt in self.val2cnt.items():
            val = float(val)
            normalized_val = np.true_divide((val - lower), (upper - lower))
            s += normalized_val * cnt
        return s

    def get_total_val(self):
        s = 0
        for val, cnt in self.val2cnt.items():
            val = float(val)
            s += val * cnt
        return s

    def get_total_cnt(self):
        s = 0
        for val, cnt in self.val2cnt.items():
            s += cnt
        return s

    def get_mean(self):
        total_val = self.get_total_val()
        total_cnt = self.get_total_cnt()
        if total_cnt == 0:
            return None

        return total_val / total_cnt


class PointInfoManager(object):
    def __init__(self, X_grid, normalize_output):
        self.X_grid = X_grid.astype(np.float64)
        self.n_points = X_grid.shape[0]
        self.normalize_output = normalize_output
        self.point_info_list = [PointInfo(x) for x in X_grid]
        self.X_seq = []
        self.T_seq = []
        self.update_cnt = 0
        self.T_lower, self.T_upper = 0, 1  ## For 'zero_one' normalization
        self.T_mean, self.T_std = 0, 1  ## For 'zero_mean_unit_var' normalization

    def get_real_T(self, excludes_none=False):
        real_T = np.array([point_info.get_mean() for point_info in self.point_info_list])
        observed_idx = [i for i, t in enumerate(real_T) if t is not None]
        if excludes_none:
            return real_T[observed_idx]
        return real_T

    def get_observed_idx(self):
        real_T = self.get_real_T()
        observed_idx = [i for i, t in enumerate(real_T) if t is not None]
        return observed_idx

    def get_observed_XT_pair(self):
        real_T = self.get_real_T()
        observed_idx = [i for i, t in enumerate(real_T) if t is not None]
        return self.X_grid[observed_idx], real_T[observed_idx]

    def set_normalized_params(self):
        _, observed_real_T = self.get_observed_XT_pair()

        if self.normalize_output == 'zero_mean_unit_var':
            _, self.T_mean, self.T_std = normalization.zero_mean_unit_var_normalization(observed_real_T)

        elif self.normalize_output == 'zero_one':
            _, self.T_lower, self.T_upper = normalization.zero_one_normalization(observed_real_T)

    def get_normalized_T(self, excludes_none=True):
        global observed_normalized_T
        _, observed_real_T = self.get_observed_XT_pair()
        normalized_T = np.array([None] * self.n_points)
        observed_idx = self.get_observed_idx()

        if self.normalize_output == 'zero_mean_unit_var':
            observed_normalized_T, _, _ = normalization.zero_mean_unit_var_normalization(observed_real_T)

        elif self.normalize_output == 'zero_one':
            observed_normalized_T, _, _ = normalization.zero_one_normalization(observed_real_T)

        if excludes_none:
            return observed_normalized_T

        normalized_T[observed_idx] = observed_normalized_T

        return normalized_T

    def get_T(self, excludes_none=False):
        if self.normalize_output:
            return self.get_normalized_T(excludes_none=excludes_none)

        return self.get_real_T(excludes_none=excludes_none)

    def get_unnormalized_value_list(self, value_list):
        if self.normalize_output == 'zero_mean_unit_var':
            return normalization.zero_mean_unit_var_unnormalization(value_list, self.T_mean, self.T_std)

        elif self.normalize_output == 'zero_one':
            return normalization.zero_one_unnormalization(value_list, self.T_lower, self.T_upper)

    def get_sum_grid(self):
        if self.normalize_output == 'zero_mean_unit_var':
            self.set_normalized_params()
            return np.array(
                [point_info.get_total_zero_mean_unit_var_normalized_val(self.T_mean, self.T_std) for point_info in
                 self.point_info_list])

        elif self.normalize_output == 'zero_one':
            self.set_normalized_params()
            return np.array(
                [point_info.get_total_zero_one_normalized_val(self.T_lower, self.T_upper) for point_info in
                 self.point_info_list])
        else:
            return np.array([point_info.get_total_val() for point_info in self.point_info_list])

    def get_n_grid(self):
        return np.array([point_info.get_total_cnt() for point_info in self.point_info_list])

    def update(self, coordinate, val2cnt):
        coordinate = coordinate.astype(np.float64)
        diff = self.X_grid - coordinate
        diff2 = diff * diff
        row_idx = np.where(diff2.sum(axis=1) == 0)[0][0]

        self.point_info_list[row_idx].update_val2cnt(val2cnt)

        self.X_seq.append(self.X_grid[row_idx])
        self.T_seq.append(list(val2cnt.keys())[0])

        self.update_cnt += 1
