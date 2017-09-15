import abc, time

import numpy as np
from scipy.stats import norm


class BaseAcquisitionFunction(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, param_dic):
        self.param_dic = param_dic

    def preprocess_observation_list(self, observation_list):
        return [x for x in observation_list if x is not None]  # remove None

    @abc.abstractmethod
    def compute(self, mu, sigma, observation_list, **args):
        pass


class EI(BaseAcquisitionFunction):
    def compute(self, mu, sigma, observation_list, **args):
        observation_list = self.preprocess_observation_list(observation_list)
        par = self.param_dic["par"]
        if len(observation_list) == 0:
            z = (mu - par) / sigma
        else:
            z = (mu - max(observation_list) - par) / sigma
        f = sigma * (z * norm.cdf(z) + norm.pdf(z))
        return f


class PI(BaseAcquisitionFunction):
    def compute(self, mu, sigma, observation_list, **args):
        observation_list = self.preprocess_observation_list(observation_list)
        par = self.param_dic["par"]
        inc_val = 0
        if len(observation_list) > 0:
            inc_val = max(observation_list)
        z = - (inc_val - mu - par) / sigma
        return norm.cdf(z)


class UCB(BaseAcquisitionFunction):
    def __init__(self, param_dic, type="normal", d_size=None):
        super().__init__(param_dic)
        self.learn_cnt = 1
        self.type = type
        self.d_size = d_size

    def get_beta(self):
        global beta
        if self.type == "normal":
            beta = self.param_dic['beta']
        elif self.type == 'theorem1':
            delta = 0.9  # must be in (0, 1)
            beta = 2 * np.log(self.d_size * ((self.learn_cnt * np.pi) ** 2) / (6 * delta))
        return beta

    def compute(self, mu, sigma, observation_list, **args):
        beta = self.get_beta()
        self.learn_cnt += 1
        return mu + sigma * np.sqrt(beta)
    

class Thompson(BaseAcquisitionFunction):
    def __init__(self, param_dic, type="normal", d_size=None):
        super().__init__(param_dic)
        self.learn_cnt = 1
        self.type = type
        self.d_size = d_size
        self.tsFactor = self.param_dic["tsFactor"]
        np.random.seed(int(time.time()))

    def compute(self, mu, sigma, observation_list, **args):
        #beta = self.get_beta()
        self.learn_cnt += 1
        return np.random.normal(mu, sigma * self.tsFactor)


class EnsembledAC(BaseAcquisitionFunction):
    def __init__(self, param_dic, type="normal", d_size=None):
        super().__init__(param_dic)
        self.learn_cnt = 1
        self.type = type
        self.d_size = d_size
        self.eps = self.param_dic['eps'] # interval [0, 1]
        self.tsFactor = self.param_dic["tsFactor"]
        np.random.seed(int(time.time()))

    def get_beta(self):
        global beta
        if self.type == "normal":
            beta = self.param_dic['beta']
        elif self.type == 'theorem1':
            delta = 0.9  # must be in (0, 1)
            beta = 2 * np.log(self.d_size * ((self.learn_cnt * np.pi) ** 2) / (6 * delta))
        return beta

    def compute(self, mu, sigma, observation_list, **kwargs): 
        self.learn_cnt += 1
        #mask = np.zeros(mu.shape).astype(np.float64)
        #mask[int(np.random.rand()*len(mask))] = 1.0
        drop = kwargs["drop"]
        return np.random.normal(mu, sigma * self.tsFactor) if (np.random.rand()<self.eps and drop) else (mu + sigma * np.sqrt(self.get_beta()))



    