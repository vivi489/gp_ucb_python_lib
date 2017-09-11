# coding: utf-8
from scipy.special import logit


def transform_click_val2real_val(n0, n1, alpha0=1, alpha1=1):
    '''

    :param n0: the number of 0 (not clicked) values
    :param n1: the number of 0 (not clicked) values
    :param alpha0: the prior of 0 (not clicked) values
    :param alpha1: the prior of 0 (not clicked) values
    :return: transformed real number
    '''
    prob = (n1 + alpha1) / (n1 + alpha1 + n0 + alpha0)

    return logit(prob)


if __name__ == '__main__':
    print(transform_click_val2real_val(10, 1))
    print(transform_click_val2real_val(1e10, 1e10))
    print(transform_click_val2real_val(1e10, 0))
    print(transform_click_val2real_val(0, 1e10))
