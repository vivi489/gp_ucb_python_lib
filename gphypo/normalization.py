import numpy as np


def zero_one_normalization(X, lower=None, upper=None, accepts_none=True):
    if len(X) < 2:
        return X, np.nan, np.nan


    if lower is None:
        lower = np.min(X, axis=0)
    if upper is None:
        upper = np.max(X, axis=0)

    X_normalized = np.true_divide((X - lower), (upper - lower))

    return X_normalized, lower, upper


def zero_one_unnormalization(X_normalized, lower, upper):
    return lower + (upper - lower) * X_normalized


def zero_mean_unit_var_normalization(X, mean=None, std=None):
    if len(X) < 2:
        return X, np.nan, np.nan

    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
    try:
        X_normalized = (X - mean) / std

    except ZeroDivisionError:
        return X, mean, 0

    return X_normalized, mean, std


def zero_mean_unit_var_unnormalization(X_normalized, mean, std):
    return X_normalized * std + mean

if __name__ == '__main__':
    A = np.array([])
    print (len(A))
    print(zero_mean_unit_var_normalization(A))
    print (zero_one_normalization(A))
