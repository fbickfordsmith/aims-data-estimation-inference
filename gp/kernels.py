import numpy as np
from scipy.spatial.distance import cdist

def exponentiated_quadratic(x_0, x_1, variance, lengthscale):
    denom = 2 * lengthscale ** 2
    return variance * np.exp(- cdist(x_0, x_1, 'sqeuclidean') / denom)

def periodic(x_0, x_1, variance, lengthscale, period):
    sine = 2 * np.sin(np.pi * cdist(x_0, x_1, 'euclidean') / period) ** 2
    return variance * np.exp(- sine / lengthscale ** 2)

def periodic_exp_quad(x_0, x_1, var_per, len_per, period, var_eq, len_eq):
    K_periodic = periodic(x_0, x_1, var_per, len_per, period)
    K_exp_quad = exponentiated_quadratic(x_0, x_1, var_eq, len_eq)
    return K_periodic + K_exp_quad
