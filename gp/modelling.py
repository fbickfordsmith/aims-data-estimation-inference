import numpy as np
from scipy.optimize import minimize

def compute_cov_matrix(x_train, y_train, kernel, params, var_noise=0.1):
    K = kernel(x_train, x_train, *params)
    L = np.linalg.cholesky(K + var_noise * np.eye(K.shape[0]))
    L_inv = np.linalg.inv(L)
    alpha = np.dot(L_inv.T, np.dot(L_inv, y_train))
    return K, L, L_inv, alpha

def log_likelihood(mean, cov, y_test, var_noise=0.1):
    var_diag = (np.diag(cov) + var_noise).reshape(-1, 1)
    log_lik = np.log(2 * np.pi * var_diag) + ((y_test - mean) ** 2) / var_diag
    log_lik *= -0.5
    return np.sum(log_lik)

def negative_log_marginal_likelihood(params, x_train, y_train, kernel):
    _, L, _, alpha = compute_cov_matrix(x_train, y_train, kernel, params)
    neg_log_marg_lik = 0.5 * np.dot(y_train.T, alpha)
    neg_log_marg_lik += np.sum(np.log(np.diag(L)))
    neg_log_marg_lik += (x_train.shape[0] / 2) * np.log(2 * np.pi)
    return float(neg_log_marg_lik)

def optimise(x_train, y_train, kernel, num_params, num_starts=1, start=None):
    # Generate num_starts sets of initial settings for kernel parameters.
    if start is None:
        start = np.ones(num_params)
    else:
        start = np.array(start)
    cov_starts = np.diag(0.5 * start)
    starts = [
        np.random.multivariate_normal(start, cov_starts)
        for _ in range(num_starts - 1)]
    starts = np.stack([start] + starts)
    # Optimise log marginal likelihood for each set of initial settings.
    log_marg_lik, params_optimised = [], []
    print(55 * '-' + '\n' + kernel.__name__)
    print(55 * '-' + '\nparams, log marginal likelihood')
    for params_start in starts:
        result = minimize(
            fun=negative_log_marginal_likelihood,
            x0=params_start,
            args=(x_train, y_train, kernel),
            method='L-BFGS-B',
            bounds=num_params*[[1e-6, None]])
        log_marg_lik.append(-result.fun)
        params_optimised.append(result.x)
        print(np.round(result.x, 3), int(-result.fun))
    print(55 * '-')
    return np.max(log_marg_lik), params_optimised[np.argmax(log_marg_lik)]

def optimise_predict_sequential(time, x_train, x_plot, y_train, kernel,
        num_params, num_starts=1):
    ind_time = np.flatnonzero(x_train >= time)[0]
    x_train = x_train[:ind_time]
    y_train = y_train[:ind_time]
    _, params = optimise(x_train, y_train, kernel, num_params, num_starts)
    mean_plot, cov_plot = predict(x_train, x_plot, y_train, kernel, params)
    return mean_plot, cov_plot

def predict(x_train, x_test, y_train, kernel, params, var_noise=0.1):
    _, _, L_inv, alpha = compute_cov_matrix(
        x_train, y_train, kernel, params, var_noise)
    K_s = kernel(x_train, x_test, *params)
    K_ss = kernel(x_test, x_test, *params)
    mean = np.dot(K_s.T, alpha)
    v = np.dot(L_inv, K_s)
    cov = K_ss - np.dot(v.T, v)
    return mean, cov
    
def root_mean_squared_error(mean, y_test):
    return np.sqrt(np.mean((y_test - mean) ** 2))
