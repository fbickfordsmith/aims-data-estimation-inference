import numpy as np
import pandas as pd

def load_data(mode=None, normalise=True, num_dims=2):
    data = pd.read_csv('sotonmet.txt')
    if mode == 'dataframe':
        return data
    x = pd.to_datetime(data['Update Date and Time (ISO)'])
    x = np.array((x - x[0]).dt.total_seconds() / (24 * 60 * 60))
    y = np.array(data['Tide height (m)'])
    if mode == 'simple_xy':
        return x, y
    inds_train = np.flatnonzero(~np.isnan(y))
    inds_test = np.flatnonzero(np.isnan(y))
    x_train = x[inds_train]
    x_test = x[inds_test]
    x_plot = np.linspace(min(x_train) - 1, max(x_train) + 1, 1000)
    y_true = np.array(data['True tide height (m)'])
    y_train = y[inds_train]
    y_test = y_true[inds_test]
    y_mean = np.mean(y_train)
    if normalise:
        y_train -= y_mean
        y_test -= y_mean
    if num_dims == 2:
        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)
        x_plot = x_plot.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
    return x_train, x_test, x_plot, y_train, y_test, y_mean
