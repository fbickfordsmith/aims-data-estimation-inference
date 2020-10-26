import matplotlib.pyplot as plt
import numpy as np
from .data import load_data

plt.rc('font', **{'size':12, 'family':'sans-serif', 'sans-serif':['cm']})
plt.rc('text', **{'usetex':True, 'latex.preamble':r'\usepackage[cm]{sfmath}'})

labels_axes = {
    'x':'Time (days; 0 = 12:05pm, 26/05/2007)',
    'y':'Tide height (m)'}

labels_legend = {
    'mean':'predictive mean',
    '1std':'mean ± 1 std',
    '2std':'mean ± 2 std',
    'train':'training points',
    'test':'test points'}

styles = {
    'mean':dict(linestyle='-', color='C0', label=labels_legend['mean']),
    '1std':dict(color='C0', alpha=0.2, label=labels_legend['1std']),
    '2std':dict(color='C0', alpha=0.1, label=labels_legend['2std']),
    'scatter':dict(linestyle='', marker='.', markersize=5),
    'train':dict(color='C0', label=labels_legend['train']),
    'test':dict(color='C3', label=labels_legend['test'])}

def add_legend(ax):
    inds = []
    handles, labels = ax.get_legend_handles_labels()
    labels = np.array(labels)
    for series in ['mean', '1std', '2std', 'train', 'test']:
        inds.append(np.flatnonzero(labels == labels_legend[series])[0])
    ax.legend(
        [handles[i] for i in inds],
        [labels[i] for i in inds],
        handlelength=0.8,
        loc='upper center',
        ncol=5)
    return ax

def format_axes(ax, x_plot):
    ax.set_xlabel(labels_axes['x'])
    ax.set_ylabel(labels_axes['y'])
    ax.set_xlim((min(x_plot), max(x_plot)))
    ax.set_ylim((0.5, 5.5))
    return ax

def plot_effects_mean_kernel(kernel):
    num_points = 100
    x = np.linspace(0, 1, num_points)
    x_2d = x.reshape(-1, 1)
    # Each row of params is a pair of [mean, var, length, label] for a plot.
    params = [
        [[0, 0.5, 0.05, 'constant $m$'], [4, 0.5, 0.05, 'linear $m$']],
        [[0, 0.05, 0.05, 'low $\sigma^2$'], [0, 1.5, 0.05, 'high $\sigma^2$']],
        [[0, 1, 0.03, 'low $l$'], [0, 1, 0.3, 'high $l$']]]
    colors = ['C0', 'C3']
    fig, ax = plt.subplots(
        nrows=1, ncols=3, figsize=(6, 1.7), sharex=True, sharey=True)
    for i, params_i in enumerate(params):
        for j, params_j in enumerate(params_i):
            mean, var, length, label = params_j
            cov = kernel(x_2d, x_2d, var, length)
            sample = np.random.multivariate_normal(mean * x, cov)
            ax[i].plot(x, sample, color=colors[j], label=label)
            ax[i].set_xticks(())
            ax[i].set_yticks(())
            ax[i].legend(handlelength=0.8, loc='upper left', ncol=1)
    ax[1].set_xlabel('$x$')
    ax[0].set_ylabel('$f(x)$')
    ax[0].set_xlim((0, 1))
    ax[0].set_ylim(top=ax[0].get_ylim()[1]*1.5)
    fig.tight_layout(pad=0.3)
    return fig, ax

def plot_posterior(mean, cov, time=None):
    x_train, x_test, x_plot, y_train, y_test, _ = load_data(
        normalise=False, num_dims=1)
    mean = mean.reshape(-1)
    std = np.sqrt(np.diag(cov))
    low_1std, high_1std = (mean - std), (mean + std)
    low_2std, high_2std = (mean - 2 * std), (mean + 2 * std)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 3.5))
    if time is not None:
        ax.axvspan(time, max(x_plot), color='grey', linewidth=0, alpha=0.1)
    ax.plot(x_plot, mean, **styles['mean'])
    ax.fill_between(x_plot, low_1std, high_1std, **styles['1std'])
    ax.fill_between(x_plot, low_2std, high_2std, **styles['2std'])
    ax.plot(x_train, y_train, **styles['scatter'], **styles['train'])
    ax.plot(x_test, y_test, **styles['scatter'], **styles['test'])
    ax = add_legend(ax)
    ax = format_axes(ax, x_plot)
    fig.tight_layout()
    return fig, ax

def plot_regression_problem(gap_width=5):
    x, y = load_data(mode='simple_xy')
    x_train, _, x_plot, y_train, _, _ = load_data(normalise=False, num_dims=1)
    inds_test = np.flatnonzero(np.isnan(y))
    # Find gaps of at least gap_width points in the training data.
    i = 0
    inds_gaps = []
    while i < len(inds_test) - gap_width:
        j = gap_width
        while inds_test[i] + j == inds_test[i + j]:
            if j == gap_width:
                inds_gaps.append([inds_test[i], inds_test[i + j]])
            else:
                inds_gaps[-1] = [inds_test[i], inds_test[i + j]]
            j += 1
        if j > gap_width:
            i += j - 1
        else:
            i += 1
    # Plot the training data. Highlight regions with wide gaps.
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 2.5))
    for i_0, i_1 in inds_gaps:
        ax.axvspan(x[i_0], x[i_1], color='grey', linewidth=0, alpha=0.1)
    ax.plot(x_train, y_train, '.', markersize=3)
    ax.set_xlabel(labels_axes['x'])
    ax.set_ylabel(labels_axes['y'])
    fig.tight_layout()
    return fig, ax

def plot_samples(mean, cov, num_samples):
    _, _, x_plot, _, _, _ = load_data(num_dims=1)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 3.5))
    for i in range(num_samples):
        sample = np.random.multivariate_normal(mean.reshape(-1), cov)
        ax.plot(x_plot, sample, alpha=0.5, label=f'sample {i+1}')
    ax = format_axes(ax, x_plot)
    fig.tight_layout()
    return fig, ax
