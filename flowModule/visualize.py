import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('pgf')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.lines as mlines

c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] 

formatting = {
    "DUN": {"color": c[0], "linestyle": "-", "marker": "o", "label": "DUN"},
    "ensemble": {"color": c[2], "linestyle": "-.", "marker": "o", "label": "Ensemble"}, 
    "dropout": {"color": c[3], "linestyle": ":", "marker": "o", "label": "Dropout"}, 
    "SGD": {"color": c[1], "linestyle": "--", "marker": "o", "label": "SGD"},
    "DUN (exact)": {"color": c[6], "linestyle": (0, [6, 2, 1, 2, 1, 2]), "marker": "o", "label": "DUN (exact)"},
    "dropout (0.3)": {"color": c[7], "linestyle": ":", "marker": "p", "label": "Dropout (0.3)"},
    "flow ": {"color": c[8], "linestyle": "-", "marker": "o", "label": "flow"},
}

text_width = 5.50107 # in  --> Confirmed with template explanation
golden_ratio = (5**.5 - 1) / 2


show_range = 5
ylim = 3

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, line_alpha=1, ax=None, lw=1, linestyle='-', fill_linewidths=0.2):
    ax = ax if ax is not None else plt.gca()
    # yerr *= 100
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    plt_return = ax.plot(x, y, color=color, lw=lw, linestyle=linestyle, alpha=line_alpha)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill, linewidths=fill_linewidths)
    return plt_return

# Visualize the result
def visualize_uncertainty(savePath, gt_x, gt_y, xdata, mean, var):
    plt.figure(dpi=200)
    var = np.sqrt(var)
    plt.plot(gt_x, gt_y, 'ok', ms=1)
    plt.plot(xdata, mean, '-', color='g')
    plt.plot(xdata, var, '-', color='r')
    plt.ylim([-ylim, ylim])
    plt.xlim([-show_range, show_range])
    mean = np.array(mean)
    var = np.array(var)
    plt.fill_between(xdata, mean - var, mean + var, color='g', alpha=0.1)
    plt.tight_layout()
    plt.savefig(savePath, format='png', bbox_inches='tight')

def plot_err_props(df, conditions, add_cond, ax, formatting, **kwargs):
    filt = (df[list(conditions)] == pd.Series(conditions)).all(axis=1)
    df_filt = df[filt].dropna(subset=['err_props'])
    
    # df_filt[df_filt.use_no_train_post == True].method = "DUN (exact)"
    
    for idx, method in enumerate(list(add_cond)):
        filt_add = (df_filt[list(add_cond[method])] == pd.Series(add_cond[method])).all(axis=1)
        df_filt_add = df_filt[filt_add]
        meth_filt = method
        if "DUN" in meth_filt: meth_filt = "DUN"
        err_props_list = np.array([[1 - float(number) for number in row["err_props"][1:-2].split(" ") if number != '']
         for _, row in df_filt_add[df_filt_add.method == meth_filt].iterrows()])
        
        mean = np.mean(err_props_list, axis=0)
        std = np.std(err_props_list, axis=0)
        errorfill(np.arange(0, 1, 0.005), mean, std, alpha_fill=0.2, color=formatting[method]["color"],
                  linestyle=formatting[method]["linestyle"], ax=ax, **kwargs)

    if conditions["dataset"] == "Fashion" or conditions["dataset"] == "MNIST": 
        rejection_step = np.arange(0, 0.5, 0.005)
        theoretical_maximum = 1 / (2 - 2*rejection_step)
    elif conditions["dataset"] == "CIFAR10" or conditions["dataset"] == "CIFAR100":
        rejection_step = np.arange(0, 1-0.27753, 0.005)
        theoretical_maximum = (10000)/((10000 + 26032)*(1-rejection_step))
    elif conditions["dataset"] == "SVHN":
        rejection_step = np.arange(0, 1-0.72247, 0.005)
        theoretical_maximum = (26032)/((10000 + 26032)*(1-rejection_step))
        
        
    ax.plot(rejection_step, theoretical_maximum, color="k", lw=1, **kwargs)


def plot_rot_stats(df, stat, conditions, add_cond, ax, formatting, **kwargs):
    filt = (df[list(conditions)] == pd.Series(conditions)).all(axis=1)
    df_filt = df[filt].dropna(subset=[stat]).copy()
    df_filt = df_filt[df_filt.corruption == 0.]
    df_filt = df_filt[df_filt.dataset == "MNIST"]
    
    for idx, method in enumerate(list(add_cond)):
        filt_add = (df_filt[list(add_cond[method])] == pd.Series(add_cond[method])).all(axis=1)
        df_filt_add = df_filt[filt_add]
        meth_filt = method
        if "DUN" in meth_filt: meth_filt = "DUN"
        
        
        rot_stats = df_filt_add[df_filt_add.method == meth_filt].groupby(['rotation'])[stat]
        mean = rot_stats.mean()
        std = rot_stats.std()
        
        errorfill(np.arange(0, 181, 15), mean, std, alpha_fill=0.2, color=formatting[method]["color"],
                  linestyle=formatting[method]["linestyle"], ax=ax, **kwargs)
        

def plot_cor_stats(df, stat, conditions, add_cond, ax, formatting, cifar="10", **kwargs):
    filt = (df[list(conditions)] == pd.Series(conditions)).all(axis=1)
    df_filt = df[filt].dropna(subset=[stat]).copy()
    df_filt = df_filt[df_filt.rotation == 0.]
    df_filt = df_filt[df_filt.dataset == "CIFAR" + cifar]
    
    for idx, method in enumerate(list(add_cond)):
        filt_add = (df_filt[list(add_cond[method])] == pd.Series(add_cond[method])).all(axis=1)
        df_filt_add = df_filt[filt_add]
        meth_filt = method
        if "DUN" in meth_filt: meth_filt = "DUN"
        
        
        rot_stats = df_filt_add[df_filt_add.method == meth_filt].groupby(['corruption'])[stat]
        mean = rot_stats.mean()
        std = rot_stats.std()
        
        x = np.arange(0, 6, 1)
        errorfill(x, mean, std, alpha_fill=0.2, lw=0.6, color=formatting[method]["color"],
                  linestyle=formatting[method]["linestyle"], ax=ax, **kwargs)
        ax.scatter(x, mean, s=4, color=formatting[method]["color"], marker=formatting[method]["marker"], **kwargs)