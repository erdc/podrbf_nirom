#! /usr/bin/env python

"""
Module for utility codes for visualizing
various features of the POD-RBF
reduced order model
"""

import rom as rom
import greedy as gdy
import rbf as rbf
import pod as pod
import itertools
from matplotlib.offsetbox import AnchoredText
from matplotlib import rcParams
import matplotlib.ticker as ticker
from IPython.display import display
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from numpy.lib.scimath import sqrt as csqrt
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, ScalarFormatter, FormatStrFormatter

from matplotlib import animation
matplotlib.rc('animation', html='html5')

# colors = itertools.cycle()
markers = itertools.cycle(['p', 'd', 'o', '^', 's', 'x', 'D', 'H', 'v', '*'])

# Plot parameters
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 20,
                     'lines.linewidth': 2,
                     # fontsize for x and y labels (was 10)
                     'axes.labelsize': 16,
                     'axes.titlesize': 20,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16,
                     'legend.fontsize': 16,
                     'axes.linewidth': 2})


def plot_sing_val(D_pod, savefig=False):
    """
    Plot the singular value decay
    """

    fig = plt.figure(figsize=(7, 5))
    comp = D_pod.keys()
    index = {}
    mkskip = D_pod[list(D_pod.keys())[0]].shape[0]//15

    for key in D_pod.keys():
        index[key] = np.arange(D_pod[key].shape[0]-1)
        plt.semilogy(index[key], D_pod[key][:-1], marker=next(markers),
                     markevery=mkskip, markersize=8,
                     label='%s' % key, linewidth=3)

    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    plt.legend(fontsize=16)


def plot_centers(ind_greedy, N, alg, **kwargs):
    """
    Plot centers chosen by greedy algorithms
    """

    if 'num_centers' in kwargs:
        x_pos = np.sort(ind_greedy[:kwargs['num_centers']+1])
    else:
        x_pos = np.sort(ind_greedy)

    if 'init' in kwargs:
        x_pos += kwargs['init']
    xran = list(zip(x_pos[:], 1.0*np.ones(len(x_pos[:]))))

    if alg == 'f':
        plt.broken_barh(xran, (0, 1), facecolors=('green'))
    elif alg == 'p':
        plt.broken_barh(xran, (0, 1), facecolors=('blue'))
    elif alg == 'psr':
        plt.broken_barh(xran, (0, 1), facecolors=('magenta'))

    # ax1 = ax #fig.gca()
    plt.xlim([0, N+1])

    plt.yticks([])
    if 'times_offline' in kwargs:
        try:
            dtTmp = np.round(
                (kwargs['times_offline'][1:] - kwargs['times_offline'][:-1]).mean())
            plt.xticks(np.linspace(kwargs['init'], N, 5), np.round(
                np.linspace(dtTmp*kwargs['init'], kwargs['times_offline'][N], 5)/3600.))
        except:
            plt.xticks(np.linspace(0, N, 5), np.round(
                np.linspace(0, kwargs['times_offline'][N], 5)/3600.))
    else:
        plt.xticks([])


def viz_sol(uh, nodes, triangles):
    """
    Visualize the NIROM and HFM solutions over the physical domain
    """

    boundaries = np.linspace(np.amin(uh), np.amax(uh), 11)
    cf = plt.tripcolor(nodes[:, 0], nodes[:, 1],
                       triangles, uh, shading='gouraud')
    plt.axis('equal')

    return cf, boundaries


def viz_err(uh, snap, nodes, triangles):
    """
    Visualize the NIROM solution relative error over the domain
    """

    cf3 = plt.tripcolor(nodes[:, 0], nodes[:, 1],
                        triangles, uh-snap, shading='flat')
    plt.axis('equal')
    cb = plt.colorbar(cf3)

    return cf3


def plot_rms_err(rms, times_online, key, lbl, numc=200, alg='uni', clr='r', mkr='p', t_end=False, **kwargs):
    """
    Plot rms errors vs time for various reduced solutions
    """

    if t_end == False:
        N_end = np.count_nonzero(
            times_online[times_online <= times_online[-1]])
        index = times_online
        end_trunc = N_end+1
    else:
        N_end = np.count_nonzero(times_online[times_online < t_end])
        index = times_online[:N_end+1]
        end_trunc = N_end+1

    try:
        start_trunc = kwargs['start']
    except:
        start_trunc = 0

    mkr_skip = len(index[start_trunc:])//25
    plt.plot(index[start_trunc:], rms[key][start_trunc:end_trunc], color=clr, marker=mkr, markersize=8,
             label='$\mathbf{%s}^{%s}_{%d}$' % (lbl, alg, numc), linewidth=2, markevery=mkr_skip)

    # lg=plt.legend(fontsize=20,ncol=2)
