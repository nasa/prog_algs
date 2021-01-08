# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

"""
Tool for prognostics algorithm visualization

Matteo Corbetta
matteo.corbetta@nasa.gov
"""

# IMPORT PACKAGES
# =================
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import itertools

from . import utils

# FUNCTIONS
# =========

def state_scatterplot(x, plot_options=None, diag_options=None, **kwargs):

    # Set options for scatterplot
    # -------------------------------------------------------------------------------------------
    if plot_options is None: 
        plot_options = {'alpha': 1.0, 's': 50, 'linewidth': 1}
    if diag_options is None:
        diag_options = {'fill': True}
    
    if kwargs:
        for arg in list(kwargs.keys()):
            if   arg == 'kind':                 kind               = kwargs['kind']
            elif arg == 'diag_kind':            diag_kind          = kwargs['diag_kind']
            elif arg == 'marker':               marker             = kwargs['marker']
            elif arg == 'lowerdiag_only':       lowerdiag_only     = kwargs['lowerdiag_only']
            elif arg == 'overlap_kde':          overlap_kde        = kwargs['overlap_kde']
            elif arg == 'overlap_kde_levels':   overlap_kde_levels = kwargs['overlap_kde_levels']
            elif arg == 'overlap_kde_color':    overlap_kde_color  = kwargs['overlap_kde_color']
    # -------------------------------------------------------------------------------------------

    xdf = pd.DataFrame.from_dict(data=x, orient='columns')
    scatterplot = sns.pairplot(data=xdf, kind=kind, diag_kind=diag_kind, markers=marker,
                               plot_kws=plot_options, diag_kws=diag_options, corner=lowerdiag_only)
    if (type(overlap_kde==bool) and overlap_kde is True) or (type(overlap_kde) == str and 'lower' in overlap_kde):
        scatterplot.map_lower(sns.kdeplot, levels=overlap_kde_levels, color=overlap_kde_color)
    if (type(overlap_kde==bool) and overlap_kde is True) or (type(overlap_kde) == str and 'upper' in overlap_kde):
        scatterplot.map_upper(sns.kdeplot, levels=overlap_kde_levels, color=overlap_kde_color)
    plt.suptitle('State Scatterplot', fontsize=18)
    return 


def plot_state_estimate(states, t, options=None):
    
    state_names, state_values = utils.get_states(states)
    n_states = len(state_names)
    n_samples = len(state_values[state_names[0]])
    state_scatterplot(state_values, kind='scatter', diag_kind='auto', marker='o',
                      lowerdiag_only=False, overlap_kde=False)

    fig = plt.figure()
    nrows, ncols = utils.get_subplot_dim(n_states)
    # Iterate over all states
    for item, state_name in enumerate(state_names):
        ax = fig.add_subplot(nrows, ncols, item + 1)
        ax.plot(state_values[state_name], np.zeros((n_samples,)), '.', markersize=10)
        x, y = utils.get_nonparametric_pdf(state_values[state_name])
        ax.fill_between(x, np.zeros((len(x),)), y, alpha=0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for item, state_name in enumerate(state_names):
        ax.plot(state_values[state_name], np.zeros((n_samples,)), '.', markersize=10)
        x, y = utils.get_nonparametric_pdf(state_values[state_name])
        ax.fill_between(x, np.zeros((len(x),)), y, alpha=0.5)

    plt.show()

    return 



if __name__ == '__main__':

    print("Prognostics Algorithm Visualization Tool")

    

