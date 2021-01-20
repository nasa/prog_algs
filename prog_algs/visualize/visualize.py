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

def plot_hist(x, options=None):
    n_hists = len(x)
    if options is None:
        options = dict(name='pdf', nbins=10, fs=16, disp_mean_std=True, show_density=True,
                            transparency=0.75, orientation='vertical', xlabel='x, -', ylabel='probability density, -',
                            log=False, histtype='bar', align='mid', cumulative=False, weights=None)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for h in range(n_hists):
        x_samps = x[h]
        labeltxt = r'$\mathrm{' + str(options['name']) + '}_{n_{'+ str(h) + r'}}$'
        if options['disp_mean_std']:       
            m, s = np.round(np.mean(x_samps), 3), np.round(np.std(x_samps), 3)
            labeltxt += '\n$\mu=$' + str(m) + ',$\sigma=$' + str(s)
        ax.hist(x_samps, bins=options['nbins'], density=options['show_density'], weights=options['weights'],
                alpha=options['transparency'], cumulative=options['cumulative'], histtype=options['histtype'],
                align=options['align'], orientation=options['orientation'], log=options['log'], label=labeltxt)
    ax.set_xlabel(options['xlabel'], fontsize=options['fs'])
    ax.set_ylabel(options['ylabel'], fontsize=options['fs'])
    plt.legend(fontsize=options['fs']-3, fancybox=True, shadow=True, bbox_to_anchor=(0.75, 0.95))
    return fig


def state_scatterplot(x, time_steps=0, **kwargs):
    
    _, x = utils.get_states(x, step=time_steps)

    # Set default options
    # -------------------------------------------------------------------------------------------
    config                       = {}
    config['kind']               = 'scatter'
    config['diag_kind']          = 'hist'
    config['marker']             = 'o'
    config['lowerdiag_only']     = False
    config['overlap_kde']        = False
    config['overlap_kde_levels'] = False
    config['overlap_kde_color']  = False
    config['plot_options']       = {'alpha': 1.0, 's': 50, 'linewidth': 1}
    config['diag_options']       = {'fill': True}
    config.update(kwargs)
    # -------------------------------------------------------------------------------------------

    scatterplot = sns.pairplot(data=pd.DataFrame.from_dict(data=x, orient='columns'),
                               kind=config['kind'], diag_kind=config['diag_kind'], markers=config['marker'],
                               plot_kws=config['plot_options'], diag_kws=config['diag_options'], corner=config['lowerdiag_only'])
    if (type(config['overlap_kde']==bool) and config['overlap_kde'] is True) or \
        (type(config['overlap_kde']) == str and 'lower' in config['overlap_kde']):
        scatterplot.map_lower(sns.kdeplot, levels=config['overlap_kde_levels'], color=config['overlap_kde_color'])
    if (type(config['overlap_kde']==bool) and config['overlap_kde'] is True) or \
        (type(config['overlap_kde']) == str and 'upper' in config['overlap_kde']):
        scatterplot.map_upper(sns.kdeplot, levels=config['overlap_kde_levels'], color=config['overlap_kde_color'])
    plt.suptitle('State Scatterplot', fontsize=18)
    return scatterplot


def plot_state_estimate(states, time_step=0, options=None):
    
    state_names, state_values = utils.get_states(states, step=time_step)
    
    n_states = len(state_names)
    n_samples = len(state_values[state_names[0]])

    state_scatterplot(states, time_step=time_step, kind='scatter', diag_kind='auto', marker='o',
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


def plot_timeseries(t, s, legend=None, options=None):
    """
    Plot time series 's' parametrized by time 't'.
    The function plot time series (in a single plot or subplots) contained in s.
    The function can handle both the output of a prognostic model, an array of dictionaries, or the output of a prognostic algorithm, 
    an array of array of dictionaries. The two are organized as follows:
    prognostic-model output:            s[ time ][ state ]
    prognostic-algorithm output:        s[ sample ][ time ][ state]
    The function handles the two structures automatically.

    Input legend, and options are optional (default is None). If provided, they must be dictionaries with options for legend, and 
    plot options, respectively. 
    
    The function is capable of plotting time series in a single plot (options['compact']=True), or in multiple subplots in the same figure (options['compact']=False).
    Legends can be displayed in each subplot or only one subplot, and names of time series, axis labels, plot title, legend titles and other are all customizable.
    Please read the help of the other functions suggested below for more info.

    Parameters:
    -----------
    t : numpy array or list of floats, time vector
    s : array of dictionaries. Each entry of the array is a dictionary with all time series, and one value per time series (corresponding to the time instant).
        Example: s = np.array([ {'x': 0.0,  'v': 1.0}, 
                                {'x': 1.0,  'v': 0.9},
                                {'x': 1.83, 'v': 0.75},  ])
    legend : dictionary of legend options. See 'set_legend' function for more details
    options : dictionary of plot options. See 'set_plot_options' function and other functions therein for more details

    Returns:
    --------
    fig : matplotlib figure object corresponding to the generated figure.
     
    Example:
    --------
    | # New Model Example
    | # ===============
    | m = ThrownObject()
    | 
    | # Step 2: Setup for simulation 
    | def future_load(t):
    |     return {}
    | 
    | # Step 3: Simulate to impact
    | event = 'impact'
    | (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
    | 
    | fig = plot_timeseries(times, states,
    |                       options = {'compact': False, 'suptitle': 'state evolution', 'title': False,
    |                                  'xlabel': 'time', 'ylabel': {'x': 'position', 'v': 'velocity'}, 'display_labels': 'minimal'},
    |                       legend  = {'display': True, 'display_at_subplot': 'all'} )
    | fig.savefig('example1.png', dpi=300)
    | fig = plot_timeseries(times, states)
    | fig.savefig('example2.pdf', dpi=150)
    """
    assert type(s) == np.ndarray,   "Time series vector s must be an array of dictionary or array of array of dictionary"
    # assert type(s[0])==dict,        "Every element of the time series vector s must be a dictionary"
    
    if type(s[0]) == dict:
        samples      = False
        series_names = list(s[0].keys())
    elif type(s[0]) == np.ndarray:
        samples      = True
        num_samples  = len(s)
        num_steps    = len(s[0])
        series_names = list(s[0][0].keys())
    else:
        raise Exception("Structure of s not recognized.")

    m = len(series_names)
    n = len(s)
    
    # Set up options
    # ====================
    fig_options    = utils.set_plot_options(options)                      # Set up figure options
    legend_options = utils.set_legend_options(legend, series_names)       # Set up legend options
    
    # Generate figure
    # =============
    if fig_options['figsize'] is not None: fig = plt.figure(figsize=fig_options['figsize'])
    else:                                  fig = plt.figure()
    
    if fig_options['compact']:  # Compact option: all time series in one plot
        # Add plot
        # --------
        ax = fig.add_subplot()
        if samples:
            for s_i in series_names:
                for samp in range(num_samples):
                    ax.plot(t[samp], [s[samp][ii][s_i] for ii in range(num_steps)], linewidth=0.75)
        else:
            ax.plot(t, [list(s_i.values()) for s_i in s])

        # Add options: plot options, title, labels, and legend
        # ------------------------------------------------------
        utils.set_ax_options(ax, fig_options)

        if fig_options['title']:
            ax.set_title(fig_options['title'], fontsize=fig_options['title_fontsize'])

        if fig_options['display_labels'] or fig_options['display_labels'] != 'no':
            utils.display_labels(1, 1, 0, ax, fig_options, series_names)
        
        if legend_options['display']:
            ax.legend(series_names, bbox_to_anchor=legend_options['bbox_to_anchor'], 
                      ncol=legend_options['ncol'], fontsize=legend_options['fontsize'],
                      fancybox=legend_options['fancybox'], shadow=legend_options['shadow'],
                      framealpha=legend_options['framealpha'], facecolor=legend_options['facecolor'],
                      edgecolor=legend_options['edgecolor'], title=legend_options['title'])
    
    else:  # "Not compact" option: one subplot per time series
        nrows, ncols = utils.get_subplot_dim(m)   # get the number of subplots
        # Iterate over all subplots to plot the time series
        for item in range(m):
            ax = fig.add_subplot(nrows, ncols, item+1)                  # add subplot
            if samples:
                for sample in range(num_samples):
                    ax.plot(t[sample], [s[sample][ii][series_names[item]] for ii in range(num_steps)], '-', linewidth=0.75)
            else:
                series_ = [s[ii][series_names[item]] for ii in range(n)]    # extract time series data from array of dictionaries
                ax.plot(t, series_)                                         # add time series to subplot
            
            # Add options: display labels, title, legend
            # ------------------------------------------
            if fig_options['display_labels'] or fig_options['display_labels'] != 'no':
                utils.display_labels(nrows, ncols, item, ax, fig_options, series_names)
            
            if fig_options['title']:
                ax.set_title(series_names[item], fontsize=fig_options['title_fontsize'])
            
            if legend_options['display']:
                if   legend_options['display_at_subplot'] == 'all':     utils.set_legend(ax, item, series_names, legend_options)
                elif legend_options['display_at_subplot'] == item+1:    utils.set_legend(ax, item, series_names, legend_options)
            
    # Other options
    # ==============
    if fig_options['suptitle']:     fig.suptitle(fig_options['suptitle'], fontsize=fig_options['title_fontsize']) # Add subtitle
    if fig_options['tight_layout']: plt.tight_layout()  # If tight-layout
        
    return fig


if __name__ == '__main__':

    print("Prognostics Algorithm Visualization Tool")

    

