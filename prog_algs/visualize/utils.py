# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.


# Import packages
# ============
import numpy as np
# from numpy import ndarray
import itertools
from scipy.stats import gaussian_kde as kde

import matplotlib as mpl
import matplotlib.pyplot as plt

# Set default options
# ====================
mpl.rcParams['lines.linewidth']  = 3
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['font.size']        = 14
mpl.rcParams['axes.labelsize']   = 'x-large'
mpl.rcParams['legend.fontsize']  = 'large'
mpl.rcParams['figure.titlesize'] = 'x-large'
mpl.rcParams['figure.figsize']   = [10.0, 9.0]
mpl.rcParams['figure.dpi']       = 100
mpl.rcParams['savefig.dpi']      = 300


# VISUALIZE FUNCTIONS
# ==========================
def get_subplot_dim(num_subplots, rowfirst=True):
    """
    Compute the number of rows and columns (nrows, ncols) for a figure with multiple subplots.
    The function returns number of rows and columns given num_subplots. 
    Those numbers are computed sequentially until nrows * ncols >= num_subplots.
    By default, the function adds a new row first if the number of subplots has not been reached, then adds a new column.
    By passing rowfirst=False, the function will add a new column first if the number of subplot has not been reached, then a new row.

    nrows and ncols are initialized to 1. If num_subplots==1, then subplots are not needed, and the function returns nrows=ncols=1.
    The command fig.add_subplot(nrows,ncols,1) generates a normal plot (no subplots).

    Parameters:
    -----------
    num_subplots : int, number of subplots the figure should contain
    rowfirst     : Boolean, whether to add a new row first or a new column first to increase the number of rows and columns if necessary.
                   Default is rowfirst=True.
    
    Returns:
    --------
    nrows : int, number of subplots along the rows (vertical axis) of the figure
    ncols : int, number of subplots along the columns (horizontal axis) of the figure

    Example:
    --------
    | states = np.random.randn(1000,5) # let us consider a state vector with 5 dimensions, and 1000 values of the states (one for each time step)
    | n_states = states.shape[-1]     # get the number of states (5)
    | print(get_subplot_dim(n_states)) # 3, 2
    | print(get_subplot_dim(n_states, rowfirst=False)) # 2, 3
    | 
    | fig = plt.figure()
    | ax = fig.add_subplot(nrows, ncols, 0)
    | # ...
    """
    nrows, ncols = 1, 1 # initialize number of rows and cols to 1.
    if rowfirst:
        while nrows * ncols < num_subplots:         
            nrows += 1
            if nrows * ncols < num_subplots:        
                ncols += 1
    else:
        while nrows * ncols < num_subplots:         
            ncols += 1
            if nrows * ncols < num_subplots:        
                nrows += 1
    return nrows, ncols


def set_plot_options(opt):
    """
    Set default plot options by integrating the options specified by the user in 'opt'
    The visualize library works with specific values to generate the plots.
    if those are not specified by the users, this function assign them their default values.

    Parameters:
    -----------
    opt : dictionary of plot options. Acceptable entries are:
          'figsize' : tuple of 2 floats, width and height of the figure
          'compact' : Boolean, whether to plot a "compact" figure. If compact, all time series are displayd in one plot (multiple colored lines)
          'xlabel'  : string, label for the x-axis. Default is 'time'
          'ylabel'  : string, label for the y-axis. Default is 'state' 
          'title'   : string or empty list or None, plot title. Default is empty list (no title)
          'title_fontsize' : string or float, plot title fontsixe. Default is 'x-large'
          'suptitle'       : string or empty list or None, plot suptitle. Default is empty list (no suptitle)
          'ticklabel_fontsize' : string or float, tick label font sizes. Defaullt is 'small'
          'tight_layout' : Boolean, whether to use tight layout (minimize figure blank space around the graph)
          'display_labels' : string, whether to display x and y-labels in the figure.

    Returns:
    --------
    opt : dictionary of plot options with default values added.

    Example:
    -------
    | opt = {}
    | opt = set_plot_options(opt)
    | print(opt)  # opt['figsize'] = (10, 9), opt['compact'] = True, opt['xlabel'] = 'time', ....
    """
    # Set up plot options
    # =======================
    # Get list of options (if provided)
    try:        opt_list = list(opt.keys())
    except:     opt, opt_list = {}, []
    
    # Fill out all options if not provided
    if 'figsize' not in opt_list:               opt['figsize'] = (10, 9)
    if 'compact' not in opt_list:               opt['compact'] = True
    if 'xlabel' not in opt_list:                opt['xlabel'] = 'time'
    if 'ylabel' not in opt_list:                opt['ylabel'] = 'state'
    if 'title' not in opt_list:                 opt['title'] = []
    if 'title_fontsize' not in opt_list:        opt['title_fontsize'] = 'x-large'
    if 'suptitle' not in opt_list:              opt['suptitle'] = []
    if 'ticklabel_fontsize' not in opt_list:    opt['ticklabel_fontsize'] = 'small'
    if 'tight_layout' not in opt_list:          opt['tight_layout'] = False
    if 'display_labels' not in opt_list:        opt['display_labels'] = 'all'
    
    # if title should be displayed but title fontsize is not specified, add it to the dictionary
    if opt['title'] and ('title_fontsize' not in opt_list or not opt['title_fontsize']):
        if 'fontsize' in opt_list:      opt['title_fontsize'] = opt['fontsize']
        else:                           opt['title_fontsize'] = 'x-large'
    # if xlabel or ylabel should be displayed but their fontsize is not specified, add it to the dictionary
    if opt['xlabel'] or opt['ylabel']:
        if 'label_fontsize' not in opt_list or not opt['label_fontsize']:
            if 'fontsize' in opt_list:       opt['label_fontsize'] = opt['fontsize']
            else:                            opt['label_fontsize'] = 'x-large'
    # if xticks are not default but rotation and fontsize are not specified, add them to the dictionary
    if 'xticks' in opt_list:         
        if 'xtick_rotation' not in opt_list:        opt['xtick_rotation'] = 0
        if 'tick_fontsize' not in opt_list:         opt['xtick_fontsize'] = 'large'
        else:                                       opt['xtick_fontsize'] = opt['tick_fontsize']
    else:
        opt['xticks'] = []
    # if yticks are not default but rotation and fontsize are not specified, add them to the dictionary
    if 'yticks' in opt_list:         
        if 'ytick_rotation' not in opt_list:        opt['ytick_rotation'] = 0
        if 'tick_fontsize' not in opt_list:         opt['ytick_fontsize'] = 'large'
        else:                                       opt['ytick_fontsize'] = opt['tick_fontsize']
    else:
        opt['yticks'] = []

    return opt


def set_legend_options(leg_opt, s_names):
    """
    Set all remaining legend options given the legend options already specified by the users "leg_opt", and the names
    of the time series in the plot "s_names."
    
    The visualize library works with specific values for some legend options.
    if those are not specified by the users, this function assign them their default values.

    Parameters:
    -----------
    leg_opt : dictionary of entries for the legend specified by the user. 
    s_names : list of strings, names of the time series in the current plot, whose names should appear in the legend.

    Returns:
    --------
    leg_opt : dictionary of enetries for the legend.

    Example:
    --------
    | s_names = ['x', 'v']
    | leg_opt = {'display': True, 'labels': None, 'loc'='best', 'fontsize', 14}
    | leg_opt = set_legend_options(leg_opt, s_names)
    | print(leg_opt['labels']) # ['x', 'v']
    | print(leg_opt['fancybox']) # False
    | print(leg_opt['facecolor']) # 'w'
    """
    try:        leg_list = list(leg_opt.keys())     # Check whether a dictionary has been provided. If not, initialize the dictionary leg_opt as empty
    except:     leg_opt, leg_list = {}, []
    if 'display' not in leg_list:               leg_opt['display'] = False
    if 'display_at_subplot' not in leg_list:    leg_opt['display_at_subplot'] = len(s_names)
    if 'labels' not in leg_list:                leg_opt['labels'] = s_names
    if 'loc' not in leg_list:                   leg_opt['loc'] = 'best'
    if 'bbox_to_anchor' not in leg_list:        leg_opt['bbox_to_anchor'] = None
    if 'ncol' not in leg_list:                  leg_opt['ncol'] = 1
    if 'fontsize' not in leg_list:              leg_opt['fontsize'] = 'x-large'
    if 'shadow' not in leg_list:                leg_opt['shadow'] = False
    if 'fancybox' not in leg_list:              leg_opt['fancybox'] = False
    if 'framealpha' not in leg_list:            leg_opt['framealpha'] = 1.0
    if 'facecolor' not in leg_list:             leg_opt['facecolor'] = 'w'
    if 'edgecolor' not in leg_list:             leg_opt['edgecolor'] = 'w'
    if 'title' not in leg_list:                 leg_opt['title'] = None
    if 'title' in leg_list:                 
        if 'title_fontsize' not in leg_list:    leg_opt['title_fontsize'] = 'medium'

    return leg_opt


def set_savefig_options(sfo):
    """
    Set all remaining save figure options given the options already specified by the user "sfo".

    Parameters:
    -----------
    sfo : dictionary of options to save the current figure
          if not provided, a figure will not be saved by default. Otherwise, sfo should provide the following dictionary entries:
          'save' (Boolean, whether to save the figure or not, True or False) 
          'dpi' (int, resolution of the figure in dots per inch)
          'filename' (string, figure filename, including file type, e.g., .pdf or .png. Default is 'timeseries_plot.pdf')
    
    Returns:
    --------
    sfo  : dictionary of default save figure options.

    Example:
    --------
    | fig = plt.figure()
    | ax = fig.add_subplot(111)
    | ax.plot([0, 1], [3, 4])
    | sfo = {'save': True}    # a figure has to be saved, but no dpi nor filename for the figure has been specified in sfo
    | sfo = set_savefig_options(sfo)
    | print(sfo)  # sfo = {'save': True, 'dpi': 300, 'filename': 'timeseries_plot.pdf'}
    """
    try:        sfo_list = list(sfo.keys())
    except:     sfo, sfo_list = {}, []
    if 'save' not in sfo_list:                                  sfo['save'] = False
    if 'dpi' not in sfo_list:                                   sfo['dpi'] = 300
    if 'save' in sfo_list and 'filename' not in sfo_list:       sfo_list['filename'] = 'timeseries_plot.pdf'
    return sfo


def set_legend(ax, item, s_names, leg_opt):
    """
    Set legend for axis 'ax' for the 'item-th' time series entry. All time series labels are defined in 's_names'.
    For a comprehensive explanation of all legend options, see the Matplotlib guide on their website.

    Parameters:
    ----------
    ax      : matplotlib axis object
    item    : int, index of the time series to be displayed in the legend
    s_names : list of strings, names of all time series in the plot or subplot
    leg_opt : dictionary containing all legend options necessary to place and modify the legend. Dictionary entries can be:
              'bbox_to_anchor' (tuple, coordinates of the legend location), 
              'ncol' (int, number of columns of the legend),
              'fontsize' (int, legend font size),
              'fancybox' (Boolean, Use a fancybox for the legend, True or False),
              'shadow' (Boolean, Whether the legend box should have a shadow, True or False)
              'facecolor' (string with color code, background color of the legend box),
              'edgecolor' (string with color code, edge color of the legend box),
              'title' (string, legend title)
    
    Returns:
    --------
    axis object with legend

    Example:
    -------
    | s_names = list(s.keys())
    | ax = fig.add_subplot()
    | ax.plot(t, [list(s_i.values()) for s_i in s])
    | 
    | leg_opt = {}
    | leg_opt['labels'] = s_names
    | leg_opt['loc'] = 'best'
    | leg_opt['bbox_to_anchor'] = None
    | leg_opt['ncol'] = 1
    | leg_opt['fontsize'] = 'x-large'
    | leg_opt['shadow'] = False
    | leg_opt['fancybox'] = False
    | leg_opt['framealpha'] = 1.0
    | leg_opt['facecolor'] = 'w'
    | leg_opt['edgecolor'] = 'w'
    | leg_opt['title'] = None
    | 
    | ax.legend(series_names, bbox_to_anchor=legend_options['bbox_to_anchor'], 
    |           ncol=legend_options['ncol'], fontsize=legend_options['fontsize'],
    |           fancybox=legend_options['fancybox'], shadow=legend_options['shadow'],
    |           framealpha=legend_options['framealpha'], facecolor=legend_options['facecolor'],
    |           edgecolor=legend_options['edgecolor'], title=legend_options['title'])
    """
    return ax.legend(s_names[item], bbox_to_anchor=leg_opt['bbox_to_anchor'], 
                     ncol=leg_opt['ncol'], fontsize=leg_opt['fontsize'],
                     fancybox=leg_opt['fancybox'], shadow=leg_opt['shadow'],
                     framealpha=leg_opt['framealpha'], facecolor=leg_opt['facecolor'],
                     edgecolor=leg_opt['edgecolor'], title=leg_opt['title'])


def display_labels(nrows, ncols, subplot_num, ax, opt, series_names):
    """
    Display label option for time series plot

    Parameters:
    -----------
    nrows        : int, number of subplot rows in plot
    ncols        : int, number of subplot columns in plot
    subplot_num  : int, subplot number
    ax           : matplotlib axis object, current axis
    opt          : dict, display options for the plot.
                   minimum options to be included are:
                   1. 'display_labels' = 'minimal' or 'all'. if 'minimal', only the minimum number of axis ticks and labels are displayed according
                   to the number of subplots. If 'all', then all x-ticks, x-labels, y-ticks, and y-labels will be displayed for each subplot.
                   2. 'xlabel' and 'ylabel'. strings containing the xlabels and ylabels to display.
    series_names : list of strings, names of the time series

    Returns:
    --------
    Nothing, the function modifies current axis "ax"

    Example:
    --------
    | nrows, ncols = 2, 1
    | subplot_num = 0
    | fig = plt.figure()
    | ax = fig.add_subplot(nrows, ncols, subplot_num+1)
    | opt = {'xlabel': 'time', 'ylabel': 'state', 'display_labels': 'all'}
    | series_names= ['x', 'v']
    | display_labels(nrows, ncols, subplot_num, ax, opt, series_names)
    """
    if 'minimal' in opt['display_labels']:
        if subplot_num+1 == nrows*ncols and opt['xlabel']:         set_labels(ax, opt, series_names, axis='x')
        if (subplot_num+1 == 1 or ncols==1) and opt['ylabel']:     set_labels(ax, opt, series_names, axis='y')
        if subplot_num+1 == 1:                                     ax.set_xticks([], [])    # If 'display_labels' is minimal, kill xticks that are not needed according to subplots
    elif 'all' in opt['display_labels']:    
        set_labels(ax, opt, series_names)
    return


def extract_option(opt, idx, series_names):
    """
    Extract option from either dictionary or list of plot options.
    The function takes the option "opt" and returns the option at index "idx" if opt is a list,
    the option corresponding to the series name at index idx, "series_names[idx]" if opt is a dictionary,
    or return the option "opt" if opt is neither a dictionary or a list.

    Parameters:
    -----------
    opt             :   dictionary of strings (label names) corresponding to entries in series_names, list of strings corresponding to the series names (in order), or a simple string
    idx             :   int, index of the option to extract in label options if opt is a string, or index of the corresponding series_names to extract if opt is a dictionary
    series_names    :   list of strings, series names used if opt is a dictionary

    Returns:
    --------
    string, option corresponding to the index idx (if opt is a list of strings), corresponding to the series_names[idx] if opt is dictionary, or simply return opt if neither of those

    Examples:
    --------
    | Example 1:
    | ...........
    | opt = {'ylabel': {'x': 'position', 'v': 'velocity}}
    | series_names = ['x', 'v']
    | print(extract_option(opt['ylabel'], 0, series_names)) # 'position'
    | print(extract_option(opt['ylabel'], 1, series_names)) # 'velocity'
    | 
    | Example 2:
    | ..........
    | opt = {'ylabel': ['state value 1', 'state value 2'] }
    | print(extract_option(opt['ylabel'], 0, series_names)) # 'state value 1', please note that series_names is ignored
    | print(extract_option(opt['ylabel'], 1, series_names)) # 'state value 2'
    | 
    | Example 3:
    | ..........
    | opt = {'ylabel': 'state output'}
    | series_names = ['x', 'v']
    | print(extract_option(opt['ylabel'], 0, series_names)) # 'state output', please note that idx and series_names are ignored
    | print(extract_option(opt['ylabel'], 1, {})) # 'state output', please note that idx and series_names are ignored
    | print(extract_option(opt['ylabel'], np.inf, [])) # 'state output', please note that idx and series_names are ignored
    """
    if type(opt) == dict:                        return opt[series_names[idx]]
    if type(opt) == list and (len(opt)>0):       return opt[idx]
    else:                                        return opt


def set_ax_options(ax, opts):
    """
    Set label options for plot axis.

    The function set a number of labels defined in dictionary "opts" for the current axis "ax".
    At the moment, the options that can be set are:
        title, title_fontsize, xlabel, ylabel, xticks, xtick_rotation, xtick_fontsize, tticks, ytick_rotation, ytick_fontsize
    Input options are not mandatory, so only a subset of them can be defined.

    Parameters:
    -----------
    ax : matplotlib axis object
        matplotlib axis to visualize in figure
    opts :  dictionary
        display options for the axis / subplot. Available: title, title_fontsize, xlabel, ylabel, xticks, xtick_rotation, xtick_fontsize, tticks, ytick_rotation, ytick_fontsize
    
    Returns:
    -------
    Nothing. This function does not return anything as the input ax object is modified.

    Example:
    -------
    | opts = {'title': 'this is a test title', 
    |         'title_fontsize': 16, 'xlabel': 'time', 'ylabel': 'state value',
    |         'xtick_rotation': 45, 'xtick_fontsize': 14, 'ytick_rotation': 0, 'ytick_fontsize': 14}
    | fig = plt.figure()
    | ax = fig.add_subplot(111)
    | set_ax_options(ax, opts)
    """
    if opts['title']:   ax.set_title(opts['title'], fontsize=opts['title_fontsize'])
    if opts['xlabel']:  ax.set_xlabel(opts['xlabel'])
    if opts['ylabel']:  ax.set_ylabel(opts['ylabel'])
    if opts['xticks']:  ax.set_xticklabels(opts['xticks'], rotation=opts['xtick_rotation'], fontsize=opts['xtick_fontsize'])
    if opts['yticks']:  ax.set_yticklabels(opts['yticks'], rotation=opts['ytick_rotation'], fontsize=opts['ytick_fontsize'])
    return 


def set_labels(ax, opt, series_names, axis='all'):
    """
    Set labels of axis "ax" according to figure options "opt" and the time series names "series_names."
    The function can set both x and y axis when input axis=='all' (default), or rather set only x or y axis (axis='x' or axis='y', respectively).

    Parameters:
    -----------
    ax : matplotlib axis object
        axis to add labels and tick options to
    opt : dictionary of label options
        options must include labels and ticks for both x and y axes; 'xlabel', 'xticks', 'ylabel', 'yticks'
        if 'xticks' is not empty (or None), then options can include also 'xtick_rotation', to rotate the axis ticks w.r.t. the plot, as well as 'xtick_fontsize', to change tick fontsize.
        the same applies to 'yticks'.
        
    series_names : list of strings
        name of time series
    axis : string to decide which axis to display the options on.
           options are: 'x', 'y', or 'all' for both x and y
    
    Returns:
    --------
    Nothing. This function modifies the ax object without returning anything.

    Example:
    -------
    | fig = plt.figure()
    | ax = fig.add_subplot(221)
    | series_names = ['x', 'y']
    | opt = {'xlabel': 'time', 'ylabel': 'state value', 'xticks': ['-\pi', '0', '\pi'], 'xtick_fontsize', 12, 'xtick_rotation', 90,
    |        'yticks': ['-\pi', '0', '\pi'], 'ytick_fontsize', 12, 'ytick_rotation', -90}
    | set_labels(ax, opt, series_names, axis='all')
    """
    idx = ax._subplotspec.colspan[0] + ax._subplotspec.rowspan[0]   # Extract the index of the current subplot
    if axis=='all' or axis=='x':    # add properties to x-axis
        xlabel    = extract_option(opt['xlabel'], idx, series_names)
        xtick     = extract_option(opt['xticks'], idx, series_names)
        ax.set_xlabel(xlabel)
        if xtick:       # if xtick options are passed, add them to the axis
            xtick_rot = extract_option(opt['xtick_rotation'], idx, series_names)
            xtick_fs  = extract_option(opt['xtick_fontsize'], idx, series_names)
            ax.set_xticklabels(xtick, rotation=xtick_rot, fontsize=xtick_fs)
    if axis=='all' or axis=='y':    # add properties to y-axis
        ylabel    = extract_option(opt['ylabel'], idx, series_names)
        ytick     = extract_option(opt['yticks'], idx, series_names)
        ax.set_ylabel(ylabel)
        if ytick:      # if ytick options are passed, add them to the axis
            ytick_rot = extract_option(opt['ytick_rotation'], idx, series_names)
            ytick_fs  = extract_option(opt['ytick_fontsize'], idx, series_names)
            ax.set_yticklabels(ytick, rotation=ytick_rot, fontsize=ytick_fs)
    return 


def remove_tuple_duplicates(l):
    new_l = []
    for item in l:
        if item[0] == item[1]:      continue
        else:                       new_l.append(item)
    return new_l


def get_states(states):
    n_samples    = len(states.raw_samples())
    state_names  = list(states.raw_samples()[0].keys())
    state_values = dict.fromkeys(state_names)
    for state_name in state_names:
        state_values[state_name] = [states.raw_samples()[ii][state_name] for ii in range(n_samples)]
    return state_names, state_values

def zip_2d_list(l):
    tuple_list = remove_tuple_duplicates([el for el in itertools.product(l, l)])
    seen_already, zipped_2d_list = set(), []
    for el in tuple_list:
        el_sort = tuple(sorted(el))
        if el_sort not in seen_already:
            zipped_2d_list.append(el)
            seen_already.add(el_sort)
    return zipped_2d_list

def get_nonparametric_pdf(vals, bandwidth_type=None, sample_weights=None):
    pdf_estimator = kde(vals, bw_method=bandwidth_type, weights=sample_weights)
    x             = np.sort(vals)
    pd            = pdf_estimator(x)
    return x, pd