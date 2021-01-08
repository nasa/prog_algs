# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

import sys
sys.path.insert(1, "../python-prognostics-models-package/")
from prog_models.models.battery_circuit import BatteryCircuit
from prog_algs import *
from prog_algs.visualize.visualize import *
from prog_algs.visualize.utils import get_subplot_dim

import matplotlib.pyplot as plt

def plot_timeseries(s, t=None):
    s_names    = list(s[0][0].keys())
    num_states = len(s_names)
    num_steps  = len(s[0])
    num_samples = len(s)
    nrows, ncols = get_subplot_dim(num_states)
    
    if t is None: # Generate fictitious, normalized time vector
        t = [np.linspace(0., 1., num_steps) for jj in range(num_samples)]

    fig = plt.figure(figsize=(9,9))
    fig.suptitle('State Evolution', fontsize=14)
    idx = 0
    for idx, s_i in enumerate(s_names):
        ax = fig.add_subplot(nrows, ncols, idx+1)
        ax.set_title(s_i, fontsize=14)
        if idx >= ncols * (nrows - 1):    
            ax.set_xlabel('time, s', fontsize=14)
        for sample in range(len(s)):
            ax.plot(t[sample], [s[sample][ii][s_i] for ii in range(num_steps)], '-', linewidth=0.75)
        
    return fig

# def plot_timeseries_v2(t, s, legend=None, options=None):
#     """
#     Plot time series 's' parametrized by time 't'.
#     The function plot time series (in a single plot or subplots) contained in the array of dictionary s, produced by a prognostic model.

#     Input legend, and options are optional (default is None). If provided, they must be dictionaries with options for legend, and 
#     plot options, respectively. 
    
#     The function is capable of plotting time series in a single plot (options['compact']=True), or in multiple subplots in the same figure (options['compact']=False).
#     Legends can be displayed in each subplot or only one subplot, and names of time series, axis labels, plot title, legend titles and other are all customizable.
#     Please read the help of the other functions suggested below for more info.

#     Parameters:
#     -----------
#     t : numpy array or list of floats, time vector
#     s : array of dictionaries. Each entry of the array is a dictionary with all time series, and one value per time series (corresponding to the time instant).
#         Example: s = np.array([ {'x': 0.0,  'v': 1.0}, 
#                                 {'x': 1.0,  'v': 0.9},
#                                 {'x': 1.83, 'v': 0.75},  ])
#     legend : dictionary of legend options. See 'set_legend' function for more details
#     options : dictionary of plot options. See 'set_plot_options' function and other functions therein for more details

#     Returns:
#     --------
#     fig : matplotlib figure object corresponding to the generated figure.
     
#     Example:
#     --------
#     | # New Model Example
#     | # ===============
#     | m = ThrownObject()
#     | 
#     | # Step 2: Setup for simulation 
#     | def future_load(t):
#     |     return {}
#     | 
#     | # Step 3: Simulate to impact
#     | event = 'impact'
#     | (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, {'x':m.parameters['thrower_height']}, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})
#     | 
#     | fig = plot_timeseries(times, states,
#     |                       options = {'compact': False, 'suptitle': 'state evolution', 'title': False,
#     |                                  'xlabel': 'time', 'ylabel': {'x': 'position', 'v': 'velocity'}, 'display_labels': 'minimal'},
#     |                       legend  = {'display': True, 'display_at_subplot': 'all'} )
#     | fig.savefig('example1.png', dpi=300)
#     | fig = plot_timeseries(times, states)
#     | fig.savefig('example2.pdf', dpi=150)
#     """
#     assert type(s) == np.ndarray,   "Time series vector s must be an array of dictionary"
#     assert type(s[0])==dict,        "Every element of the time series vector s must be a dictionary"

#     series_names = list(s[0].keys())
#     m = len(series_names)
#     n = len(s)
    
#     # Set up options
#     # ====================
#     # fig_options    = set_plot_options(options)                      # Set up figure options
#     # legend_options = set_legend_options(legend, series_names)       # Set up legend options
    
#     # Generate figure
#     # =============
#     # if fig_options['figsize'] is not None: fig = plt.figure(figsize=fig_options['figsize'])
#     # else:                                  fig = plt.figure()
#     fig_options = dict(compact=True)
#     fig = plt.figure()
#     if fig_options['compact']:  # Compact option: all time series in one plot
#         # Add plot
#         # --------
#         ax = fig.add_subplot()
#         ax.plot(t, [list(s_i.values()) for s_i in s])

#         # Add options: plot options, title, labels, and legend
#         # ------------------------------------------------------
#         # set_ax_options(ax, fig_options)

#         # if fig_options['title']:
#         #     ax.set_title(fig_options['title'], fontsize=fig_options['title_fontsize'])

#         # if fig_options['display_labels'] or fig_options['display_labels'] != 'no':
#         #     display_labels(1, 1, 0, ax, fig_options, series_names)
        
#         # if legend_options['display']:
#         #     ax.legend(series_names, bbox_to_anchor=legend_options['bbox_to_anchor'], 
#         #               ncol=legend_options['ncol'], fontsize=legend_options['fontsize'],
#         #               fancybox=legend_options['fancybox'], shadow=legend_options['shadow'],
#         #               framealpha=legend_options['framealpha'], facecolor=legend_options['facecolor'],
#         #               edgecolor=legend_options['edgecolor'], title=legend_options['title'])
    
#     else:   # "Not compact" option: one subplot per time series
#         nrows, ncols = get_subplot_dim(m)   # get the number of subplots
        
#         # Iterate over all subplots to plot the time series
#         for item in range(m):
#             ax = fig.add_subplot(nrows, ncols, item+1)                  # add subplot
#             series_ = [s[ii][series_names[item]] for ii in range(n)]    # extract time series data from array of dictionaries
#             ax.plot(t, series_)                                         # add time series to subplot
            
#             # # Add options: display labels, title, legend
#             # # ------------------------------------------
#             # if fig_options['display_labels'] or fig_options['display_labels'] != 'no':
#             #     display_labels(nrows, ncols, item, ax, fig_options, series_names)
            
#             # if fig_options['title']:
#             #     ax.set_title(series_names[item], fontsize=fig_options['title_fontsize'])
            
#             # if legend_options['display']:
#             #     if legend_options['display_at_subplot'] == 'all':               set_legend(ax, item, series_names, legend_options)
#             #     elif legend_options['display_at_subplot'] == item+1:            set_legend(ax, item, series_names, legend_options)
            
#     # Other options
#     # # ==============
#     # if fig_options['suptitle']:     fig.suptitle(fig_options['suptitle'], fontsize=fig_options['title_fontsize']) # Add subtitle
#     # if fig_options['tight_layout']: plt.tight_layout()  # If tight-layout
        
#     return fig


def run_example():
    ## Setup
    def future_loading(t, x = None):
        # Variable (piece-wise) future loading scheme 
        if (t < 600):
            i = 2
        elif (t < 900):
            i = 1
        elif (t < 1800):
            i = 4
        elif (t < 3000):
            i = 2
        else:
            i = 3
        return {'i': i}

    batt = BatteryCircuit()

    ## State Estimation - perform a single ukf state estimate step
    # filt = state_estimators.unscented_kalman_filter.UnscentedKalmanFilter(batt, batt.parameters['x0'])
    filt = state_estimators.particle_filter.ParticleFilter(batt, batt.parameters['x0'])

    print("Prior State:", filt.x.mean)
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])
    t = 0.1
    load = future_loading(t)
    filt.estimate(t, load, {'t': 32.2, 'v': 3.915})
    print("Posterior State:", filt.x.mean)
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])

    ## Prediction - Predict EOD given current state
    # Setup prediction
    mc = predictors.monte_carlo.MonteCarlo(batt)
    if isinstance(filt, state_estimators.unscented_kalman_filter.UnscentedKalmanFilter):
        samples = filt.x.sample(20)
    else: # Particle Filter
        samples = filt.x.raw_samples()
    
    # Predict with a step size of 0.1
    (times, inputs, states, outputs, event_states, eol) = mc.predict(samples, future_loading, dt=0.1)

    ## Print Metrics
    print("\nEOD Predictions (s):")
    from prog_algs.metrics import samples as metrics 
    print('\tPercentage between 3005.2 and 3005.6: ',   metrics.percentage_in_bounds(eol, [3005.2, 3005.6])*100.0, '%')
    print('\tAssuming ground truth 3002.25: ',          metrics.eol_metrics(eol, 3005.25))
    print('\tP(Success) if mission ends at 3002.25: ',  metrics.prob_success(eol, 3005.25))


    fig  = plot_timeseries(states, times)
    # fig2 = plot_timeseries(outputs)


    plt.show()

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()