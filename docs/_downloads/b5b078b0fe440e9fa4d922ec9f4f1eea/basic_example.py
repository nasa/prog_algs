# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

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