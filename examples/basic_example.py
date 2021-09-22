# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from prog_models.models import BatteryCircuit
from prog_algs import *

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
    # filt = state_estimators.UnscentedKalmanFilter(batt, batt.parameters['x0'])
    filt = state_estimators.ParticleFilter(batt, batt.parameters['x0'])
    print("Prior State:", filt.x.mean)
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])
    t = 0.1
    load = future_loading(t)
    filt.estimate(t, load, {'t': 32.2, 'v': 3.915})
    print("Posterior State:", filt.x.mean)
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])

    ## Prediction - Predict EOD given current state
    # Setup prediction
    mc = predictors.MonteCarlo(batt)
    if isinstance(filt, state_estimators.UnscentedKalmanFilter):
        samples = filt.x.sample(20)
    else: # Particle Filter
        samples = filt.x.raw_samples()

    # Predict with a step size of 0.1
    (times, inputs, states, outputs, event_states, eol) = mc.predict(samples, future_loading, dt=0.1)

    # The results of prediction can be accessed by sample, e.g.,
    times_sample_1 = times[1]
    states_sample_1 = states[1]
    # now states_sample_1[n] corresponds to time_sample_1[n]
    # you can also plot the results (state_sample_1.plot())

    # You can also access a state at a specific time using the .snapshot function
    states_time_1 = states.snapshot(1)
    # now you have all the samples from the times[sample][1]
    
    ## Print Metrics
    print("\nEOD Predictions (s):")
    from prog_algs.metrics import samples as metrics 
    print('\tPercentage between 3005.2 and 3005.6: ', metrics.percentage_in_bounds(eol, [3005.2, 3005.6])*100.0, '%')
    print('\tAssuming ground truth 3002.25: ', metrics.eol_metrics(eol, 3005.25))
    print('\tP(Success) if mission ends at 3002.25: ', metrics.prob_success(eol, 3005.25))

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()