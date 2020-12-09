# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import sys
sys.path.insert(1, "/Users/cteubert/Desktop/python-prognostics-models-package/")
from prog_models.models import battery_circuit
from prog_algs import *

## Setup
def future_loading(t):
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

batt = battery_circuit.BatteryCircuit()

## State Estimation - perform a single ukf state estimate step

# filt = state_estimators.unscented_kalman_filter.UnscentedKalmanFilter(batt, batt.parameters['x0'])
filt = state_estimators.particle_filter.ParticleFilter(batt, batt.parameters['x0'])

print("Prior State:", filt.x)
print('\tSOC: ', batt.event_state(filt.t, filt.x)['EOD'])
t = 0.1
load = future_loading(t)
filt.estimate(t, load, {'t': 32.2, 'v': 3.915})
print("Posterior State:", filt.x)
print('\tSOC: ', batt.event_state(filt.t, filt.x)['EOD'])

## Prediction - Predict EOD given current state
mc = predictors.monte_carlo.MonteCarlo(batt)
if isinstance(filt, state_estimators.unscented_kalman_filter.UnscentedKalmanFilter):
    state_sampler = samplers.generate_mean_cov_random_sampler(batt.states, list(filt.x.values()), filt.Q)
    prediction_config = {'dt': 0.025, 'num_samples':5}
else: # Particle Filter
    def state_sampler(num_samples):
        return filt.particles
    prediction_config = {'dt': 0.1, 'num_samples':len(filt.particles)}
    
(times, inputs, states, outputs, event_states, eol) = mc.predict(state_sampler, future_loading, prediction_config)

## Print Metrics
print("\nEOD Predictions (s):")
from prog_algs.metrics import samples as metrics 
print('\tPercentage between 3005.2 and 3005.6: ', metrics.percentage_in_bounds(eol, [3005.2, 3005.6])*100.0, '%')
print('\tAssuming ground truth 3002.25: ', metrics.eol_metrics(eol, 3005.25))
print('\tP(Success) if mission ends at 3002.25: ', metrics.prob_success(eol, 3005.25))