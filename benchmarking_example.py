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

##  Setup State Estimation 
filt = state_estimators.unscented_kalman_filter.UnscentedKalmanFilter(batt, future_loading, batt.parameters['x0'])

## Setup Prediction
mc = predictors.monte_carlo.MonteCarlo(batt)
state_sampler = samplers.generate_mean_cov_random_sampler(batt.states, list(filt.x.values()), filt.Q)
prediction_config1 = {'dt': 0.05, 'num_samples':2}
prediction_config2 = {'dt': 0.05, 'num_samples':5}
prediction_config3 = {'dt': 0.05, 'num_samples':10}

# Playback 
from prog_algs.metrics import samples as metrics 
print('Run 1 ({} samples)'.format(prediction_config1['num_samples']))
(t, u, x, z, es, eol) = run_prog_playback(filt, mc, state_sampler, future_loading, [(0.1, {'t': 32.2, 'v': 3.915}), (0.1, {'t': 32.3, 'v': 3.91})], {'predict_config': prediction_config1})
print('MSE: ', metrics.mean_square_error(eol, 3005.4))
prediction_times = [0.1, 0.2]
print('Alpha-lambda met: ', metrics.alpha_lambda(prediction_times, eol, 3005.4, 0.2, 1e-4, 0.65))

print('Run 2 ({} samples)'.format(prediction_config2['num_samples']))
(t, u, x, z, es, eol) = run_prog_playback(filt, mc, state_sampler, future_loading, [(0.1, {'t': 32.2, 'v': 3.915}), (0.1, {'t': 32.3, 'v': 3.91})], {'predict_config': prediction_config2})
print('MSE: ', metrics.mean_square_error(eol, 3005.4))
prediction_times = [0.1, 0.2]
print('Alpha-lambda met: ', metrics.alpha_lambda(prediction_times, eol, 3005.4, 0.2, 1e-4, 0.65))

print('Run 3 ({} samples)'.format(prediction_config3['num_samples']))
(t, u, x, z, es, eol) = run_prog_playback(filt, mc, state_sampler, future_loading, [(0.1, {'t': 32.2, 'v': 3.915}), (0.1, {'t': 32.3, 'v': 3.91})], {'predict_config': prediction_config3})
print('MSE: ', metrics.mean_square_error(eol, 3005.4))
prediction_times = [0.1, 0.2]
print('Alpha-lambda met: ', metrics.alpha_lambda(prediction_times, eol, 3005.4, 0.2, 1e-4, 0.65))