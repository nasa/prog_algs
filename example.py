# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import sys
sys.path.insert(1, "/Users/cteubert/Desktop/python-prognostics-models-package/")
from prog_models.models import battery_circuit
from prog_algs.state_estimators import unscented_kalman_filter, particle_filter

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

# filt = unscented_kalman_filter.UnscentedKalmanFilter(batt, future_loading, batt.parameters['x0'])
filt = particle_filter.ParticleFilter(batt, future_loading, batt.parameters['x0'])

print("Prior State:", filt.x)
print('\tSOC: ', batt.event_state(filt.t, filt.x)['EOD'])
filt.estimate(0.1, {'t': 32.2, 'v': 3.915})
print("Posterior State:", filt.x)
print('\tSOC: ', batt.event_state(filt.t, filt.x)['EOD'])

## Prediction - Predict EOD given current state
from prog_algs.predictors import monte_carlo
from prog_algs import samplers
mc = monte_carlo.MonteCarlo(batt)
if isinstance(filt, unscented_kalman_filter.UnscentedKalmanFilter):
    state_sampler = samplers.generate_mean_cov_random_sampler(batt.states, list(filt.x.values()), filt.Q)
    prediction_config = {'dt': 0.025, 'num_samples':5}
else:
    def state_sampler(num_samples):
        return filt.particles
    prediction_config = {'dt': 0.025, 'num_samples':len(filt.particles)}
(times, inputs, states, outputs, event_states, eol) = mc.predict(state_sampler, future_loading, prediction_config)

# Print Prediction Results
print("\nEOD Predictions (s):")
print('\t', eol)