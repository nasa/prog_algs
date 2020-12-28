# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import sys
sys.path.insert(1, "../python-prognostics-models-package/")
from prog_models.models import battery_circuit
from prog_algs import *

def run_example():
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
    filt = state_estimators.unscented_kalman_filter.UnscentedKalmanFilter(batt, batt.parameters['x0'])

    ## Setup Prediction
    mc = predictors.monte_carlo.MonteCarlo(batt)
    prediction_config = {'dt': 0.05}

    # Playback 
    from prog_algs.metrics import samples as metrics 
    print('Run 1 (2 samples)')
    (t, u, x, z, es, eol) = run_prog_playback(filt, mc, future_loading, [(0.1, {'t': 32.2, 'v': 3.915}), (0.1, {'t': 32.3, 'v': 3.91})], {'predict_config': prediction_config, 'num_samples': 2})
    print('MSE: ', metrics.mean_square_error(eol, 3005.4))
    prediction_times = [0.1, 0.2]
    print('Alpha-lambda met: ', metrics.alpha_lambda(prediction_times, eol, 3005.4, 0.2, 1e-4, 0.65))

    print('Run 2 (5 samples)')
    (t, u, x, z, es, eol) = run_prog_playback(filt, mc, future_loading, [(0.1, {'t': 32.2, 'v': 3.915}), (0.1, {'t': 32.3, 'v': 3.91})], {'predict_config': prediction_config, 'num_samples': 5})
    print('MSE: ', metrics.mean_square_error(eol, 3005.4))
    prediction_times = [0.1, 0.2]
    print('Alpha-lambda met: ', metrics.alpha_lambda(prediction_times, eol, 3005.4, 0.2, 1e-4, 0.65))

    print('Run 3 (10 samples)')
    (t, u, x, z, es, eol) = run_prog_playback(filt, mc, future_loading, [(0.1, {'t': 32.2, 'v': 3.915}), (0.1, {'t': 32.3, 'v': 3.91})], {'predict_config': prediction_config, 'num_samples': 10})
    print('MSE: ', metrics.mean_square_error(eol, 3005.4))
    prediction_times = [0.1, 0.2]
    print('Alpha-lambda met: ', metrics.alpha_lambda(prediction_times, eol, 3005.4, 0.2, 1e-4, 0.65))

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()