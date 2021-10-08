# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

"""
This example performs benchmarking for a state estimation and prediction with uncertainty given a Prognostics Model.
 
Method: An instance of the BatteryCircuit model in prog_models is created, state estimation is set up with a chosen state_estimator, and prediction is set up with a chosen predictor.
        Prediction of future states (with uncertainty) is then performed for various sample sizes. 
        Metrics are calculated and displayed. 

Results: 
    i) Predicted future values (inputs, states, outputs, event_states) with uncertainty from prediction for each distinct sample size
    ii) Time event is predicted to occur (with uncertainty)
    iii) Various prediction metrics, including alpha-lambda metric 
"""

from prog_models.models import BatteryCircuit
from prog_algs import *
from prog_algs import run_prog_playback

def run_example():
    ## Setup
    def future_loading(t, x={}):
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

    ##  Setup State Estimation 
    filt = state_estimators.UnscentedKalmanFilter(batt, batt.parameters['x0'])

    ## Setup Prediction
    mc = predictors.MonteCarlo(batt, dt= 0.05)

    # Playback 
    from prog_algs.metrics import samples as metrics 
    print('Run 1 (2 samples)')
    (t, u, x, z, es, eol) = run_prog_playback(filt, mc, future_loading, [(0.1, {'t': 32.2, 'v': 3.915}), (0.1, {'t': 32.3, 'v': 3.91})], num_samples= 2)
    print('MSE: ', metrics.mean_square_error(eol, 3005.4))
    prediction_times = [0.1, 0.2]
    print('Alpha-lambda met: ', metrics.alpha_lambda(prediction_times, eol, 3005.4, 0.2, 1e-4, 0.65))

    print('Run 2 (5 samples)')
    (t, u, x, z, es, eol) = run_prog_playback(filt, mc, future_loading, [(0.1, {'t': 32.2, 'v': 3.915}), (0.1, {'t': 32.3, 'v': 3.91})], num_samples= 5)
    print('MSE: ', metrics.mean_square_error(eol, 3005.4))
    prediction_times = [0.1, 0.2]
    print('Alpha-lambda met: ', metrics.alpha_lambda(prediction_times, eol, 3005.4, 0.2, 1e-4, 0.65))

    print('Run 3 (10 samples)')
    (t, u, x, z, es, eol) = run_prog_playback(filt, mc, future_loading, [(0.1, {'t': 32.2, 'v': 3.915}), (0.1, {'t': 32.3, 'v': 3.91})], num_samples= 10)
    print('MSE: ', metrics.mean_square_error(eol, 3005.4))
    prediction_times = [0.1, 0.2]
    print('Alpha-lambda met: ', metrics.alpha_lambda(prediction_times, eol, 3005.4, 0.2, 1e-4, 0.65))

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()
