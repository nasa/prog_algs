# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

__all__ = ['predictors', 'uncertain_data', 'state_estimators', 'run_prog_playback']
from .predictors import *
from .state_estimators import *
from .uncertain_data import *

import numpy as np

def run_prog_playback(obs, pred, future_loading, output_measurements, **kwargs):
    config = {# Defaults
        'predict_rate': 0, # Default- predict every step
        'num_samples': 10,
        'predict_config': {}
    }
    config.update(kwargs)

    next_predict = output_measurements[0][0] + config['predict_rate']
    times = np.empty((len(output_measurements), config['num_samples']), dtype=object)
    inputs = np.empty((len(output_measurements), config['num_samples']), dtype=object)
    states = np.empty((len(output_measurements), config['num_samples']), dtype=object)
    outputs = np.empty((len(output_measurements), config['num_samples']), dtype=object)
    event_states = np.empty((len(output_measurements), config['num_samples']), dtype=object)
    eols = np.empty((len(output_measurements), config['num_samples']), dtype=object)
    index = 0
    for (t, measurement) in output_measurements:
        obs.estimate(t, future_loading(t), measurement)
        if t >= next_predict:
            (t, u, x, z, es, eol) = pred.predict(obs.x.sample(config['num_samples']), future_loading, **config['predict_config'])
            times[index, :] = t
            inputs[index, :]  = u
            states[index, :]  = x
            outputs[index, :]  = z
            event_states[index, :]  =  es
            eols[index, :] = eol
            index += 1
            next_predict += config['predict_rate']
    return (times, inputs, states, outputs, event_states, eols) 
