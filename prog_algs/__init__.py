# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

__all__ = ['predictors', 'samplers', 'state_estimators', 'run_prog_playback']
from .predictors import *
from .state_estimators import *
from .samplers import *

import numpy as np

def run_prog_playback(obs, pred, state_sampler, future_loading, output_measurements, options = {}):
    config = {# Defaults
        'predict_rate': 0, # Default- predict every step
        'predict_config': {}
    }
    config.update(options)

    next_predict = output_measurements[0][0] + config['predict_rate']
    times = np.empty((len(output_measurements), config['predict_config']['num_samples']), dtype=object)
    inputs = np.empty((len(output_measurements), config['predict_config']['num_samples']), dtype=object)
    states = np.empty((len(output_measurements), config['predict_config']['num_samples']), dtype=object)
    outputs = np.empty((len(output_measurements), config['predict_config']['num_samples']), dtype=object)
    event_states = np.empty((len(output_measurements), config['predict_config']['num_samples']), dtype=object)
    eols = np.empty((len(output_measurements), config['predict_config']['num_samples']), dtype=object)
    index = 0
    for (t, measurement) in output_measurements:
        obs.estimate(t, measurement)
        if t >= next_predict:
            (t, u, x, z, es, eol) = pred.predict(state_sampler, future_loading, config['predict_config'])
            times[index, :] = t
            inputs[index, :]  = u
            states[index, :]  = x
            outputs[index, :]  = z
            event_states[index, :]  =  es
            eols[index, :] = eol
            index += 1
            next_predict += config['predict_rate']
    return (times, inputs, states, outputs, event_states, eols) 
