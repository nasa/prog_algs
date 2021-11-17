# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

__all__ = ['predictors', 'uncertain_data', 'state_estimators', 'run_prog_playback', 'metrics']
from . import predictors, state_estimators, uncertain_data

import warnings

__version__ = '1.2.1'

def run_prog_playback(obs, pred, future_loading, output_measurements, **kwargs):
    warnings.warn("Depreciated in 1.2.0, will be removed in a future release.", DeprecationWarning)
    config = {# Defaults
        'predict_rate': 0, # Default- predict every step
        'num_samples': 10,
        'predict_config': {}
    }
    config.update(kwargs)

    next_predict = output_measurements[0][0] + config['predict_rate']
    times = []
    inputs = []
    states = []
    outputs = []
    event_states = []
    toes = []
    index = 0
    for (t, measurement) in output_measurements:
        obs.estimate(t, future_loading(t), measurement)
        if t >= next_predict:
            (t, u, x, z, es, toe) = pred.predict(obs.x.sample(config['num_samples']), future_loading, **config['predict_config'])
            times.append(t)
            inputs.append(u)
            states.append(x)
            outputs.append(z)
            event_states.append(es)
            toes.append(toe)
            index += 1
            next_predict += config['predict_rate']
    return (times, inputs, states, outputs, event_states, toes) 
