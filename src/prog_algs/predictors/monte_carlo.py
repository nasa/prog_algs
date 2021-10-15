# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from .prediction import UnweightedSamplesPrediction
from .predictor import Predictor
from ..exceptions import ProgAlgTypeError
from copy import deepcopy
from functools import partial
from prog_algs.uncertain_data import UnweightedSamples

def prediction_fcn(x, model, params, loading):
    # This is the main prediction function for the multi-threading
    first_output = model.output(x)
    params['x'] = x
    (times, inputs, states, outputs, event_states) = model.simulate_to_threshold(loading, first_output, **params, print=False)
    if (model.threshold_met(states[-1])):
        time_of_event = times[-1]
    else:
        time_of_event = None
    return (times, inputs, states, outputs, event_states, time_of_event)


class MonteCarlo(Predictor):
    """
    Class for performing model-based prediction using sampling. 

    This class defines logic for performing model-based state prediction using sampling methods. A Predictor is constructed using a PrognosticsModel object, (See Prognostics Model Package). The states are simulated until either a specified time horizon is met, or the threshold is reached for all samples, as defined by the threshold equation. A provided future loading equation is used to compute the inputs to the system at any given time point. 

    Parameters
    ----------
    * model : prog_models.prognostics_model.PrognosticsModel\n
        See: Prognostics Model Package\n
        A prognostics model to be used in prediction
    * options (optional, kwargs): configuration options\n
        Any additional configuration values. See default parameters. Additionally, the following configuration parameters are supported: \n
        * dt : Step size (s)
        * horizon : Prediction horizon (s)
        * save_freq : Frequency at which results are saved (s)
        * save_pts : Any additional savepoints (s) e.g., [10.1, 22.5]
        * cores : Number of cores to use in multithreading
    """
    default_parameters = { # Default Parameters
        'dt': 0.5,          # Timestep, seconds
        'horizon': 4000,    # Prediction horizon, seconds
        'save_freq': 10,    # Frequency at which results are saved
        'cores': 6          # Number of cores to use in parallelization
    }

    def predict(self, state_samples, future_loading_eqn, **kwargs):
        params = deepcopy(self.parameters) # copy parameters
        params.update(kwargs) # update for specific run

        # Perform prediction
        pred_fcn = partial(
            prediction_fcn, 
            model = self.model, 
            params = params,
            loading = future_loading_eqn)
        
        result = [pred_fcn(sample) for sample in state_samples]
        times_all, inputs_all, states_all, outputs_all, event_states_all, time_of_event = map(list, zip(*result))
        
        # Return longest time array
        times_length = [len(t) for t in times_all]
        times_max_len = max(times_length)
        times = times_all[times_length.index(times_max_len)] 
        
        inputs_all = UnweightedSamplesPrediction(times, inputs_all)
        states_all = UnweightedSamplesPrediction(times, states_all)
        outputs_all = UnweightedSamplesPrediction(times, outputs_all)
        event_states_all = UnweightedSamplesPrediction(times, event_states_all)
        return (times, inputs_all, states_all, outputs_all, event_states_all, time_of_event)
