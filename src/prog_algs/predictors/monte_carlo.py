# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from . import predictor
from numpy import empty
from ..exceptions import ProgAlgTypeError
from copy import deepcopy
from functools import partial

def prediction_fcn(x, model, params, loading):
    # This is the main prediction function for the multi-threading
    first_output = model.output(x)
    params['x'] = x
    (times, inputs, states, outputs, event_states) = model.simulate_to_threshold(loading, first_output, **params)
    if (model.threshold_met(states[-1])):
        time_of_event = times[-1]
    else:
        time_of_event = None
    return (times, inputs, states, outputs, event_states, time_of_event)


class MonteCarlo(predictor.Predictor):
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
        """
        Perform a single prediction

        Parameters
        ----------
        state_samples : collection of samples for the MonteCarlo
            Function to generate n samples of the state. 
            e.g., def f(n): return [x1, x2, x3, ... xn]
        future_loading_eqn : function (t, x={}) -> z
            Function to generate an estimate of loading at future time t
        config : keyword arguments, optional
            Any additional configuration values. See default parameters

        Returns (tuple)
        -------
        times: [[number]]
            Times for each simulated point in format times[sample_id][index]
        inputs: [[dict]]
            Future input (from future_loading_eqn) for each sample and time in times
            where inputs[sample_id][index] corresponds to time times[sample_id][index]
        states: [[dict]]
            Estimated states for each sample and time in times
            where states[sample_id][index] corresponds to time times[sample_id][index]
        outputs: [[dict]]
            Estimated outputs for each sample and time in times
            where outputs[sample_id][index] corresponds to time times[sample_id][index]
        event_states: [[dict]]
            Estimated event state (e.g., SOH), between 1-0 where 0 is event occurance, for each sample and time in times
            where event_states[sample_id][index] corresponds to time times[sample_id][index]
        toe: [number]
            Estimated time where a predicted event will occur for each sample.
        """
        params = deepcopy(self.parameters) # copy parameters
        params.update(kwargs) # update for specific run

        times_all = empty(state_samples.size, dtype=object)
        inputs_all = empty(state_samples.size, dtype=object)
        states_all = empty(state_samples.size, dtype=object)
        outputs_all = empty(state_samples.size, dtype=object)
        event_states_all = empty(state_samples.size, dtype=object)
        time_of_event = empty(state_samples.size)

        # Perform prediction
        pred_fcn = partial(
            prediction_fcn, 
            model = self.model, 
            params = params,
            loading = future_loading_eqn)
        
        result = [pred_fcn(sample) for sample in state_samples]
        times_all = [tmp[0] for tmp in result]
        inputs_all = [tmp[1] for tmp in result]
        states_all = [tmp[2] for tmp in result]
        outputs_all = [tmp[3] for tmp in result]
        event_states_all = [tmp[4] for tmp in result]
        time_of_event = [tmp[5] for tmp in result]
        return (times_all, inputs_all, states_all, outputs_all, event_states_all, time_of_event)
