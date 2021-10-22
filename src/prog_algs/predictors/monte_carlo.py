# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from .prediction import UnweightedSamplesPrediction
from .predictor import Predictor
from copy import deepcopy
from functools import partial
from prog_models.sim_result import SimResult, LazySimResult
from prog_algs.uncertain_data import UnweightedSamples

def prediction_fcn(x, model, params, events, loading):
    # This is the main prediction function for the multi-threading
    events_remaining = deepcopy(events)
    first_output = model.output(x)
    time_of_event = {}
    params['t'] = 0
    times = []
    # inputs will be the same as states unless we explicitly deepcopy
    inputs = deepcopy(SimResult())
    states = deepcopy(SimResult())  
    outputs = LazySimResult(fcn = model.output)
    event_states = LazySimResult(fcn = model.event_state)
    params['x'] = x
    while len(events_remaining) > 0:  # Still events to predict
        (t, u, xi, z, es) = model.simulate_to_threshold(loading, first_output, **params, threshold_keys=events)

        # Add results
        times.extend(t)
        inputs.extend(u)
        states.extend(xi)
        outputs.extend(z)
        event_states.extend(es)

        # Get which event occurs
        t_met = model.threshold_met(states[-1])
        try:
            event = list(t_met.keys())[list(t_met.values()).index(True)]
        except ValueError:
            # no event has occured
            for event in events_remaining:
                time_of_event[event] = None

        # An event has occured
        # TODO(CT): Multiple events occur simulatiously 
        time_of_event[event] = times[-1]
        events_remaining.remove(event)  # No longer an event to predect to

        # Remove last state (event)
        params['t'] = times.pop()
        inputs.pop()
        params['x'] = states.pop()
        outputs.pop()
        event_states.pop()
        
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
    }

    def predict(self, state_samples, future_loading_eqn, **kwargs):
        params = deepcopy(self.parameters) # copy parameters
        params.update(kwargs) # update for specific run

        # Perform prediction
        pred_fcn = partial(
            prediction_fcn, 
            model = self.model, 
            params = params,
            events = params['events'],
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
        time_of_event = UnweightedSamples(time_of_event)
        return (times, inputs_all, states_all, outputs_all, event_states_all, time_of_event)
