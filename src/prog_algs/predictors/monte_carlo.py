# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from .prediction import UnweightedSamplesPrediction
from .predictor import Predictor
from copy import deepcopy
from functools import partial
from prog_models.sim_result import SimResult, LazySimResult
from prog_algs.uncertain_data import UnweightedSamples, UncertainData

def prediction_fcn(x, model, params, events, loading):
    # This is the main prediction function for the multi-threading
    events_remaining = deepcopy(events)
    first_output = model.output(x)
    time_of_event = {}
    last_state = {}
    times = []
    # inputs will be the same as states unless we explicitly deepcopy
    inputs = SimResult()
    states = SimResult()
    outputs = LazySimResult(fcn = model.output)
    event_states = LazySimResult(fcn = model.event_state)
    params = deepcopy(params)
    params['x'] = x
    while len(events_remaining) > 0:  # Still events to predict
        (t, u, xi, z, es) = model.simulate_to_threshold(loading, first_output, **params, threshold_keys=events_remaining, print=False)

        # Add results
        times.extend(t)
        inputs.extend(u)
        states.extend(xi)
        outputs.extend(z)
        event_states.extend(es)

        # Get which event occurs
        t_met = model.threshold_met(states[-1])
        t_met = {key: t_met[key] for key in events_remaining}  # Only look at remaining keys
        try:
            event = list(t_met.keys())[list(t_met.values()).index(True)]
        except ValueError:
            # no event has occured
            for event in events_remaining:
                time_of_event[event] = None
                last_state[event] = None
            break

        # An event has occured
        time_of_event[event] = times[-1]
        events_remaining.remove(event)  # No longer an event to predect to

        # Remove last state (event)
        params['t0'] = times.pop()
        if 'horizon' in params:
            # Reset horizon to account for time spent
            params['horizon'] = params['horizon'] - params['t0']
        inputs.pop()
        params['x'] = states.pop()
        last_state[event] = deepcopy(params['x'])
        outputs.pop()
        event_states.pop()
        
    return (times, inputs, states, outputs, event_states, time_of_event, last_state)


class MonteCarlo(Predictor):
    """
    Class for performing a monte-carlo model-based prediction.

    A Predictor using the monte carlo algorithm. The provided initial states are simulated until either a specified time horizon is met, or the threshold for all simulated events is reached for all samples. A provided future loading equation is used to compute the inputs to the system at any given time point. 

    The following configuration parameters are supported (as kwargs in constructor or as parameters in predict method):
    
    Configuration Parameters
    ------------------------------
    dt : float
        Simulation step size (s), e.g., 0.1
    events : List[string]
        Events to predict (subset of model.events) e.g., ['event1', 'event2']
    horizon : float
        Prediction horizon (s)
    n_samples : int
        Number of samples to use. If not specified, a default value is used. If state is type UnweightedSamples and n_samples is not provided, the provided unweighted samples will be used directly.
    save_freq : float
        Frequency at which results are saved (s)
    save_pts : List[float]
        Any additional savepoints (s) e.g., [10.1, 22.5]
    """
    DEFAULT_N_SAMPLES = 100  # Default number of samples to use, if none specified

    def predict(self, state : UncertainData, future_loading_eqn, **kwargs):
        params = deepcopy(self.parameters) # copy parameters
        params.update(kwargs) # update for specific run

        # Sample from state if n_samples specified or state is not UnweightedSamples
        if 'n_samples' in params:
            # If n_samples is specified, sample
            state = state.sample(params['n_samples'])
        elif not isinstance(state, UnweightedSamples):
            # If no n_samples specified, but state is not UnweightedSamples, then sample with default
            state = state.sample(self.DEFAULT_N_SAMPLES)

        # Perform prediction
        pred_fcn = partial(
            prediction_fcn, 
            model = self.model, 
            params = params,
            events = params['events'],
            loading = future_loading_eqn)
        
        result = [pred_fcn(sample) for sample in state]
        times_all, inputs_all, states_all, outputs_all, event_states_all, time_of_event, last_states = map(list, zip(*result))
        
        # Return longest time array
        times_length = [len(t) for t in times_all]
        times_max_len = max(times_length)
        times = times_all[times_length.index(times_max_len)] 
        
        inputs_all = UnweightedSamplesPrediction(times, inputs_all)
        states_all = UnweightedSamplesPrediction(times, states_all)
        outputs_all = UnweightedSamplesPrediction(times, outputs_all)
        event_states_all = UnweightedSamplesPrediction(times, event_states_all)
        time_of_event = UnweightedSamples(time_of_event)

        # Transform final states:
        last_states = {
            key: UnweightedSamples([sample[key] for sample in last_states]) for key in time_of_event.keys()
        }
        time_of_event.final_state = last_states
        return (times, inputs_all, states_all, outputs_all, event_states_all, time_of_event)
