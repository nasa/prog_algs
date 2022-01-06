# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from .prediction import UnweightedSamplesPrediction
from .predictor import Predictor
from copy import deepcopy, copy
from numpy import array, where
from types import MethodType
from prog_models.sim_result import SimResult, LazySimResult
from prog_algs.uncertain_data import UnweightedSamples, UncertainData


class MonteCarlo(Predictor):
    """
    Class for performing a monte-carlo model-based prediction.

    A Predictor using the monte carlo algorithm. The provided initial states are simulated until either a specified time horizon is met, or the threshold for all simulated events is reached for all samples. A provided future loading equation is used to compute the inputs to the system at any given time point. 

    The following configuration parameters are supported (as kwargs in constructor or as parameters in predict method):
    
    Configuration Parameters
    ------------------------------
    t0 : float
        Initial time at which prediction begins, e.g., 0
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
        if isinstance(state, dict):
            # Convert to UnweightedSamples
            from prog_algs.uncertain_data import ScalarData
            state = ScalarData(state)

        params = deepcopy(self.parameters) # copy parameters
        params.update(kwargs) # update for specific run

        # Sample from state if n_samples specified or state is not UnweightedSamples
        if 'n_samples' in params:
            # If n_samples is specified, sample
            state = state.sample(params['n_samples'])
        elif not isinstance(state, UnweightedSamples):
            # If no n_samples specified, but state is not UnweightedSamples, then sample with default
            state = state.sample(self.DEFAULT_N_SAMPLES)

        model = self.model

        time_of_event_all = []
        last_states = []

        # Vectorized Prediction
        if model.is_vectorized:
            params = deepcopy(params)
            def apply_limits_vectorized(self, x):
                for (key, limit) in self.state_limits.items():
                    x[key] = where(x[key] < limit[0], limit[0], x[key])
                    x[key] = where(x[key] > limit[1], limit[1], x[key])
                return x
            model.apply_limits = MethodType(apply_limits_vectorized, model)
            first_state = {key: array([x[key] for x in state]) for key in state.keys()}
            first_output = model.output(first_state)
            params['x'] = first_state
            (times_all, inputs_all, states_all, outputs_all, event_states_all) = model.simulate_to_threshold(future_loading_eqn, first_output, **params, print=False)

            # Get last state (and remove from states_all)
            state_keys = list(states_all[0].keys())
            states_last = states_all.pop()
            state = UnweightedSamples([{key: states_last[key][i] for key in state_keys} for i in range(len(states_last[state_keys[0]]))])

            # Remove last value (i.e., event occurance)
            params['t0'] = times_all.pop()
            inputs_all.pop()
            outputs_all.pop()
            event_states_all.pop()

            # Transform _all structures
            # Convert from List[Dict[Str, float]] to Dict[Str, List[float]]
            times_all = [copy(times_all) for _ in range(len(state))]
            inputs_all = [deepcopy(inputs_all) for _ in range(len(state))]
            states_all = states_all.split()
            outputs_all = outputs_all.split()
            event_states_all = event_states_all.split()
        else:
            times_all = [[] for _ in range(len(state))]
            inputs_all = [SimResult() for _ in range(len(state))]
            states_all = [SimResult() for _ in range(len(state))]
            outputs_all = [LazySimResult(fcn = model.output) for _ in range(len(state))]
            event_states_all = [LazySimResult(fcn = model.event_state) for _ in range(len(state))]

        # Perform prediction
        for i, x in enumerate(state):
            events_remaining = deepcopy(params['events'])
            first_output = model.output(x)
            
            time_of_event = {}
            last_state = {}
            times = []
            inputs = SimResult()
            states = SimResult()
            outputs = LazySimResult(fcn = model.output)
            event_states = LazySimResult(fcn = model.event_state)

            params = deepcopy(params)
            params['x'] = x

            # Non-vectorized prediction
            while len(events_remaining) > 0:  # Still events to predict
                (t, u, xi, z, es) = model.simulate_to_threshold(future_loading_eqn, first_output, **params, threshold_keys=events_remaining, print=False)

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
            
            # Add to "all" structures
            times_all[i].extend(times)
            inputs_all[i].extend(inputs)
            states_all[i].extend(states)
            outputs_all[i].extend(outputs)
            event_states_all[i].extend(event_states)
            time_of_event_all.append(time_of_event)
            last_states.append(last_state)
        
        # Return longest time array
        times_length = [len(t) for t in times_all]
        times_max_len = max(times_length)
        times = times_all[times_length.index(times_max_len)] 
        
        inputs_all = UnweightedSamplesPrediction(times, inputs_all)
        states_all = UnweightedSamplesPrediction(times, states_all)
        outputs_all = UnweightedSamplesPrediction(times, outputs_all)
        event_states_all = UnweightedSamplesPrediction(times, event_states_all)
        time_of_event = UnweightedSamples(time_of_event_all)

        # Transform final states:
        last_states = {
            key: UnweightedSamples([sample[key] for sample in last_states]) for key in time_of_event.keys()
        }
        time_of_event.final_state = last_states

        return (times, inputs_all, states_all, outputs_all, event_states_all, time_of_event)
