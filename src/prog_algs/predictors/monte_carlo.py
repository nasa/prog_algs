# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from .prediction import UnweightedSamplesPrediction, PredictionResults
from .predictor import Predictor
from copy import deepcopy
from typing import Callable
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

    default_parameters = { 
        'n_samples': 100  # Default number of samples to use, if none specified
    }

    def predict(self, state : UncertainData, future_loading_eqn : Callable, **kwargs) -> PredictionResults:
        if isinstance(state, dict) or isinstance(state, self.model.StateContainer):
            from prog_algs.uncertain_data import ScalarData
            state = ScalarData(state, _type = self.model.StateContainer)
        elif isinstance(state, UncertainData):
            state._type = self.model.StateContainer
        else:
            raise TypeError("state must be UncertainData, dict, or StateContainer")

        params = deepcopy(self.parameters) # copy parameters
        params.update(kwargs) # update for specific run

        # Sample from state if n_samples specified or state is not UnweightedSamples
        state = state.sample(params['n_samples'])

        ouput_eqn = self.model.output
        es_eqn = self.model.event_state
        tm_eqn = self.model.threshold_met
        simulate_to_threshold = self.model.simulate_to_threshold

        time_of_event_all = []
        last_states = []
        times_all = []
        inputs_all = []
        states_all = []
        outputs_all = []
        event_states_all = []

        # Perform prediction
        for x in state:
            events_remaining = params['events'].copy()
            first_output = ouput_eqn(x)
            
            time_of_event = {}
            last_state = {}
            times = []
            inputs = SimResult()
            states = SimResult()
            outputs = LazySimResult(fcn = ouput_eqn)
            event_states = LazySimResult(fcn = es_eqn)

            t0 = params.get('t0', 0)
            if 'save_freq' in params and not isinstance(params['save_freq'], tuple):
                params['save_freq'] = (t0, params['save_freq'])
            

            # Non-vectorized prediction
            while len(events_remaining) > 0:  # Still events to predict
                (t, u, xi, z, es) = simulate_to_threshold(future_loading_eqn, first_output, 
                    **{**params, 'threshold_keys': events_remaining, 'print': False, 'progress': False, 't0': t0, 'x': x}  # Merge then separate
                )
                if len(times) != 0:
                    # is not the first iteration, therefore the first saved state is when the last event 
                    t.pop(0)
                    u.pop(0)
                    xi.pop(0)
                    z.pop(0)
                    es.pop(0)

                # Add results
                times.extend(t)
                inputs.extend(u)
                states.extend(xi)
                outputs.extend(z)
                event_states.extend(es)

                # Get which event occurs
                t_met = tm_eqn(states[-1])
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
                t0 = times.pop()
                inputs.pop()
                x = states.pop()
                last_state[event] = x.copy()
                outputs.pop()
                event_states.pop()
            
            # Add to "all" structures
            times_all.append(times)
            inputs_all.append(inputs)
            states_all.append(states)
            outputs_all.append(outputs)
            event_states_all.append(event_states)
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
            key: UnweightedSamples([sample[key] for sample in last_states], _type = self.model.StateContainer) for key in time_of_event.keys()
        }
        time_of_event.final_state = last_states

        return PredictionResults(
            times, 
            inputs_all, 
            states_all, 
            outputs_all, 
            event_states_all, 
            time_of_event
        )
