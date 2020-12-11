# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from . import predictor
from numpy import empty
from ..exceptions import ProgAlgTypeError

class MonteCarlo(predictor.Predictor):
    """
    Class for performing model-based prediction using sampling. 

    This class defines logic for performing model-based state prediction using sampling methods. A Predictor is constructed using a PrognosticsModel object, (See Prognostics Model Package). The states are simulated until either a specified time horizon is met, or the threshold is reached for all samples, as defined by the threshold equation. A provided future loading equation is used to compute the inputs to the system at any given time point. 
    """
    parameters = { # Default Parameters
        'dt': 0.5,          # Timestep, seconds
        'horizon': 4000,    # Prediction horizon, seconds
        'save_freq': 10     # Frequency at which results are saved
    }

    def __init__(self, model):
        """
        Construct a MonteCarlo Predictor

        Parameters
        ----------
        model : prog_models.prognostics_model.PrognosticsModel
            See: Prognostics Model Package
            A prognostics model to be used in prediction
        """
        self._model = model
        if not hasattr(model, 'output'):
            raise ProgAlgTypeError("model must have `output` method")
        if not hasattr(model, 'next_state'):
            raise ProgAlgTypeError("model must have `next_state` method")
        if not hasattr(model, 'inputs'):
            raise ProgAlgTypeError("model must have `inputs` property")
        if not hasattr(model, 'outputs'):
            raise ProgAlgTypeError("model must have `outputs` property")
        if not hasattr(model, 'states'):
            raise ProgAlgTypeError("model must have `states` property")
        if not hasattr(model, 'simulate_to_threshold'):
            raise ProgAlgTypeError("model must have `simulate_to_threshold` property")

    def predict(self, state_samples, future_loading_eqn, options = {}):
        """
        Perform a single prediction

        Parameters
        ----------
        state_samples : collection of samples for the MonteCarlo
            Function to generate n samples of the state. 
            e.g., def f(n): return [x1, x2, x3, ... xn]
        future_loading_eqn : function (t) -> z
            Function to generate an estimate of loading at future time t
        options : dict, optional
            Dictionary of any additional configuration values. See default parameters, above

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
        params = self.parameters # copy default parameters
        params.update(options)

        times_all = empty(state_samples.size, dtype=object)
        inputs_all = empty(state_samples.size, dtype=object)
        states_all = empty(state_samples.size, dtype=object)
        outputs_all = empty(state_samples.size, dtype=object)
        event_states_all = empty(state_samples.size, dtype=object)
        time_of_event = empty(state_samples.size)

        # Optimization to reduce lookup
        output = self._model.output
        simulate_to_threshold = self._model.simulate_to_threshold
        threshold_met = self._model.threshold_met

        # Perform prediction
        for (i, x) in zip(range(len(state_samples)), state_samples):
            first_output = output(0, x)
            params['x'] = x
            (times, inputs, states, outputs, event_states) = simulate_to_threshold(future_loading_eqn, first_output, params)
            if (threshold_met(times[-1], states[-1])):
                time_of_event[i] = times[-1]
            else:
                time_of_event[i] = None
            times_all[i] = times
            inputs_all[i] = inputs
            states_all[i] = states
            outputs_all[i] = outputs
            event_states_all[i] = event_states
        return (times_all, inputs_all, states_all, outputs_all, event_states_all, time_of_event)
            
        
