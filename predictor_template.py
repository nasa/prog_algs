# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from prog_algs import predictor

class TemplatePredictor(predictor.Predictor):
    """
    Template class for performing model-based prediction
    """

    # REPLACE THE FOLLOWING LIST WITH CONFIGURED PARAMETERS
    parameters = { # Default Parameters, used as config for UKF
        'Example Parameter': 0.0
    } 

    def __init__(self, model, options = {}):
        """
        Constructor
        """
        self.model = model

        self.parameters.update(options)# Merge configuration options into default
        # ADD PARAMETER CHECKS HERE

    def predict(self, state_sampler, future_loading_eqn, options = {}):
        """
        Perform a single prediction

        Parameters
        ----------
        state_sampler : function (n) -> [x1, x2, ... xn]
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

        # PERFORM PREDICTION HERE, REPLACE THE FOLLOWING LISTS
        times = [] # array of double (e.g., [0.0, 0.5, 1.0, ...])
        inputs = [] # array of dict (e.g., [{'input 1': 1.2, ...}, ...])
        states = [] # array of dict (e.g., [{'state 1': 1.2, ...}, ...])
        outputs = [] # array of dict (e.g., [{'output 1': 1.2, ...}, ...])
        event_states = [] # array of dict (e.g., [{'event_state 1': 1.2, ...}, ...])
        time_of_event = [] # array of double, time for each event prediction

        return (times_all, inputs_all, states_all, outputs_all, event_states_all, time_of_event)