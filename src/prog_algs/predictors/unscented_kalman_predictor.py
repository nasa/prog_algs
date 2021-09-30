# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from .prediction import Prediction
from .predictor import Predictor
from numpy import diag, array, transpose
from copy import deepcopy
from filterpy import kalman
from prog_algs.uncertain_data import MultivariateNormalDist

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


class UnscentedKalmanPredictor(Predictor):
    """
    Class for performing model-based prediction using an unscented kalman filter. 

    This class defines logic for performing model-based state prediction using the unscented kalman filter method. A Predictor is constructed using a PrognosticsModel object, (See Prognostics Model Package). The states are simulated until either a specified time horizon is met, or the threshold is reached, as defined by the threshold equation. A provided future loading equation is used to compute the inputs to the system at any given time point. 

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
        'alpha': 1,     # UKF scaling param
        'beta': 0,      # UKF scaling param
        'kappa': -1,    # UKF scaling param
        'dt': 0.5,          # Timestep, seconds
        'horizon': 4000,    # Prediction horizon, seconds
        'save_freq': 10,    # Frequency at which results are saved
        't': 0
    }

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

        self.model = model
        self.__input = None

        # setup UKF
        num_states = len(model.states)
        num_measurements = len(model.outputs)

        if 'Q' not in self.parameters:
            self.parameters['Q'] = diag([1.0e-1 for i in range(num_states)])
        if 'R' not in self.parameters:
            # Size of what's being measured (not output) 
            # This is determined by running the measure function on the first state
            self.parameters['R'] = diag([1.0e-1 for i in range(num_measurements)])
        
        def measure(x):
                x = {key: value for (key, value) in zip(self.__state_keys, x)}
                z = model.output(x)
                return array(list(z.values()))

        def state_transition(x, dt):
            x = {key: value for (key, value) in zip(self.__state_keys, x)}
            x = model.next_state(x, self.__input, dt)
            x = model.apply_limits(x)
            return array(list(x.values()))

        self.sigma_points = kalman.MerweScaledSigmaPoints(num_states, alpha=self.parameters['alpha'], beta=self.parameters['beta'], kappa=self.parameters['kappa'])
        self.filter = kalman.UnscentedKalmanFilter(num_states, num_measurements, self.parameters['dt'], measure, state_transition, self.sigma_points)
        self.filter.Q = self.parameters['Q']
        self.filter.R = self.parameters['R']

    def predict(self, state, future_loading_eqn, **kwargs):
        """
        Perform a single prediction

        Parameters
        ----------
        state (UncertaintData): Distribution of states
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
        dt = self.parameters['dt']
        model = self.model
        sigma_points = self.sigma_points
        t = self.parameters['t']

        self.__state_keys = state.mean.keys()  # Used to maintain ordering as we strip keys and return
        self.filter.x = [x for x in state.mean.values()]
        self.filter.P = state.cov
        mean_state = {key: x for (key, x) in zip(state.mean.keys(), self.filter.x)}
        n_points = sigma_points.num_sigmas()
        EOL = {key: [None for i in range(n_points)] for key in self.model.events}

        while True:
            t += dt
            self.__input = future_loading_eqn(t, mean_state)
            self.filter.predict(dt=dt)
            mean_state = {key: x for (key, x) in zip(state.mean.keys(), self.filter.x)}
            
            # Check that every sigma point has hit event
            points = sigma_points.sigma_points(self.filter.x, self.filter.P)
            all_failed = True
            for i, point in zip(range(n_points), points):
                x = {key: x for (key, x) in zip(state.mean.keys(), point)}
                t_met = model.threshold_met(x)
                for key in t_met.keys():
                    if t_met[key]:
                        if EOL[key][i] is None:
                            EOL[key][i] = t
                    else:
                        all_failed = False
            if all_failed:
                break 
        
        pts = array([[e for e in EOL[key]] for key in EOL.keys()])
        pts = transpose(pts)
        mean, cov = kalman.unscented_transform(pts, sigma_points.Wm, sigma_points.Wc)

        times_all = []
        inputs_all = Prediction(times_all, [])
        states_all = Prediction(times_all, [])
        outputs_all = Prediction(times_all, [])
        event_states_all = Prediction(times_all, [])
        time_of_event = MultivariateNormalDist(EOL.keys(), mean, cov)
        return (times_all, inputs_all, states_all, outputs_all, event_states_all, time_of_event)
