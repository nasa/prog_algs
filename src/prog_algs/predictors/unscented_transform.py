# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from .prediction import MultivariateNormalDistPrediction
from .predictor import Predictor
from numpy import diag, array, transpose
from copy import deepcopy
from math import isnan
from filterpy import kalman
from prog_algs.uncertain_data import MultivariateNormalDist

class UnscentedTransformPredictor(Predictor):
    """
    Class for performing model-based prediction using an unscented transform. 

    This class defines logic for performing model-based state prediction using sigma points and an unscented transform. A Predictor is constructed using a PrognosticsModel object, (See Prognostics Model Package). The Unscented Transform Predictor propagates the sigma-points in the state-space in time domain until the event threshold is met. The step at which the i-th sigma point reaches the threshold is the step at which the i-th sigma point will be placed along the time dimension. By repeating the procedure for all sigma-points, we obtain the sigma-points defining the distribution of the event; for example, the End Of Life EOL event. The provided future loading equation is used to compute the inputs to the system at any given time point. 

    Parameters
    ----------
    * model : prog_models.prognostics_model.PrognosticsModel\n
        See: Prognostics Model Package\n
        A prognostics model to be used in prediction
    * options (optional, kwargs): configuration options\n
        Any additional configuration values. Note: These parameters can also be specified for an individual prediction. The following configuration parameters are supported: \n
        * alpha, beta, kappa: UKF Scaling parameters
        * dt : Step size (s)
        * horizon : Prediction horizon (s)
        
    NOTE: The resulting sigma-points along the time dimension are used to compute mean and covariance of the event time (EOL time), under the hypothesis that the EOL distribution would also be well represented by a Gaussian. This is a strong assumption that likely cannot be satisfied for real systems with strong non-linear state propagation or nonlinear EOL curves. Therefore, the user should be cautious and verify that modeling the event time using a Gaussian distribution is satisfactory.
    """
    default_parameters = { # Default Parameters
        'alpha': 1,     # UKF scaling param
        'beta': 0,      # UKF scaling param
        'kappa': -1,    # UKF scaling param
        't': 0,         # Starting Time (s)
        'dt': 0.5,      # Timestep, seconds
        'horizon': 1e99 # Prediction horizon, seconds
    }

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

        self.model = model
        self.__input = None  # Input at an individual step. Note, this needs to be a member to pass between state_transition and predict

        # setup UKF
        num_states = len(model.states)
        num_measurements = len(model.outputs)

        if 'Q' not in self.parameters:
            # Default 
            self.parameters['Q'] = diag([1.0e-1 for i in range(num_states)])
        if 'R' not in self.parameters:
            self.parameters['R'] = diag([1.0e-1 for i in range(num_measurements)])
        
        def measure(x):
            x = {key: value for (key, value) in zip(self.__state_keys, x)}
            z = model.output(x)
            return {array(list(z.values()))}

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
        options (optional, kwargs): configuration options\n
        Any additional configuration values. Note: These parameters can also be specified in the predictor constructor. The following configuration parameters are supported: \n
            * alpha, beta, kappa: UKF Scaling parameters
            * t: Starting time (s)
            * dt : Step size (s)
            * horizon : Prediction horizon (s)

        Returns (tuple)
        -------
        times: [number]
            Times for each simulated point in format times[index]
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
        toe: UncertainData
            Estimated time where a predicted event will occur for each sample. Note: Mean and Covariance Matrix will both 
            be nan if every sigma point doesnt reach threshold within horizon
        """
        params = deepcopy(self.parameters) # copy parameters
        params.update(kwargs) # update for specific run

        # Optimizations 
        dt = params['dt']
        model = self.model
        filt = self.filter
        sigma_points = self.sigma_points
        n_points = sigma_points.num_sigmas()

        # Update State 
        self.__state_keys = state.mean.keys()  # Used to maintain ordering as we strip keys and return
        filt.x = [x for x in state.mean.values()]
        filt.P = state.cov

        # Setup first states
        t = params['t']
        EOL = {key: [float('nan') for i in range(n_points)] for key in self.model.events}  # Keep track of final EOL values

        # Simulation
        while t < params['horizon']:
            # Iterate through time
            t += dt
            mean_state = {key: x for (key, x) in zip(state.mean.keys(), filt.x)}
            self.__input = future_loading_eqn(t, mean_state)
            filt.predict(dt=dt)
            
            # Check that any sigma point has hit event
            points = sigma_points.sigma_points(filt.x, filt.P)
            all_failed = True
            for i, point in zip(range(n_points), points):
                x = {key: x for (key, x) in zip(state.mean.keys(), point)}
                t_met = model.threshold_met(x)
                for key in t_met.keys():
                    if t_met[key]:
                        if isnan(EOL[key][i]):
                            # First time event has been reached
                            EOL[key][i] = t
                    else:
                        all_failed = False
            if all_failed:
                # If all events have been reched for every sigma point
                break 
        
        # Prepare Results
        pts = array([[e for e in EOL[key]] for key in EOL.keys()])
        pts = transpose(pts)
        mean, cov = kalman.unscented_transform(pts, sigma_points.Wm, sigma_points.Wc)

        # At this point only time of event is calculated 
        times_all = []
        empty_prediction = MultivariateNormalDistPrediction(times_all, [])
        time_of_event = MultivariateNormalDist(EOL.keys(), mean, cov)
        return (times_all, empty_prediction, empty_prediction, empty_prediction, empty_prediction, time_of_event)
