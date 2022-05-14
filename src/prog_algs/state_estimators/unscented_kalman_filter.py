# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from warnings import warn
from typing import Callable
from . import state_estimator
from filterpy import kalman
from numpy import diag, array
from ..uncertain_data import MultivariateNormalDist, UncertainData

class UnscentedKalmanFilter(state_estimator.StateEstimator):
    """
    An Unscented Kalman Filter (UKF) for state estimation

    This class defines logic for performing an unscented kalman filter with a Prognostics Model (see Prognostics Model Package). This filter uses measurement data with noise to generate a state estimate and covariance matrix. 

    The supported configuration parameters (keyword arguments) for UKF construction are described below:

    Constructor Configuration Parameters:
        alpha, beta, kappa: float
            UKF Scaling parameters
        t0 : float
            Starting time (s)
        dt : float 
            time step (s)
        process_noise : 2darray[float]
            Process Noise Matrix (n_states, n_states) - default model.process_noise
        measurement_noise : 2darray[float]
            Measurement Noise Matrix (n_measurements x n_measurements) - default model.measurement_noise
    """
    default_parameters = {
        'alpha': 1, 
        'beta': 0, 
        'kappa': -1,
        't0': -1e-10,
        'dt': 1
    } 

    def __init__(self, model, x0, measurement_eqn : Callable = None, **kwargs):
        super().__init__(model, x0, measurement_eqn, **kwargs)

        self.__input = None
        self.x0 = x0
        # Saving for reduce pickling

        if measurement_eqn is None: 
            def measure(x):
                x = {key: value for (key, value) in zip(x0.keys(), x)}
                R_err = model.parameters['measurement_noise'].copy()
                model.parameters['measurement_noise'] = dict.fromkeys(R_err, 0)
                z = model.output(x)
                model.parameters['measurement_noise'] = R_err
                return array(list(z.values())).ravel()
        else:
            def measure(x):
                x = {key: value for (key, value) in zip(x0.keys(), x)}
                z = measurement_eqn(x)
                return array(list(z.values())).ravel()

        num_states = model.n_states

        # Process Noise (Q)
        # Users can use process_noise (like in prog_models) or Q (like in filterpy). They're synced.
        if 'Q' in self.parameters:
            self.parameters['process_noise'] = self.parameters['Q']
        elif 'process_noise' not in self.parameters:
            # Not provided
            if 'process_noise' in model.parameters:
                # If model has process noise, use it
                self.parameters['process_noise'] = diag([model.parameters['process_noise'][key] for key in x0.keys()])
            else:
                self.parameters['process_noise'] = diag([1.0e-3 for _ in range(num_states)])
        else:
            # If process noise is provided

            # Manage type
            if isinstance(self.parameters['process_noise'], list):
                self.parameters['process_noise'] = array(self.parameters['process_noise'])
            
            # Check dims
            if self.parameters['process_noise'].shape != (num_states, num_states):
                raise Exception('process_noise must be a square matrix with size equal to the number of states')

        # Measurement Noise (R)
        # Users can use measurement_noise (like in prog_models) or R (like in filterpy). They're synced.
        if 'R' in self.parameters:
            self.parameters['measurement_noise'] = self.parameters['R']
        elif 'measurement_noise' not in self.parameters:
            # Not provided
            if 'measurement_noise' in model.parameters and measurement_eqn is None:
                # Pull from model noise (doesn't work when measurement equation is provided)
                self.parameters['measurement_noise'] = diag([model.parameters['measurement_noise'][key] for key in model.outputs])
            else:
                # Default to 1.0e-3 standard deviation on every output
                model.parameters['measurement_noise'] = 0
                num_measurements = len(measure(x0.values()))
                self.parameters['measurement_noise'] = diag([1.0e-3 for _ in range(num_measurements)])
        else:
            # Manage type
            if isinstance(self.parameters['measurement_noise'], list):
                self.parameters['measurement_noise'] = array(self.parameters['measurement_noise'])
            
            # Check dims
            num_measurements = len(measure(x0.values()))
            if self.parameters['measurement_noise'].shape != (num_measurements, num_measurements):
                raise Exception('measurement_noise must be a square matrix with size equal to the number of outputs')
        
        num_measurements = model.n_outputs

        def state_transition(x, dt):
            x = {key: value for (key, value) in zip(x0.keys(), x)}
            Q_err = model.parameters['process_noise'].copy()
            model.parameters['process_noise'] = dict.fromkeys(Q_err, 0)
            x = model.next_state(x, self.__input, dt)
            model.parameters['process_noise'] = Q_err
            return array(list(x.values())).ravel()

        points = kalman.MerweScaledSigmaPoints(num_states, alpha=self.parameters['alpha'], beta=self.parameters['beta'], kappa=self.parameters['kappa'])
        self.filter = kalman.UnscentedKalmanFilter(num_states, num_measurements, self.parameters['dt'], measure, state_transition, points)
        
        if isinstance(x0, dict) or isinstance(x0, model.StateContainer):
            warn("Warning: Use UncertainData type if estimating filtering with uncertain data.")
            self.filter.x = array(list(x0.values()))
            self.filter.P = self.parameters['Q'] / 10
        elif isinstance(x0, UncertainData):
            x_mean = x0.mean
            self.filter.x = array(list(x_mean.values()))
            self.filter.P = x0.cov
        else:
            raise TypeError("TypeError: x0 initial state must be of type {{dict, UncertainData}}")

        if 'R' not in self.parameters:
                # Size of what's being measured (not output) 
                # This is determined by running the measure function on the first state
                self.parameters['R'] = diag([1.0e-3 for i in range(len(measure(self.filter.x)))])
        self.filter.Q = self.parameters['Q']
        self.filter.R = self.parameters['R']

    def estimate(self, t : float, u, z):
        """
        Perform one state estimation step (i.e., update the state estimate)

        Parameters
        ----------
        t : double
            Current timestamp in seconds (≥ 0.0)
            e.g., t = 3.4
        u : dict
            Measured inputs, with keys defined by model.inputs.
            e.g., u = {'i':3.2} given inputs = ['i']
        z : dict
            Measured outputs, with keys defined by model.outputs.
            e.g., z = {'t':12.4, 'v':3.3} given inputs = ['t', 'v']
        """
        assert t > self.t, "New time must be greater than previous"
        dt = t - self.t
        self.__input = u
        self.t = t
        self.filter.predict(dt=dt)
        self.filter.update(array(list(z.values())))
    
    @property
    def x(self) -> MultivariateNormalDist:
        """
        Getter for property 'x', the current estimated state. 

        Example
        -------
        state = observer.x
        """
        return MultivariateNormalDist(self.x0.keys(), self.filter.x, self.filter.P)
