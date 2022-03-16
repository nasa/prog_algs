# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from . import state_estimator
from filterpy import kalman
from numpy import diag, array
from warnings import warn
from ..uncertain_data import MultivariateNormalDist

class UnscentedKalmanFilter(state_estimator.StateEstimator):
    """
    An Unscented Kalman Filter (UKF) for state estimation

    This class defines logic for performing an unscented kalman filter with a Prognostics Model (see Prognostics Model Package). This filter uses measurement data with noise to generate a state estimate and covariance matrix. 

    Arguments:
        model: A Prognostics Model object
        x0: A dictionary of initial state values
        measurement_eqn (optional): A function that takes a dictionary of state values and returns a dictionary of measurement values. If not specified, the model's output function is used.

    The supported configuration parameters (keyword arguments) for UKF construction are described below:

    Constructor Configuration Parameters:
        alpha, beta, kappa: float
            UKF Scaling parameters
        t0 : float
            Starting time (s)
        dt : float 
            time step (s)
        R : numpy.ndarray
            Measurement noise covariance matrix (n_measurements x n_measurements). Only applicable if using measurement_eqn
    """
    default_parameters = {
        'alpha': 1, 
        'beta': 0, 
        'kappa': -1,
        't0': -1e-10,
        'dt': 1
    } 

    def __init__(self, model, x0, measurement_eqn = None, **kwargs):
        super().__init__(model, x0, measurement_eqn, **kwargs)

        self.__input = None
        self.x0 = x0

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

        if 'Q' in self.parameters:
            warn("UKF does not support Q parameter. Instead, set process noise, model.parameters['process_noise']")

        def state_transition(x, dt):
            x = {key: value for (key, value) in zip(x0.keys(), x)}
            Q_err = model.parameters['process_noise'].copy()
            model.parameters['process_noise'] = dict.fromkeys(Q_err, 0)
            x = model.next_state(x, self.__input, dt)
            model.parameters['process_noise'] = Q_err
            return array(list(x.values())).ravel()

        num_states = len(x0.keys())
        num_measurements = model.n_outputs
        points = kalman.MerweScaledSigmaPoints(num_states, alpha=self.parameters['alpha'], beta=self.parameters['beta'], kappa=self.parameters['kappa'])
        self.filter = kalman.UnscentedKalmanFilter(num_states, num_measurements, self.parameters['dt'], measure, state_transition, points)
        self.filter.x = array(list(x0.values())).ravel()
        self.filter.Q = diag([model.parameters['process_noise'][key] for key in x0.keys()])

        if measurement_eqn is not None:
            if 'R' in self.parameters:
                # If specified, use
                self.filter.R = self.parameters['R']
            else:
                z = measurement_eqn(x0)
                if all((key in model.outputs for key in z.keys())):
                    # Subset of measurements
                    self.filter.R = diag([model.parameters['measurement_noise'][key] for key in z.keys()])                
                # Otherwise use default
        else:
            if 'R' in self.parameters:
                warn("UKF does not support R parameter when not using a measurement_eqn. Instead, set measurement noise, model.parameters['measurement_noise']")
            # Not using measurement_eqn - then use model noise
            self.filter.R = diag([model.parameters['measurement_noise'][key] for key in model.outputs])

    def estimate(self, t, u, z):
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
    def x(self):
        """
        Getter for property 'x', the current estimated state. 

        Example
        -------
        state = observer.x
        """
        return MultivariateNormalDist(self.x0.keys(), self.filter.x, self.filter.P)
