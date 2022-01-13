# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from copy import deepcopy
import numpy as np
from filterpy import kalman
from . import StateEstimator
from ..uncertain_data import MultivariateNormalDist

class KalmanFilter(StateEstimator):
    """
    An Kalman Filter (KF) for state estimation

    This class defines logic for performing an kalman filter with a LinearModel (see Prognostics Model Package). This filter uses measurement data with noise to generate a state estimate and covariance matrix. 

    The supported configuration parameters (keyword arguments) for UKF construction are described below:

    Constructor Configuration Parameters:
        alpha: float
            KF Scaling parameter. An alpha > 1 turns this into a fading memory filter.
        t0 : float
            Starting time (s)
        dt : float 
            time step (s)
        Q : List[List[float]]
            Process Noise Matrix 
        R : List[List[float]]
            Measurement Noise Matrix 
    """
    default_parameters = {
        'alpha': 1, 
        't0': -1e-10,
        'dt': 1
    } 

    def __init__(self, model, x0, measurement_eqn = None, **kwargs):
        super().__init__(model, x0, measurement_eqn, **kwargs)

        self.x0 = x0

        if measurement_eqn is None: 
            def measure(x):
                x = {key: value for (key, value) in zip(x0.keys(), x)}
                R_err = model.parameters['measurement_noise'].copy()
                model.parameters['measurement_noise'] = dict.fromkeys(R_err, 0)
                z = model.output(x)
                model.parameters['measurement_noise'] = R_err
                return np.array(list(z.values())).ravel()
        else:
            def measure(x):
                x = {key: value for (key, value) in zip(x0.keys(), x)}
                z = measurement_eqn(x)
                return np.array(list(z.values())).ravel()

        if 'Q' not in self.parameters:
            self.parameters['Q'] = np.diag([1.0e-3 for i in x0.keys()])
        if 'R' not in self.parameters:
            # Size of what's being measured (not output) 
            # This is determined by running the measure function on the first state
            self.parameters['R'] = np.diag([1.0e-3 for i in range(len(measure(x0.values())))])

        num_states = len(x0.keys())
        num_inputs = len(model.inputs) + 1
        num_measurements = len(model.outputs)
        F = deepcopy(model.A)
        B = deepcopy(model.B)
        B = np.append(B, deepcopy(model.E).T, 0)

        self.filter = kalman.KalmanFilter(num_states, num_measurements, num_inputs)

        self.filter.x = np.array(list(x0.values())).ravel()
        self.filter.P = self.parameters['Q'] / 10
        self.filter.Q = self.parameters['Q']
        self.filter.R = self.parameters['R']
        self.filter.F = F
        self.filter.B = B

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
        # Create u array, ensuring order of model.inputs
        inputs = np.array([u[key] for key in self.model.inputs])

        # Add row of ones (to account for constant E term)
        inputs = np.array(inputs, [[1]]* len(inputs), 1)

        self.t = t
        self.filter.predict(u=inputs, dt=dt)

        # Create z array, ensuring order of model.outputs
        outputs = np.array([z[key] for key in self.model.outputs])

        self.filter.update(outputs)
    
    @property
    def x(self):
        """
        Getter for property 'x', the current estimated state. 

        Example
        -------
        state = observer.x
        """
        return MultivariateNormalDist(self.x0.keys(), self.filter.x, self.filter.P)
