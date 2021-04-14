# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from . import state_estimator
from filterpy import kalman
from numpy import diag, array
from ..uncertain_data import MultivariateNormalDist
from ..exceptions import ProgAlgTypeError
from copy import deepcopy

class UnscentedKalmanFilter(state_estimator.StateEstimator):
    """
    An Unscented Kalman Filter (UKF) for state estimation

    This class defines logic for performing an unscented kalman filter with a Prognostics Model (see Prognostics Model Package). This filter uses measurement data with noise to generate a state estimate and covariance matrix. 
    
    Constructor parameters:
     * model (ProgModel): Model to be used in state estimation \n
        See: Prognostics Model Package \n
        A prognostics model to be used in state estimation
     * x0 (dict): Initial State \n
        Initial (starting) state, with keys defined by model.states \n
        e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
     * measurement_eqn (optional, function): Measurement equation (x)->z. Usually used in situations where what's measured don't exactly match the output (e.g., different unit, not ever output measured, etc.). see `examples.measurement_eqn_example`
     * options (optional, kwargs): configuration options\n
        Dictionary of any additional configuration values. See default parameters. Additionally, the following configuration parameters are supported: \n
         * alpha, beta, kappa: UKF Scaling parameters
         * t0 : Starting time
         * dt : time step
         * Q : Process Noise Matrix 
         * R : Measurement Noise Matrix 
    """
    t = 0 # Last timestep
    default_parameters = {
        'alpha': 1,     # UKF scaling param
        'beta': 0,      # UKF scaling param
        'kappa': -1,    # UKF scaling param
        't0': 0,        # Starting time
        'dt': 1         # Time step
    } 

    def __init__(self, model, x0, measurement_eqn = None, **kwargs):
        super().__init__(model, x0, measurement_eqn, **kwargs)

        self._input = None
        self.t = self.parameters['t0']

        if measurement_eqn is None: 
            def measure(x):
                x = {key: value for (key, value) in zip(model.states, x)}
                z = model.output(x)
                return array(list(z.values()))
        else:
            def measure(x):
                x = {key: value for (key, value) in zip(model.states, x)}
                z = measurement_eqn(x)
                return array(list(z.values()))

        if 'Q' not in self.parameters:
            self.parameters['Q'] = diag([1.0e-1 for i in model.states])
        if 'R' not in self.parameters:
            self.parameters['R'] = diag([1.0e-1 for i in range(len(measure(x0.values())))])

        def state_transition(x, dt):
            x = {key: value for (key, value) in zip(model.states, x)}
            x = model.next_state(x, self._input, dt)
            return array(list(x.values()))

        num_states = len(model.states)
        num_measurements = len(model.outputs)
        points = kalman.MerweScaledSigmaPoints(num_states, alpha=self.parameters['alpha'], beta=self.parameters['beta'], kappa=self.parameters['kappa'])
        self.filter = kalman.UnscentedKalmanFilter(num_states, num_measurements, self.parameters['dt'], measure, state_transition, points)
        self.filter.x = array(list(x0.values()))
        self.filter.Q = self.parameters['Q']
        self.filter.R = self.parameters['R']

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
        dt = t - self.t
        self._input = u
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
        return MultivariateNormalDist(self.model.states, self.filter.x, self.filter.Q)
