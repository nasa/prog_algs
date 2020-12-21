# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

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
     * model (ProgModel): Model to be used in state estimation
     * x0 (dict): Initial State
     * measurement_eqn (optional, function): Measurement equation (x)->z. Usually used in situations where what's measured don't exactly match the output (e.g., different unit, not ever output measured, etc.). see `examples.measurement_eqn_example`
     * options (optional, dict): configuration options
    """
    t = 0 # Last timestep
    default_parameters = { # Default Parameters, used as config for UKF
        'alpha': 1,     # UKF scaling param
        'beta': 0,      # UKF scaling param
        'kappa': -1,    # UKF scaling param
        't0': 0,        # First timestep
        'dt': 1         # Time step
    } 

    def __init__(self, model, x0, measurement_eqn = None, options = {}):
        """
        Construct Unscented Kalman Filter

        Parameters
        ----------
        model : prog_models.prognostics_model.PrognosticsModel
            See: Prognostics Model Package
            A prognostics model to be used in state estimation
        input_eqn : function (t) -> z
            Function to generate an estimate of loading at future time t
        x0 : dict
            Initial (starting) state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        options : dict, optional
            Dictionary of any additional configuration values. See default parameters, above. Additionally, the following configuration parameters are supported:
            Q : Process Noise Matrix
            R : Measurement Noise Matrix
        """

        self.__model = model
        if not hasattr(model, 'output'):
            raise ProgAlgTypeError("model must have `output` method")
        if not hasattr(model, 'next_state'):
            raise ProgAlgTypeError("model must have `next_state` method")
        if not hasattr(model, 'outputs'):
            raise ProgAlgTypeError("model must have `outputs` property")
        if not hasattr(model, 'states'):
            raise ProgAlgTypeError("model must have `states` property")
        for key in model.states:
            if key not in x0:
                raise ProgAlgTypeError("x0 missing state `{}`".format(key))

        self._input = None
        self.parameters = deepcopy(default_parameters)
        self.parameters.update(options)

        self.t = self.parameters['t0']

        num_states = len(model.states)
        num_measurements = len(model.outputs)
        if measurement_eqn is None: 
            def measure(x):
                x = {key: value for (key, value) in zip(model.states, x)}
                z = model.output(0, x)
                return array(list(z.values()))
        else:
            def measure(x):
                x = {key: value for (key, value) in zip(model.states, x)}
                z = measurement_eqn(x)
                return array(list(z.values()))

        if 'Q' not in self.parameters:
            self.parameters['Q'] = diag([1.0e-1 for i in model.states])
        if 'R' not in self.parameters:
            self.parameters['R'] = diag([1.0e-1 for i in range(len(measure(x0)))])

        def state_transition(x, dt):
            x = {key: value for (key, value) in zip(model.states, x)}
            x = model.next_state(self.t, x, self._input, dt)
            return array(list(x.values()))

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
        return MultivariateNormalDist(self.__model.states, self.filter.x, self.filter.Q)
