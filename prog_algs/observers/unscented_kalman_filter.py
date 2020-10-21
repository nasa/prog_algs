from . import observer
from filterpy import kalman
import numpy as np

class UnscentedKalmanFilter(observer.Observer):
    """
    """
    t = 0       # Last timestep
    parameters = {
        'alpha': 1, # UKF scaling param
        'beta': 0,  # UKF scaling param
        'kappa': -1, 
        't0': 0, # First timestep
        'dt': 1 # Time step
    } # Configuration parameters for ukf

    def __init__(self, model, options: dict):
        if 'input_eqn' not in options:
            raise Exception("Options must include 'input_eqn'")

        if 'x0' not in options:
            raise Exception("Options must include 'x0' (first state)")

        self._model = model

        self.parameters.update(options)

        if 'Q' not in self.parameters:
            self.parameters['Q'] = np.diag([1.0e-1 for i in model.states])

        if 'R' not in self.parameters:
            self.parameters['R'] = np.diag([1.0e-1 for i in model.outputs])

        self.t = self.parameters['t0']

        num_states = len(model.states)
        num_measurements = len(model.outputs)
        def measurement(x):
            x = {key: value for (key, value) in zip(model.states, x)}
            z = model.output(0, x)
            return np.array(list(z.values()))

        def state_transition(x, dt):
            x = {key: value for (key, value) in zip(model.states, x)}
            x = model.state(self.t, x, self.parameters['input_eqn'](self.t), dt)
            return np.array(list(x.values()))

        points = kalman.MerweScaledSigmaPoints(num_states, alpha=self.parameters['alpha'], beta=self.parameters['beta'], kappa=self.parameters['kappa'])
        self.filter = kalman.UnscentedKalmanFilter(num_states, num_measurements, self.parameters['dt'], measurement, state_transition, points)
        self.filter.x = np.array(list(self.parameters['x0'].values()))
        self.filter.Q = self.parameters['Q']
        self.filter.R = self.parameters['R']

    def step(self, t, z):
        dt = t - self.t
        self.t = t
        self.filter.predict(dt=dt)
        self.filter.update(np.array(list(z.values()))) #todo(ct): Add noise
    
    @property
    def x(self):
        return {key: value for (key, value) in zip(self._model.states, self.filter.x)}

    @property
    def Q(self):
        return self.filter.Q