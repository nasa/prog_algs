# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from . import state_estimator
import math
import numpy as np
import filterpy.monte_carlo
from numbers import Number
from scipy.stats import norm

class ParticleFilter(state_estimator.StateEstimator):
    """
    """
    t = 0 # last timestep

    parameters = { # Default Parameters
        'n': 0.1, # Sensor Noise
        'num_particles': 20, 
        'resample_fcn': filterpy.monte_carlo.residual_resample, # Resampling function ([weights]) -> [indexes]
        'x0_uncertainty': 0.5   # Initial State Uncertainty
                                # Can be:
                                #   1. scalar (standard deviation applied to all),
                                #   2. dict (stardard deviation for each)
                                #   Todo(CT): covar, function
    }


    def __init__(self, model, input_eqn, x0, options = {}):
        self._model = model
        self.input_eqn = input_eqn

        self.parameters.update(options)
        if isinstance(self.parameters['n'], Number):
            self.parameters['n'] = {key : self.parameters['n'] for key in model.outputs}
        # todo(CT): Check fields on n

        if isinstance(self.parameters['x0_uncertainty'], Number):
            # Build array inplace
            x = np.array(list(x0.values()))
            sd = np.array([self.parameters['x0_uncertainty']] * len(x0))
            samples = np.array([np.random.normal(x, sd) for i in range(self.parameters['num_particles'])])
            self.particles = np.array([{key: value for (key, value) in zip(model.states, x)} for x in samples])
        else:
            raise Exception
            #TODO(CT): Custom exception
        self.weights = np.array([1.0/len(self.particles)] * len(self.particles))
        # todo(ct): Maybe we should use numpy here
    
    def estimate(self, t, z):
        # todo(CT): assert t > self.t?
        # todo(CT): Should we change input_eqn to be an input rather than eqn
        dt = t - self.t
        weights = np.empty(len(self.particles))
        
        # Propogate and calculate weights
        for i in range(len(self.particles)):
            self.particles[i] = self._model.next_state(t, self.particles[i], self.input_eqn(t), dt) 
            zPredicted = self._model.output(t, self.particles[i])
            weights[i] = self.__likelihood(z, zPredicted)
        
        # Normalize
        total_weight = sum(self.weights)
        weights = np.array([weight/total_weight for weight in weights])

        # Resample
        indexes = self.parameters['resample_fcn'](self.weights)
        self.particles = np.array([self.particles[i] for i in indexes])

    @property
    def x(self):
        """
        Getter for property 'x', the current estimated state. 

        Example
        -------
        state = observer.x
        """
        return self.particles[0]
        # TODO(CT): Do something smarter

    def __likelihood(self, zActual, zPredicted):
        return sum([norm(zPredicted[key], self.parameters['n'][key]).pdf(zActual[key]) for key in zPredicted.keys()])
