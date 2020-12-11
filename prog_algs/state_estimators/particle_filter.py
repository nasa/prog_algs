# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from . import state_estimator
from numpy import array, empty, random
import filterpy.monte_carlo
from numbers import Number
from scipy.stats import norm
from ..uncertain_data import UnweightedSamples
from ..exceptions import ProgAlgTypeError

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


    def __init__(self, model, x0, options = {}):
        self._model = model
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

        self.parameters.update(options)
        if isinstance(self.parameters['n'], Number):
            self.parameters['n'] = {key : self.parameters['n'] for key in model.outputs}
        # todo(CT): Check fields on n

        if isinstance(self.parameters['x0_uncertainty'], Number):
            # Build array inplace
            x = array(list(x0.values()))
            sd = array([self.parameters['x0_uncertainty']] * len(x0))
            samples = array([random.normal(x, sd) for i in range(self.parameters['num_particles'])])
            self.particles = array([{key: value for (key, value) in zip(model.states, x)} for x in samples])
        else:
            raise Exception
            #TODO(CT): Custom exception
        self.weights = array([1.0/len(self.particles)] * len(self.particles))
        # todo(ct): Maybe we should use numpy here
    
    def __str__(self):
        return "{} State Estimator".format(self.__class__)
        
    def estimate(self, t, u, z):
        # todo(CT): assert t > self.t?
        dt = t - self.t
        weights = empty(len(self.particles))
        
        # Optimization
        particles = self.particles
        next_state = self._model.next_state
        output = self._model.output
        noise_params = self.parameters['n']

        # Propogate and calculate weights
        for i in range(len(particles)):
            self.particles[i] = next_state(t, particles[i], u, dt) 
            zPredicted = output(t, self.particles[i])
            weights[i] = sum([norm(zPredicted[key], noise_params[key]).pdf(z[key]) for key in zPredicted.keys()])
        
        # Normalize
        total_weight = sum(self.weights)
        weights = array([weight/total_weight for weight in weights])

        # Resample
        indexes = self.parameters['resample_fcn'](self.weights)
        self.particles = array([self.particles[i] for i in indexes])

    @property
    def x(self):
        """
        Getter for property 'x', the current estimated state. 

        Example
        -------
        state = observer.x
        """
        return UnweightedSamples(self.particles)
