# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from . import state_estimator
from numpy import array, empty, random
import filterpy.monte_carlo
from numbers import Number
from scipy.stats import norm
from ..uncertain_data import UnweightedSamples
from ..exceptions import ProgAlgTypeError
from copy import deepcopy

class ParticleFilter(state_estimator.StateEstimator):
    """
    Estimates state using a particle filter algorithm

    Constructor parameters:
     * model (ProgModel): Model to be used in state estimation \n
        See: Prognostics Model Package \n
        A prognostics model to be used in state estimation
     * x0 (dict): Initial State \n
        Initial (starting) state, with keys defined by model.states \n
        e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
     * measurement_eqn (optional, function): Measurement equation (x)->z. Usually used in situations where what's measured don't exactly match the output (e.g., different unit, not ever output measured, etc.). see `examples.measurement_eqn_example`
     * options (optional, kwargs): configuration options\n
        Any additional configuration values. See default parameters. Additionally, the following configuration parameters are supported: \n
         * num_particles : Number of particles used in PF e.g., 20
         * resample_fcn : Resampling function ([weights]) -> [indexes] e.g., filterpy.monte_carlo.residual_resample
         * x0_uncertainty : Initial uncertainty in state e.g., 0.5
         * R (Number) : Measurement Noise. e.g., 0.1
    """
    t = 0 # last timestep

    default_parameters = {
            'n': 0.1, # Sensor Noise
            'num_particles': 20, 
            'resample_fcn': filterpy.monte_carlo.residual_resample, # Resampling function ([weights]) -> [indexes]
            'x0_uncertainty': 0.5   # Initial State Uncertainty
                                    # Can be:
                                    #   1. scalar (standard deviation applied to all),
                                    #   2. dict (stardard deviation for each)
                                    #   Todo(CT): covar, function
        }

    def __init__(self, model, x0, measurement_eqn = None, **kwargs):
        super().__init__(model, x0, measurement_eqn = measurement_eqn, **kwargs)

        if measurement_eqn is None:
            self.__measure = model.output
        else:
            self.__measure = measurement_eqn

        # State-estimator specific logic
        if isinstance(self.parameters['n'], Number):
            self.parameters['n'] = {key : self.parameters['n'] for key in self.__measure(x0).keys()}
        # todo(CT): Check fields on n

        if isinstance(self.parameters['x0_uncertainty'], Number):
            # Build array inplace
            x = array(list(x0.values()))
            sd = array([self.parameters['x0_uncertainty']] * len(x0))
            samples = array([random.normal(x, sd) for i in range(self.parameters['num_particles'])])
            self.particles = array([{key: value for (key, value) in zip(model.states, x)} for x in samples])
        else:
            raise ProgAlgTypeError
    
    def __str__(self):
        return "{} State Estimator".format(self.__class__)
        
    def estimate(self, t, u, z):
        # todo(CT): assert t > self.t?
        dt = t - self.t
        weights = empty(len(self.particles))
        
        # Optimization
        particles = self.particles
        next_state = self.model.next_state
        output = self.__measure
        noise_params = self.parameters['n']

        # Propogate and calculate weights
        for i in range(len(particles)):
            self.particles[i] = next_state(particles[i], u, dt) 
            zPredicted = output(self.particles[i])
            weights[i] = sum([norm(zPredicted[key], noise_params[key]).pdf(z[key]) for key in zPredicted.keys()])
        
        # Normalize
        total_weight = sum(weights)
        weights = array([weight/total_weight for weight in weights])

        # Resample
        indexes = self.parameters['resample_fcn'](weights)
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
