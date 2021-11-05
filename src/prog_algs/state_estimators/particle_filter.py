# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from . import state_estimator
from numpy import array, empty, random, take, exp, max, take, sort, log, pi
from filterpy.monte_carlo import residual_resample
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
    default_parameters = {
            'num_particles': 20, 
            'resample_fcn': residual_resample, # Resampling function ([weights]) -> [indexes]
            'x0_uncertainty': 0.5   # Initial State Uncertainty
                                    # Can be:
                                    #   1. scalar (standard deviation applied to all),
                                    #   2. dict (stardard deviation for each)
        }

    def __init__(self, model, x0, measurement_eqn = None, **kwargs):
        super().__init__(model, x0, measurement_eqn = measurement_eqn, **kwargs)

        self.t = 0 # last timestep
        
        if measurement_eqn is None:
            self.__measure = model.output
        else:
            self.__measure = measurement_eqn

        # Build array inplace
        x = array(list(x0.values()))

        if isinstance(self.parameters['x0_uncertainty'], dict):
            sd = array([self.parameters['x0_uncertainty'][key] for key in x0.keys()])
        elif isinstance(self.parameters['x0_uncertainty'], Number):
            sd = array([self.parameters['x0_uncertainty']] * len(x0))
        else:
            raise ProgAlgTypeError

        samples = [random.normal(
            x[i], sd[i], self.parameters['num_particles']) for i in range(len(x))]
        self.particles = dict(zip(x0.keys(), samples))
    
    def __str__(self):
        return "{} State Estimator".format(self.__class__)
        
    def estimate(self, t, u, z):
        assert t > self.t, "New time must be greater than previous"
        dt = t - self.t
        self.t = t

        # Optimization
        particles = self.particles
        next_state = self.model.next_state
        apply_process_noise = self.model.apply_process_noise
        output = self.__measure
        apply_measurement_noise = self.model.apply_measurement_noise
        noise_params = self.model.parameters['measurement_noise']

        # Propagate particles state
        self.particles = apply_process_noise(next_state(particles, u, dt))

        # Get particle measurements
        zPredicted = apply_measurement_noise(output(self.particles))

        # Calculate pdf values
        pdfs = array([norm(zPredicted[key], noise_params[key]).logpdf(z[key])
                      for key in zPredicted.keys()])

        # Calculate log weights
        log_weights = pdfs.sum(0)

        # Scale
        # We subtract the max log weights for numerical stability. 
        # Sometimes log weights can be a large negative value
        # when you exponentiate that value the computer will round the result to 0 for most of the weights (sometimes all of them) 
        # this causes problems when trying to sample from the particles. 
        # We shift them up by the max log weight (essentially making the max log weight 0) to help avoid that problem. 
        # When we normalize the weights by dividing by the sum of all the weights, that constant cancels out.
        max_log_weight = max(log_weights)
        scaled_weights = log_weights - max_log_weight

        # Convert to weights
        unnorm_weights = exp(scaled_weights)

        # Normalize
        total_weight = sum(unnorm_weights)
        self.weights = unnorm_weights / total_weight

        # Resample indices
        indexes = self.parameters['resample_fcn'](self.weights)

        # Resampled particles
        samples = [take(self.particles[state], indexes)
                   for state in self.particles.keys()]

        # Particles as a dictionary
        self.particles = dict(zip(self.particles.keys(), samples))

    @property
    def x(self):
        """
        Getter for property 'x', the current estimated state. 

        Example
        -------
        state = observer.x
        """
        return UnweightedSamples(self.particles)
