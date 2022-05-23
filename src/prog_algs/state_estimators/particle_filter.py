# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from prog_algs.uncertain_data.uncertain_data import UncertainData
from typing import Callable
from . import state_estimator
from numpy import array, empty, random, take, exp, max, take
from filterpy.monte_carlo import residual_resample
from numbers import Number
from scipy.stats import norm
from ..uncertain_data import UnweightedSamples
from ..exceptions import ProgAlgTypeError
from warnings import warn


class ParticleFilter(state_estimator.StateEstimator):
    """
    Estimates state using a Particle Filter (PF) algorithm.

    This class defines logic for a PF using a Prognostics Model (see Prognostics Model Package). This filter uses measurement data with noise to estimate the state of the system using a particles. At each step, particles are predicted forward (with noise). Particles are resampled with replacement from the existing particles according to how well the particles match the observed measurements.

    The supported configuration parameters (keyword arguments) for UKF construction are described below:

    Args:
        model : ProgModel
            A prognostics model to be used in state estimation
            See: Prognostics Model Package
        x0 : UncertainData, model.StateContainer, or dict
            Initial (starting) state, with keys defined by model.states \n
            e.g., x = ScalarData({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

    Keyword Args:
        t0 : float
            Starting time (s)
        dt : float 
            time step (s)
        num_particles : int
            Number of particles in particle filter
        resample_fcn : function 
            Resampling function ([weights]) -> [indexes] e.g., filterpy.monte_carlo.residual_resample
    """
    default_parameters = {
            't0': -1e-99,  # practically 0, but allowing for a 0 first estimate
            'num_particles': 20, 
            'resample_fcn': residual_resample,
            'x0_uncertainty': 0.5
        }

    def __init__(self, model, x0, measurement_eqn : Callable = None, **kwargs):
        super().__init__(model, x0, measurement_eqn = measurement_eqn, **kwargs)
        
        if measurement_eqn:
            warn("Warning: measurement_eqn depreciated as of v1.3.1, will be removed in v1.4. Use Model subclassing instead. See examples.measurement_eqn_example")
            # update output_container
            from prog_models.utils.containers import DictLikeMatrixWrapper
            z0 = measurement_eqn(x0)
            class MeasureContainer(DictLikeMatrixWrapper):
                def __init__(self, z):
                    super().__init__(list(z0.keys()), z)

            def __measure(x):
                return MeasureContainer(measurement_eqn(x))
                
            self._measure = __measure
        else:
            self._measure = model.output

        # Caching for optimization
        paramters_x0_exist = 'x0_uncertainty' in self.parameters
        if paramters_x0_exist: # Only create these optimizations if x0_uncertainty exists as key in self.parameters
            parameters_x0_dict, parameters_x0_num = isinstance(self.parameters['x0_uncertainty'], dict), isinstance(self.parameters['x0_uncertainty'], Number)
        # Build array inplace
        if isinstance(x0, UncertainData):
            sample_gen = x0.sample(self.parameters['num_particles'])
            samples = [array(sample_gen.key(k)) for k in x0.keys()]
        elif paramters_x0_exist and (parameters_x0_dict or parameters_x0_num):
            warn("Warning: x0_uncertainty depreciated as of v1.3, will be removed in v1.4. Use UncertainData type if estimating filtering with uncertain data.")
            x = array(list(x0.values()))
            if parameters_x0_dict:
                sd = array([self.parameters['x0_uncertainty'][key] for key in x0.keys()])
            elif parameters_x0_num:
                sd = array([self.parameters['x0_uncertainty']] * len(x0))
            samples = [random.normal(x[i], sd[i], self.parameters['num_particles']) for i in range(len(x))]
        else:
            raise ProgAlgTypeError("ProgAlgTypeError: x0 must be of type UncertainData or x0_uncertainty must be of type [dict, Number].")
        self.particles = dict(zip(x0.keys(), samples))


        if 'R' in self.parameters:
            # For backwards compatibility
            warn("'R' is deprecated. Use 'measurement_noise' instead.", DeprecationWarning)
            self.parameters['measurement_noise'] = self.parameters['R']
        elif 'measurement_noise' not in self.parameters:
            self.parameters['measurement_noise'] = {key: 0.0 for key in x0.keys()}
    
    def __str__(self):
        return "{} State Estimator".format(self.__class__)
        
    def estimate(self, t : float, u, z):
        assert t > self.t, "New time must be greater than previous"
        dt = t - self.t
        self.t = t

        # Optimization
        particles = self.particles
        next_state = self.model.next_state
        apply_process_noise = self.model.apply_process_noise
        output = self._measure
        # apply_measurement_noise = self.model.apply_measurement_noise
        noise_params = self.model.parameters['measurement_noise']
        num_particles = self.parameters['num_particles']
        # Check which output keys are present (i.e., output of measurement function)
        measurement_keys = output(self.model.StateContainer({key: particles[key][0] for key in particles.keys()})).keys()
        zPredicted = {key: empty(num_particles) for key in measurement_keys}

        if self.model.is_vectorized:
            # Propagate particles state
            self.particles = apply_process_noise(next_state(particles, u, dt), dt)

            # Get particle measurements
            zPredicted = output(self.particles)
        else:
            # Propogate and calculate weights
            for i in range(num_particles):
                x = self.model.StateContainer({key: particles[key][i] for key in particles.keys()})
                x = next_state(x, u, dt) 
                x = apply_process_noise(x, dt)
                for key in particles.keys():
                    self.particles[key][i] = x[key]
                z = output(x)
                for key in measurement_keys:
                    zPredicted[key][i] = z[key]

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
    def x(self) -> UnweightedSamples:
        """
        Getter for property 'x', the current estimated state. 

        Example
        -------
        state = observer.x
        """
        return UnweightedSamples(self.particles, _type = self.model.StateContainer)
