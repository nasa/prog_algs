# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from prog_algs.uncertain_data.uncertain_data import UncertainData
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
    Estimates state using a particle filter (PF) algorithm.

    This class defines logic for a PF using a Prognostics Model (see Prognostics Model Package). This filter uses measurement data with noise to estimate the state of the system using a particles. At each step, particles are predicted forward (with noise). Particles are resampled with replacement from the existing particles according to how well the particles match the observed measurements.

    The supported configuration parameters (keyword arguments) for UKF construction are described below:

    Constructor Configuration Parameters:
        t0 : float
            Starting time (s)
        dt : float 
            time step (s)
        num_particles : int
            Number of particles in particle filter
        resample_fcn : function 
            Resampling function ([weights]) -> [indexes] e.g., filterpy.monte_carlo.residual_resample
        x0_uncertainty : float or dict
            Initial uncertainty in state. Can be 1. scalar (standard deviation applied to all), or 2. dict (stardard deviation for each)\n
            e.g., 0.5 or {'state1': 0.5, 'state2': 0.2}
    """
    default_parameters = {
            't0': -1e-99,  # practically 0, but allowing for a 0 first estimate
            'num_particles': 20, 
            'resample_fcn': residual_resample,
            'x0_uncertainty': 0.5
        }

    def __init__(self, model, x0, measurement_eqn = None, **kwargs):
        super().__init__(model, x0, measurement_eqn = measurement_eqn, **kwargs)
        
        if measurement_eqn:
            self.__measure = measurement_eqn
            
            # update output_container
            z0 = measurement_eqn(x0)
            from prog_models.utils.containers import DictLikeMatrixWrapper
            class MeasureContainer(DictLikeMatrixWrapper):
                def __init__(self, z):
                    super().__init__(list(z0.keys()), z)
            self.model.OutputContainer = MeasureContainer
        else:
            self.__measure = model.output

        # Build array inplace
        if isinstance(x0, UncertainData):
            x = array(list(x0.mean.values()))
            sd = None # Using sampling method of UncertainData # dict of arrays
            # try using samples.normal?  no sd for scale 
            # samples = x0.sample() ()returns as unweightedsamples
            # samples = [random.normal(x[i]) for i in range(len(x0.sample().keys()))]
            samples = [array(x0.sample().key(k)) for k in x0.keys()]
            # print("A type:",type(samples[0]))
            # print(samples)

        elif 'x0_uncertainty' in self.parameters: # allows for x0 as UnweightedSamples
            warn("Warning: Use UncertainData type if estimating filtering with uncertain data.")
            x = array(list(x0.values()))
            if isinstance(self.parameters['x0_uncertainty'], dict):
                sd = array([self.parameters['x0_uncertainty'][key] for key in x0.keys()])
            elif isinstance(self.parameters['x0_uncertainty'], Number):
                sd = array([self.parameters['x0_uncertainty']] * len(x0))
            samples = [random.normal(x[i], sd[i], self.parameters['num_particles']) for i in range(len(x))]
            # loc, scale, size=None
            # print("B type:",type(samples[0]))
            # print(samples)

        else:
            raise ProgAlgTypeError("ProgAlgTypeError: x0 must be of type {{UncertainData}} or x0_uncertainty must be of type {{dict, Number}}.")
        self.particles = dict(zip(x0.keys(), samples))

        if 'R' in self.parameters:
            # For backwards compatibility
            warn("'R' is deprecated. Use 'measurement_noise' instead.", DeprecationWarning)
            self.parameters['measurement_noise'] = self.parameters['R']
        elif 'measurement_noise' not in self.parameters:
            self.parameters['measurement_noise'] = {key: 0.0 for key in x0.keys()}
    
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
        num_particles = self.parameters['num_particles']
        # Check which output keys are present (i.e., output of measurement function)
        measurement_keys = output({key: particles[key][0] for key in particles.keys()}).keys()
        zPredicted = {key: empty(num_particles) for key in measurement_keys}

        if self.model.is_vectorized:
            # Propagate particles state
            self.particles = apply_process_noise(next_state(particles, u, dt))

            # Get particle measurements
            zPredicted = apply_measurement_noise(output(self.particles))
        else:
            # Propogate and calculate weights
            for i in range(num_particles):
                x = {key: particles[key][i] for key in particles.keys()}
                x = next_state(x, u, dt) 
                x = apply_process_noise(x)
                for key in particles.keys():
                    self.particles[key][i] = x[key]
                z = output(x)
                z = apply_measurement_noise(z)
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
    def x(self):
        """
        Getter for property 'x', the current estimated state. 

        Example
        -------
        state = observer.x
        """
        return UnweightedSamples(self.particles)
