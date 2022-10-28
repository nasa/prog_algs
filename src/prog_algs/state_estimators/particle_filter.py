# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from filterpy.monte_carlo import residual_resample
from numpy import array, empty, take, exp, max, take
from scipy.stats import norm
from warnings import warn

from prog_models.utils.containers import DictLikeMatrixWrapper

from . import state_estimator
from ..uncertain_data import UnweightedSamples, ScalarData, UncertainData
from ..exceptions import ProgAlgTypeError


class ParticleFilter(state_estimator.StateEstimator):
    """
    Estimates state using a Particle Filter (PF) algorithm.

    This class defines logic for a PF using a Prognostics Model (see Prognostics Model Package). This filter uses measurement data with noise to estimate the state of the system using a particles. At each step, particles are predicted forward (with noise). Particles are resampled with replacement from the existing particles according to how well the particles match the observed measurements.

    The supported configuration parameters (keyword arguments) for UKF construction are described below:

    Args:
        model (PrognosticsModel):
            A prognostics model to be used in state estimation
            See: Prognostics Model Package
        x0 (UncertainData, model.StateContainer, or dict):
            Initial (starting) state, with keys defined by model.states \n
            e.g., x = ScalarData({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

    Keyword Args:
        t0 (float, optional):
            Starting time (s)
        dt (float, optional): 
            time step (s)
        num_particles (int, optional):
            Number of particles in particle filter
        resample_fcn (function, optional):
            Resampling function ([weights]) -> [indexes] e.g., filterpy.monte_carlo.residual_resample
    """
    default_parameters = {
            't0': -1e-99,  # practically 0, but allowing for a 0 first estimate
            'num_particles': 20, 
            'resample_fcn': residual_resample,
        }

    def __init__(self, model, x0, **kwargs):
        super().__init__(model, x0, **kwargs)
        
        self._measure = model.output

        # Build array inplace
        if isinstance(x0, DictLikeMatrixWrapper) or isinstance(x0, dict):
            x0 = ScalarData(x0)
        elif not isinstance(x0, UncertainData):
            raise ProgAlgTypeError(f"ProgAlgTypeError: x0 must be of type UncertainData or StateContainer, was {type(x0)}.")

        sample_gen = x0.sample(self.parameters['num_particles'])
        samples = [array(sample_gen.key(k)) for k in x0.keys()]
        
        self.particles = model.StateContainer(array(samples))

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

        # Check Types
        if isinstance(u, dict):
            u = self.model.InputContainer(u)
        if isinstance(z, dict):
            z = self.model.OutputContainer(z)

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
        self.particles = self.model.StateContainer(array(samples))

    @property
    def x(self) -> UnweightedSamples:
        """
        Getter for property 'x', the current estimated state. 

        Example
        -------
        state = observer.x
        """
        return UnweightedSamples(self.particles, _type = self.model.StateContainer)
