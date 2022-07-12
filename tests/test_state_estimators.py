# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
import unittest
import numpy as np
import random

from prog_models import PrognosticsModel, LinearModel
from prog_models.models import ThrownObject, BatteryElectroChem
from prog_algs.state_estimators import ParticleFilter, KalmanFilter, UnscentedKalmanFilter
from prog_algs.exceptions import ProgAlgTypeError
from prog_algs.uncertain_data import ScalarData, MultivariateNormalDist, UnweightedSamples


def equal_cov(pair1, pair2):
    """
    Compare 2 covariance matricies, considering the order

    Args:
        pair1 (tuple[list, array[array[float]]]):
            keys (list[str]): Labels for keys
            covar (array[array[float]]): Covariance matrix
        pair2 (tuple[list, array[array[float]]]):
            keys (list[str]): Labels for keys
            covar (array[array[float]]): Covariance matrix
    """
    (keys1, cov1) = pair1
    (keys2, cov2) = pair2
    mapping = {i: keys2.index(key) for i, key in enumerate(keys1)}
    return all([cov1[i][j] == cov2[mapping[i]][mapping[j]] for i in range(len(keys1)) for j in range(len(keys1))])


class MockProgModel(PrognosticsModel):
    states = ['a', 'b', 'c', 't']
    inputs = ['i1', 'i2']
    outputs = ['o1']
    events = ['e1', 'e2']
    default_parameters = {
        'p1': 1.2,
    }

    def initialize(self, u = {}, z = {}):
        return {'a': 1, 'b': 5, 'c': -3.2, 't': 0}

    def next_state(self, x, u, dt):
        x['a']+= u['i1']*dt
        x['c']-= u['i2']
        x['t']+= dt
        return x

    def output(self, x):
        return {'o1': x['a'] + x['b'] + x['c']}
    
    def event_state(self, x):
        t = x['t']
        return {
            'e1': max(1-t/5.0,0),
            'e2': max(1-t/15.0,0)
            }

    def threshold_met(self, x):
        return {key : value < 1e-6 for (key, value) in self.event_state(x).items()}


class MockProgModel2(MockProgModel):
    outputs = ['o1', 'o2']
    def output(self, x):
        return self.OutputContainer({
            'o1': x['a'] + x['b'] + x['c'], 
            'o2': 7
            })


class TestStateEstimators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._m_mock = MockProgModel()
        cls._m = ThrownObject()
        def future_loading(t, x= None):
            return cls._m.InputContainer({})
        cls._future_loading = future_loading
        cls._s = cls._m.generate_surrogate([future_loading], states = ['v'], dt = 0.1, save_freq = 0.1)

    def test_state_est_template(self):
        from state_estimator_template import TemplateStateEstimator
        se = TemplateStateEstimator(self._m_mock, {'a': 0.0, 'b': 0.0, 'c': 0.0, 't':0.0})

    def __test_state_est(self, filt, m):
        x_guess = m.StateContainer(filt.x.mean)  # Might be new
        x = m.initialize()

        self.assertTrue(all(key in filt.x.mean for key in m.states))

        # run for a while
        dt = 0.01
        u = m.InputContainer({})
        for i in range(1250):
            # Get simulated output (would be measured in a real application)
            x = m.next_state(x, u, dt)
            x_guess = m.next_state(x_guess, u, dt)
            z = m.output(x)

            # Estimate New State
            filt.estimate((i+1)*dt, u, z)

        # Check results - make sure it converged
        x_est = filt.x.mean
        for key in m.states:
            # should be close to right
            self.assertAlmostEqual(x_est[key], x[key], delta=0.4)

    def test_UKF(self):
        m = ThrownObject(process_noise=5e-2, measurement_noise=5e-2)
        x_guess = {'x': 1.75, 'v': 35} # Guess of initial state, actual is {'x': 1.83, 'v': 40}

        filt = UnscentedKalmanFilter(m, x_guess)
        self.__test_state_est(filt, m)

        m = ThrownObject(process_noise=5e-2, measurement_noise=5e-2)

        # Test UnscentedKalmanFilter ScalarData
        x_scalar = ScalarData({'x': 1.75, 'v': 35})
        filt_scalar = UnscentedKalmanFilter(m, x_scalar)
        mean1 = filt_scalar.x.mean
        mean2 = x_scalar.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue(
            equal_cov(
                (list(x_scalar.keys()), x_scalar.cov), 
                (list(filt_scalar.x.keys()), filt_scalar.x.cov)))

        # Test UnscentedKalmanFilter MultivariateNormalDist
        x_mvnd = MultivariateNormalDist(['x', 'v'], np.array([2, 10]), np.array([[1, 0], [0, 1]]))
        filt_mvnd = UnscentedKalmanFilter(m, x_mvnd)
        mean1 = filt_mvnd.x.mean
        mean2 = x_mvnd.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue(
            equal_cov(
                (list(x_mvnd.keys()), x_mvnd.cov), 
                (list(filt_mvnd.x.keys()), filt_mvnd.x.cov)))

        # Now with a different order
        x_mvnd = MultivariateNormalDist(['v', 'x'], np.array([10, 2]), np.array([[1, 0], [0, 2]]))
        filt_mvnd = UnscentedKalmanFilter(m, x_mvnd)
        mean1 = filt_mvnd.x.mean
        mean2 = x_mvnd.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue(
            equal_cov(
                (list(x_mvnd.keys()), x_mvnd.cov), 
                (list(filt_mvnd.x.keys()), filt_mvnd.x.cov)), "Covs are not equal for multivariate in different order")

        # Test UnscentedKalmanFilter UnweightedSamples
        x_us = UnweightedSamples([{'x': 1, 'v':2}, {'x': 3, 'v':-2}])
        filt_us = UnscentedKalmanFilter(m, x_us)
        mean1 = filt_us.x.mean
        mean2 = x_us.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue(
            equal_cov(
                (list(x_us.keys()), x_us.cov), 
                (list(filt_us.x.keys()), filt_us.x.cov)))

        with self.assertRaises(Exception):
            # Not linear model
            UnscentedKalmanFilter(BatteryElectroChem, {})

        with self.assertRaises(Exception):
            # Missing states
            UnscentedKalmanFilter(ThrownObject, {})
        
    def __incorrect_input_tests(self, filter):
        class IncompleteModel:
            outputs = []
            states = ['a', 'b']
            def next_state(self):
                pass
            def output(self):
                pass
        m = IncompleteModel()
        x0 = {'a': 0, 'c': 2}
        with self.assertRaises(ProgAlgTypeError):
            filter(m, x0)

        class IncompleteModel:
            states = ['a', 'b']
            def next_state(self):
                pass
            def output(self):
                pass
        m = IncompleteModel()
        x0 = {'a': 0, 'b': 2}
        with self.assertRaises(ProgAlgTypeError):
            filter(m, x0)

        class IncompleteModel:
            outputs = []
            def next_state(self):
                pass
            def output(self):
                pass
        m = IncompleteModel()
        x0 = {'a': 0, 'b': 2}
        with self.assertRaises(ProgAlgTypeError):
            filter(m, x0)

        class IncompleteModel:
            outputs = []
            states = ['a', 'b']
            def output(self):
                pass
        m = IncompleteModel()
        x0 = {'a': 0, 'b': 2}
        with self.assertRaises(ProgAlgTypeError):
            filter(m, x0)
        class IncompleteModel:
            outputs = []
            states = ['a', 'b']
            def next_state(self):
                pass
        m = IncompleteModel()
        x0 = {'a': 0, 'b': 2}
        with self.assertRaises(ProgAlgTypeError):
            filter(m, x0)

    def test_UKF_incorrect_input(self):
        self.__incorrect_input_tests(UnscentedKalmanFilter)

    def test_PF(self):
        m = ThrownObject(process_noise={'x': 0.25, 'v': 0.25}, measurement_noise=1)
        x_guess = {'x': 1.75, 'v': 38.5} # Guess of initial state, actual is {'x': 1.83, 'v': 40}

        filt = ParticleFilter(m, x_guess, num_particles = 1000)
        self.__test_state_est(filt, m)

        # Test ParticleFilter ScalarData
        x_scalar = ScalarData({'x': 1.75, 'v': 38.5})
        filt_scalar = ParticleFilter(m, x_scalar, num_particles = 20) # Sample count does not affect ScalarData testing
        mean1 = filt_scalar.x.mean
        mean2 = x_scalar.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue((filt_scalar.x.cov == x_scalar.cov).all())
        
        # Test ParticleFilter MultivariateNormalDist
        x_mvnd = MultivariateNormalDist(['x', 'v'], np.array([2, 10]), np.array([[1, 0], [0, 1]]))
        filt_mvnd = ParticleFilter(m, x_mvnd, num_particles = 100000)
        for k, v in filt_mvnd.x.mean.items():
            self.assertAlmostEqual(v, x_mvnd.mean[k], delta = 0.01)
        for i in range(len(filt_mvnd.x.cov)):
            for j in range(len(filt_mvnd.x.cov[i])):
                self.assertAlmostEqual(filt_mvnd.x.cov[i][j], x_mvnd.cov[i][j], delta=0.1)

        # Test ParticleFilter UnweightedSamples
        uw_input = []
        x_bounds, v_bounds, x0_samples = 5, 5, 10000
        for i in range(x0_samples):
            uw_input.append({'x': random.randrange(-x_bounds, x_bounds), 'v': random.randrange(-v_bounds, v_bounds)})
        x_us = UnweightedSamples(uw_input)
        filt_us = ParticleFilter(m, x_us, num_particles = 100000)
        for k, v in filt_us.x.mean.items():
            self.assertAlmostEqual(v, x_us.mean[k], delta=0.02)
        for i in range(len(filt_us.x.cov)):
            for j in range(len(filt_us.x.cov[i])):
                self.assertAlmostEqual(filt_us.x.cov[i][j], x_us.cov[i][j], delta=0.1)

        # Test x0 if-else Control
        # Case 0: Both isinstance(x0, UncertainData) and x0_uncertainty parameter provided; expect x0_uncertainty to be skipped
        x_scalar = ScalarData({'x': 1.75, 'v': 38.5}) # Testing with ScalarData
        filt_scalar = ParticleFilter(m, x_scalar, num_particles = 20, x0_uncertainty = 0.5) # Sample count does not affect ScalarData testing
        mean1 = filt_scalar.x.mean
        mean2 = x_scalar.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue((filt_scalar.x.cov == x_scalar.cov).all())
        # Case 1: Only x0_uncertainty provided; expect a warning issued
        with self.assertWarns(Warning):
            filt_scalar = ParticleFilter(m, {'x': 1.75, 'v': 38.5}, x0_uncertainty = 0.5)
        # Case 2: Raise ProgAlgTypeError if x0 not UncertainData or x0_uncertainty not of type {{dict, Number}}.
        with self.assertRaises(ProgAlgTypeError):
            filt_scalar = ParticleFilter(m, {'x': 1.75, 'v': 38.5}, num_particles = 20, x0_uncertainty = [])
        
    def test_PF_incorrect_input(self):
        self.__incorrect_input_tests(ParticleFilter)

    def _test_state_est_surrogate(self, StateEst):
        m = self._m
        s = self._s

        # Setup ParticleFilter
        x0 = s.initialize()
        filt = StateEst(s, x0)

        # Test
        x0 = m.initialize()
        x = m.next_state(x0, {}, 0.1)
        filt.estimate(0.1, s.InputContainer({}), m.output(x))
        self.assertIsInstance(filt.x.mean, s.StateContainer)
        # mean = filt.x.mean
        # self.assertAlmostEqual(mean['x'], x['x'], delta=10)
        # self.assertAlmostEqual(mean['v'], x['v'], delta=1)
        # es = m.event_state(x)
        # for key in es.keys():
        #     self.assertAlmostEqual(mean[key], es[key], delta=0.2)

    def test_PF_Surrogate(self):
        self._test_state_est_surrogate(ParticleFilter)

    def test_UKF_Surrogate(self):
        self._test_state_est_surrogate(UnscentedKalmanFilter)

    def test_KF_Surrogate(self):
        self._test_state_est_surrogate(KalmanFilter)
    
    def test_KF(self):
        class ThrownObject(LinearModel):
            inputs = []  # no inputs, no way to control
            states = ['x', 'v']
            outputs = ['x']
            events = ['falling', 'impact']

            A = np.array([[0, 1], [0, 0]])
            E = np.array([[0], [-9.81]])
            C = np.array([[1, 0]])
            F = None # Will override method

            default_parameters = {
                'thrower_height': 1.83, 
                'throwing_speed': 40, 
                'g': -9.81 
            }

            def initialize(self, u=None, z=None):
                return self.StateContainer({
                    'x': self.parameters['thrower_height'], 
                    'v': self.parameters['throwing_speed'] 
                    })
            
            def threshold_met(self, x):
                return {
                    'falling': x['v'] < 0,
                    'impact': x['x'] <= 0
                }

            def event_state(self, x): 
                x_max = x['x'] + np.square(x['v'])/(-self.parameters['g']*2) # Use speed and position to estimate maximum height
                return {
                    'falling': np.maximum(x['v']/self.parameters['throwing_speed'],0),  # Throwing speed is max speed
                    'impact': np.maximum(x['x']/x_max,0) if x['v'] < 0 else 1  # 1 until falling begins, then it's fraction of height
                }

        m = ThrownObject(process_noise=5e-2, measurement_noise=5e-2)
        x_guess = {'x': 1.75, 'v': 35} # Guess of initial state, actual is {'x': 1.83, 'v': 40}

        filt = KalmanFilter(m, x_guess)

        self.__test_state_est(filt, m)

        m = ThrownObject(process_noise=5e-2, measurement_noise=5e-2)

        # Test KalmanFilter ScalarData
        x_scalar = ScalarData({'x': 1.75, 'v': 35})
        filt_scalar = KalmanFilter(m, x_scalar)
        mean1 = filt_scalar.x.mean
        mean2 = x_scalar.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue(
            equal_cov(
                (list(x_scalar.keys()), x_scalar.cov), 
                (list(filt_scalar.x.keys()), filt_scalar.x.cov)))

        # Test KalmanFilter MultivariateNormalDist
        x_mvnd = MultivariateNormalDist(['x', 'v'], np.array([2, 10]), np.array([[1, 0], [0, 1]]))
        filt_mvnd = KalmanFilter(m, x_mvnd)
        mean1 = filt_mvnd.x.mean
        mean2 = x_mvnd.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue(
            equal_cov(
                (list(x_mvnd.keys()), x_mvnd.cov), 
                (list(filt_mvnd.x.keys()), filt_mvnd.x.cov)))

        # Now with a different order
        x_mvnd = MultivariateNormalDist(['v', 'x'], np.array([10, 2]), np.array([[1, 0], [0, 2]]))
        filt_mvnd = KalmanFilter(m, x_mvnd)
        mean1 = filt_mvnd.x.mean
        mean2 = x_mvnd.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue(
            equal_cov(
                (list(x_mvnd.keys()), x_mvnd.cov), 
                (list(filt_mvnd.x.keys()), filt_mvnd.x.cov)), "Covs are not equal for multivariate in different order")

        # Test KalmanFilter UnweightedSamples
        x_us = UnweightedSamples([{'x': 1, 'v':2}, {'x': 3, 'v':-2}])
        filt_us = KalmanFilter(m, x_us)
        mean1 = filt_us.x.mean
        mean2 = x_us.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue(
            equal_cov(
                (list(x_us.keys()), x_us.cov), 
                (list(filt_us.x.keys()), filt_us.x.cov)))

        with self.assertRaises(Exception):
            # Not linear model
            KalmanFilter(BatteryElectroChem, {})

        with self.assertRaises(Exception):
            # Missing states
            KalmanFilter(ThrownObject, {})

# This allows the module to be executed directly    
def run_tests():
     # This ensures that the directory containing StateEstimatorTemplate is in the python search directory
    import sys
    from os.path import dirname, join
    sys.path.append(join(dirname(__file__), ".."))

    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting State Estimators")
    result = runner.run(l.loadTestsFromTestCase(TestStateEstimators)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    run_tests()
