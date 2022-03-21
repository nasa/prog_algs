# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
import unittest
import numpy as np

from prog_models import PrognosticsModel, LinearModel
from prog_algs.exceptions import ProgAlgTypeError

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

class TestStateEstimators(unittest.TestCase):
    def test_state_est_template(self):
        from state_estimator_template import TemplateStateEstimator
        m = MockProgModel()
        se = TemplateStateEstimator(m, {'a': 0.0, 'b': 0.0, 'c': 0.0, 't':0.0})

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
        from prog_models.models import ThrownObject
        from prog_algs.state_estimators import UnscentedKalmanFilter

        m = ThrownObject(process_noise=5e-2, measurement_noise=5e-2)
        x_guess = {'x': 1.75, 'v': 35} # Guess of initial state, actual is {'x': 1.83, 'v': 40}

        filt = UnscentedKalmanFilter(m, x_guess)
        self.__test_state_est(filt, m)
        
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
        from prog_algs.state_estimators import UnscentedKalmanFilter
        self.__incorrect_input_tests(UnscentedKalmanFilter)

    def test_PF(self):
        from prog_algs.state_estimators import ParticleFilter
        from prog_models.models import ThrownObject

        m = ThrownObject(process_noise={'x': 1, 'v': 3}, measurement_noise=1, num_particles = 1000)
        x_guess = {'x': 1.75, 'v': 38.5} # Guess of initial state, actual is {'x': 1.83, 'v': 40}

        filt = ParticleFilter(m, x_guess)
        self.__test_state_est(filt, m)

        # Test ParticleFilter ScalarData
        from prog_algs.uncertain_data.scalar_data import ScalarData
        x_scalar = ScalarData({'x': 1.75, 'v': 38.5})
        filt_scalar = ParticleFilter(m, x_scalar, n_samples=200, x0_uncertainty=0.1)
        self.assertDictEqual(filt_scalar.x.mean, x_scalar.mean)
        self.assertTrue((filt_scalar.x.cov == x_scalar.cov).all())

        # Test ParticleFilter MultivariateNormalDist
        from numpy import array
        from prog_algs.uncertain_data.multivariate_normal_dist import MultivariateNormalDist
        x_mvnd = MultivariateNormalDist(['x', 'v'], array([2, 10]), array([[1, 0], [0, 1]]))
        filt_mvnd = ParticleFilter(m, x_mvnd)
        for k, v in filt_mvnd.x.mean.items():
            self.assertAlmostEqual(v, x_mvnd.mean[k], 0)
        # print(filt_mvnd.x.cov) # [[ 0.45593521 -0.19017238] [-0.19017238  1.75163799]]
        # print(x_mvnd.cov) #[[1 0] [0 1]]
        # self.assertTrue((filt_mvnd.x.cov == x_mvnd.cov).all()) FAILS

        # Test KalmanFilter UnweightedSamples
        from prog_algs.uncertain_data.unweighted_samples import UnweightedSamples
        x_us = UnweightedSamples([{'x': 1, 'v':2}, {'x': 3, 'v':-2}])
        filt_us = ParticleFilter(m, x_us)
        for k, v in filt_mvnd.x.mean.items():
            self.assertAlmostEqual(v, x_mvnd.mean[k], 0)
        self.assertTrue((filt_us.x.cov == x_us.cov).all())

        # m = MockProgModel(process_noise=5e-2, measurement_noise=0)
        # x0 = m.initialize()
        # filt = ParticleFilter(m, x0, n_samples=200, x0_uncertainty=0.1)
        # self.assertTrue(all(key in filt.x[0] for key in m.states))
        # # self.assertDictEqual(x0, filt.x) // Not true - sample production means they may not be equal
        # u = {'i1': 1, 'i2': 2}
        # x = m.next_state(m.initialize(), u, 0.1)
        # filt.estimate(0.1, u, m.output(x))  
        # x_est = filt.x.mean
        # self.assertFalse( x0 == x_est )
        # self.assertFalse( {'a': 1.1, 'b': 2, 'c': -5.2} == x_est )
        # Between the model and sense outputs
        # o_est = m.output(x_est)
        # o0 = m.output(x0)
        # self.assertGreater(o_est['o1'], 0.7) # Should be between 0.9-o0['o1'], choosing this gives some buffer for noise
        # self.assertLess(o_est['o1'], o0['o1']) # Should be between 0.8-0.9, choosing this gives some buffer for noise. Testing that the estimate is improving

        # with self.assertRaises(Exception):
        #     # Only given half of the inputs 
        #     filt.estimate(0.5, {}, {'o1': -2.0})

        # with self.assertRaises(Exception):
        #     # Missing output
        #     filt.estimate(0.5, {'i1': 0, 'i2': 0}, {})

    def test_measurement_eq_UKF(self):
        class MockProgModel2(MockProgModel):
            outputs = ['o1', 'o2']
            def output(self, x):
                return {
                    'o1': x['a'] + x['b'] + x['c'], 
                    'o2': 7
                    }

        m = MockProgModel2()
        x0 = m.initialize()

        # Setup
        from prog_algs.state_estimators import UnscentedKalmanFilter
        filt = UnscentedKalmanFilter(m, x0)
        
        # Try using
        filt.estimate(0.2, {'i1': 1, 'i2': 2}, {'o1': -2.0, 'o2': 7})

        # Add Measurement eqn
        def measurement_eqn(x):
            z = m.output(x)
            del z['o2']
            return z
        filt = UnscentedKalmanFilter(m, x0, measurement_eqn=measurement_eqn)
        filt.estimate(0.1, {'i1': 1, 'i2': 2}, {'o1': -2.0})

    def test_measurement_eq_PF(self):
        class MockProgModel2(MockProgModel):
            outputs = ['o1', 'o2']
            def output(self, x):
                return {
                    'o1': x['a'] + x['b'] + x['c'], 
                    'o2': 7
                    }

        m = MockProgModel2()
        x0 = m.initialize()

        # Setup
        from prog_algs.state_estimators import ParticleFilter
        filt = ParticleFilter(m, x0)
        
        # This one should work
        filt.estimate(0.2, {'i1': 1, 'i2': 2}, {'o1': -2.0, 'o2': 7})

        # Add Measurement eqn
        def measurement_eqn(x):
            z = m.output(x)
            del z['o2']
            return z
        filt = ParticleFilter(m, x0, measurement_eqn=measurement_eqn)
        filt.estimate(0.1, {'i1': 1, 'i2': 2}, {'o1': -2.0}) 
        
    def test_PF_incorrect_input(self):
        from prog_algs.state_estimators import ParticleFilter
        self.__incorrect_input_tests(ParticleFilter)

    def test_KF(self):
        from prog_algs.state_estimators import KalmanFilter
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
        x = m.initialize()
        # Test KalmanFilter ScalarData
        from prog_algs.uncertain_data.scalar_data import ScalarData
        x_scalar = ScalarData({'x': 1.75, 'v': 35})
        filt_scalar = KalmanFilter(m, x_scalar)
        self.assertDictEqual(filt_scalar.x.mean, x_scalar.mean)
        self.assertTrue((filt_scalar.x.cov == x_scalar.cov).all())

        # Test KalmanFilter MultivariateNormalDist
        from numpy import array
        from prog_algs.uncertain_data.multivariate_normal_dist import MultivariateNormalDist
        x_mvnd = MultivariateNormalDist(['x', 'v'], array([2, 10]), array([[1, 0], [0, 1]]))
        filt_mvnd = KalmanFilter(m, x_mvnd)
        self.assertDictEqual(filt_mvnd.x.mean, x_mvnd.mean)
        self.assertTrue((filt_mvnd.x.cov == x_mvnd.cov).all())

        # Test KalmanFilter UnweightedSamples
        from prog_algs.uncertain_data.unweighted_samples import UnweightedSamples
        x_us = UnweightedSamples([{'x': 1, 'v':2}, {'x': 3, 'v':-2}])
        filt_us = KalmanFilter(m, x_us)
        self.assertDictEqual(filt_us.x.mean, x_us.mean)
        self.assertTrue((filt_us.x.cov == x_us.cov).all())

        from prog_models.models import BatteryElectroChem

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
