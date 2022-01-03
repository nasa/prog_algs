# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
import unittest
from prog_models import PrognosticsModel
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

    def test_UKF(self):
        from prog_algs.state_estimators import UnscentedKalmanFilter
        m = MockProgModel(process_noise = 1e-3, measurement_noise = 1e-4)
        x0 = m.initialize()
        filt = UnscentedKalmanFilter(m, x0)
        self.assertTrue(all(key in filt.x.mean for key in m.states))
        self.assertDictEqual(x0, filt.x.mean)
        filt.estimate(0.1, {'i1': 1, 'i2': 2}, {'o1': 0.8}) 
        filt.estimate(0.15, {'i1': 1, 'i2': 2}, {'o1': 0.8})# note- if input is correct, o1 should be 0.9
        x = filt.x.mean
        self.assertFalse( x0 == x )
        self.assertFalse( {'a': 1.1, 'b': 2, 'c': -5.2, 't': 0} == x )

        # Between the model and sense outputs
        o = m.output(x)
        o0 = m.output(x0)
        self.assertGreater(o['o1'], 0.5)
        self.assertLess(o['o1'], o0['o1']) 

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
        m = MockProgModel(process_noise=5e-2, measurement_noise=0)
        x0 = m.initialize()
        filt = ParticleFilter(m, x0, n_samples=200, x0_uncertainty=0.1)
        self.assertTrue(all(key in filt.x[0] for key in m.states))
        # self.assertDictEqual(x0, filt.x) // Not true - sample production means they may not be equal
        u = {'i1': 1, 'i2': 2}
        x = m.next_state(m.initialize(), u, 0.1)
        filt.estimate(0.1, u, m.output(x))  
        x_est = filt.x.mean
        self.assertFalse( x0 == x_est )
        self.assertFalse( {'a': 1.1, 'b': 2, 'c': -5.2} == x_est )

        # Between the model and sense outputs
        o_est = m.output(x_est)
        o0 = m.output(x0)
        self.assertGreater(o_est['o1'], 0.7) # Should be between 0.9-o0['o1'], choosing this gives some buffer for noise
        self.assertLess(o_est['o1'], o0['o1']) # Should be between 0.8-0.9, choosing this gives some buffer for noise. Testing that the estimate is improving

        with self.assertRaises(Exception):
            # Only given half of the inputs 
            filt.estimate(0.5, {'i1': 0}, {'o1': -2.0})

        with self.assertRaises(Exception):
            # Missing output
            filt.estimate(0.5, {'i1': 0, 'i2': 0}, {})

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
