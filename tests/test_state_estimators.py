# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
import unittest
from prog_algs.exceptions import ProgAlgTypeError

class MockProgModel():
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
        from prog_algs.state_estimators import unscented_kalman_filter
        m = MockProgModel()
        x0 = m.initialize()
        filt = unscented_kalman_filter.UnscentedKalmanFilter(m, x0)
        self.assertTrue(all(key in filt.x.mean for key in m.states))
        self.assertDictEqual(x0, filt.x.mean)
        filt.estimate(0.1, {'i1': 1, 'i2': 2}, {'o1': 0.8}) # note- if input is correct, o1 should be 0.9
        x = filt.x.mean
        self.assertFalse( x0 == x )
        self.assertFalse( {'a': 1.1, 'b': 2, 'c': -5.2, 't': 0} == x )

        # Between the model and sense outputs
        self.assertGreater(m.output(x)['o1'], 0.8)
        self.assertLess(m.output(x)['o1'], 0.9) 

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
        try:
            filt = filter(m, x0)
            self.fail()
        except ProgAlgTypeError:
            pass

        class IncompleteModel:
            states = ['a', 'b']
            def next_state(self):
                pass
            def output(self):
                pass
        m = IncompleteModel()
        x0 = {'a': 0, 'b': 2}
        try:
            filt = filter(m, x0)
            self.fail()
        except ProgAlgTypeError:
            pass

        class IncompleteModel:
            outputs = []
            def next_state(self):
                pass
            def output(self):
                pass
        m = IncompleteModel()
        x0 = {'a': 0, 'b': 2}
        try:
            filt = filter(m, x0)
            self.fail()
        except ProgAlgTypeError:
            pass

        class IncompleteModel:
            outputs = []
            states = ['a', 'b']
            def output(self):
                pass
        m = IncompleteModel()
        x0 = {'a': 0, 'b': 2}
        try:
            filt = filter(m, x0)
            self.fail()
        except ProgAlgTypeError:
            pass

        class IncompleteModel:
            outputs = []
            states = ['a', 'b']
            def next_state(self):
                pass
        m = IncompleteModel()
        x0 = {'a': 0, 'b': 2}
        try:
            filt = filter(m, x0)
            self.fail()
        except ProgAlgTypeError:
            pass

    def test_UKF_incorrect_input(self):
        from prog_algs.state_estimators import unscented_kalman_filter
        self.__incorrect_input_tests(unscented_kalman_filter.UnscentedKalmanFilter)

    def test_PF(self):
        from prog_algs.state_estimators import particle_filter
        m = MockProgModel()
        x0 = m.initialize()
        filt = particle_filter.ParticleFilter(m, x0)
        self.assertTrue(all(key in filt.x[0] for key in m.states))
        # self.assertDictEqual(x0, filt.x) // Not true - sample production means they may not be equal
        filt.estimate(0.1, {'i1': 1, 'i2': 2}, {'o1': 0.8}) # note- if input is correct, o1 should be 0.9
        x = filt.x.mean
        self.assertFalse( x0 == x )
        self.assertFalse( {'a': 1.1, 'b': 2, 'c': -5.2} == x )

        # Between the model and sense outputs
        self.assertGreater(m.output(x)['o1'], 0.7) # Should be between 0.8-0.9, choosing this gives some buffer for noise
        self.assertLess(m.output(x)['o1'], 0.95) # Should be between 0.8-0.9, choosing this gives some buffer for noise

        try:
            # Only given half of the inputs 
            filt.estimate(0.5, {'i1': 0}, {'o1': -2.0})
            self.fail("Shouldn't have made it here- only half the inputs")
        except Exception:
            pass

        try:
            # Missing output
            filt.estimate(0.5, {'i1': 0, 'i2': 0}, {})
            self.fail("Shouldn't have made it here- missing output")
        except Exception:
            pass

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
        from prog_algs.state_estimators import unscented_kalman_filter
        filt = unscented_kalman_filter.UnscentedKalmanFilter(m, x0)
        
        # Try using
        try:
            filt.estimate(0.1, {'i1': 1, 'i2': 2}, {'o1': -2.0}) 
            self.fail('Should have failed- missing o2')
        except Exception:
            pass
        
        filt.estimate(0.2, {'i1': 1, 'i2': 2}, {'o1': -2.0, 'o2': 7})

        # Add Measurement eqn
        def measurement_eqn(x):
            z = m.output(x)
            del z['o2']
            return z
        filt = unscented_kalman_filter.UnscentedKalmanFilter(m, x0, measurement_eqn=measurement_eqn)
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
        from prog_algs.state_estimators import particle_filter
        filt = particle_filter.ParticleFilter(m, x0)
        
        # Try using
        try:
            filt.estimate(0.1, {'i1': 1, 'i2': 2}, {'o1': -2.0}) 
            self.fail('Should have failed- missing o2')
        except Exception:
            pass
        
        filt.estimate(0.2, {'i1': 1, 'i2': 2}, {'o1': -2.0, 'o2': 7})

        # Add Measurement eqn
        def measurement_eqn(x):
            z = m.output(x)
            del z['o2']
            return z
        filt = particle_filter.ParticleFilter(m, x0, measurement_eqn=measurement_eqn)
        filt.estimate(0.1, {'i1': 1, 'i2': 2}, {'o1': -2.0}) 
        
    def test_PF_incorrect_input(self):
        from prog_algs.state_estimators import particle_filter
        self.__incorrect_input_tests(particle_filter.ParticleFilter)
