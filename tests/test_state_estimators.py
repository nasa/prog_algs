# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
import unittest

class MockProgModel():
    events = ['e1']
    states = ['a', 'b', 'c']
    inputs = ['i1', 'i2']
    outputs = ['o1']
    parameters = {
        'p1': 1.2,
    }

    def initialize(self, u = {}, z = {}):
        return {'a': 1, 'b': 2, 'c': -3.2}

    def next_state(self, t, x, u, dt):
        x['a']+= u['i1']*dt
        x['c']-= u['i2']
        return x

    def output(self, t, x):
        return {'o1': x['a'] + x['b'] + x['c']}

    def event_state(self, t, x):
        return {'e1': max(1-t/5.0,0)}

    def threshold_met(self, t, x):
        return {'e1': self.event_state(t, x)['e1'] < 1e-6}

class TestStateEstimators(unittest.TestCase):
    def test_UKF(self):
        from prog_algs.state_estimators import unscented_kalman_filter
        m = MockProgModel()
        x0 = m.initialize()
        filt = unscented_kalman_filter.UnscentedKalmanFilter(m, x0)
        self.assertTrue(all(x in filt.x for x in m.states))
        self.assertDictEqual(x0, filt.x)
        filt.estimate(0.1, {'i1': 1, 'i2': 2}, {'o1': -2.0}) # note- if input is correct, o1 should be -2.1
        x = filt.x
        self.assertFalse( x0 == x )
        self.assertFalse( {'a': 1.1, 'b': 2, 'c': -5.2} == x )

        # Between the model and sense outputs
        self.assertGreater(m.output(0.1, x)['o1'], -2.1)
        self.assertLess(m.output(0.1, x)['o1'], -2.0) 

    def test_PF(self):
        from prog_algs.state_estimators import particle_filter
        m = MockProgModel()
        x0 = m.initialize()
        filt = particle_filter.ParticleFilter(m, x0)
        self.assertTrue(all(x in filt.x for x in m.states))
        # self.assertDictEqual(x0, filt.x) // Not true - sample production means they may not be equal
        print(filt.x)
        filt.estimate(0.1, {'i1': 1, 'i2': 2}, {'o1': -2.0}) # note- if input is correct, o1 should be -2.1
        x = filt.x
        print(x)
        self.assertFalse( x0 == x )
        self.assertFalse( {'a': 1.1, 'b': 2, 'c': -5.2} == x )

        filt.estimate(0.1, {'i1': 1, 'i2': 2}, {'o1': -3.8}) # note- if input is correct, o1 should be -2.1
        filt.estimate(0.1, {'i1': 1, 'i2': 2}, {'o1': -5.6}) # note- if input is correct, o1 should be -2.1

        # Between the model and sense outputs
        self.assertGreater(m.output(0.1, x)['o1'], -5.9)
        self.assertLess(m.output(0.1, x)['o1'], -5.6) 
        
