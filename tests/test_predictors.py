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

class TestPredictors(unittest.TestCase):
    def test_pred_template(self):
        from predictor_template import TemplatePredictor
        pred = TemplatePredictor(None)

    def test_MC(self):
        from prog_algs.predictors import monte_carlo
        m = MockProgModel()
        mc = monte_carlo.MonteCarlo(m)
        
