# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from sys import path
path.insert(1, "/Users/cteubert/Desktop/python-prognostics-models-package/")

import unittest
from prog_algs import *
from prog_models import *

class MockProgModel(prognostics_model.PrognosticsModel):
    events = ['e1']
    states = ['a', 'b', 'c']
    inputs = ['i1', 'i2']
    outputs = ['o1']
    parameters = {
        'p1': 1.2,
        'x0': {'a': 1, 'b': 2, 'c': -3.2}
    }

    def __init__(self, options = {}):
        self.parameters.update(options)
        super().__init__()

    def initialize(self, u, z):
        return deepcopy(self.parameters['x0'])

    def next_state(self, t, x, u, dt):
        x['a']+= u['i1']*dt
        x['c']-= u['i2']
        return x

    def output(self, t, x):
        return {'o1': x['a'] + sum(x['b']) + x['c']}
    def event_state(self, t, x):
        return {'e1': max(1-t/5.0,0)}

    def threshold_met(self, t, x):
        return {'e1': self.event_state(t, x)['e1'] < 1e-6}

class TestModels(unittest.TestCase):
    def test_templates(self):
        import state_estimator_template
        import predictor_template
        m = MockProgModel({'process_noise': 0.0})
        se = state_estimator_template.TemplateStateEstimator(m)
        p = predictor_template.TemplatePredictor(m)

    def test_ukf(self):
        from prog_algs.state_estimators import unscented_kalman_filter
        m = MockProgModel({'process_noise': 0.0})
        unscented_kalman_filter.UnscentedKalmanFilter(m, 1.0, m.parameters['x0'])

if __name__ == '__main__':
    unittest.main()