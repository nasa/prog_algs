# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
import sys
import unittest

from prog_models import PrognosticsModel
# This ensures that the directory containing predictor_template is in the Python search directory
from os.path import dirname, join
sys.path.append(join(dirname(__file__), ".."))


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

class TestPredictors(unittest.TestCase):
    def test_pred_template(self):
        from predictor_template import TemplatePredictor
        m = MockProgModel()
        pred = TemplatePredictor(m)

    def test_UKP_ThrownObject(self):
        from prog_algs.predictors import UnscentedTransformPredictor
        from prog_algs.uncertain_data import MultivariateNormalDist
        from prog_models.models.thrown_object import ThrownObject
        m = ThrownObject()
        pred = UnscentedTransformPredictor(m)
        samples = MultivariateNormalDist(['x', 'v'], [1.83, 40], [[0.1, 0.01], [0.01, 0.1]])
        def future_loading(t, x={}):
            return {}

        mc_results = pred.predict(samples, future_loading, dt=0.01, save_freq=1)
        self.assertAlmostEqual(mc_results.time_of_event.mean['impact'], 8.21, 0)
        self.assertAlmostEqual(mc_results.time_of_event.mean['falling'], 4.15, 0)
        self.assertAlmostEqual(mc_results.times[-1], 9, 1)  # Saving every second, last time should be around the 1s after impact event (because one of the sigma points fails afterwards)

    def test_UKP_ThrownObject_One_Event(self):
        # Test thrown object, similar to test_UKP_ThrownObject, but with only the 'falling' event
        from prog_algs.predictors import UnscentedTransformPredictor
        from prog_algs.uncertain_data import MultivariateNormalDist
        from prog_models.models.thrown_object import ThrownObject
        m = ThrownObject()
        pred = UnscentedTransformPredictor(m)
        samples = MultivariateNormalDist(['x', 'v'], [1.83, 40], [[0.1, 0.01], [0.01, 0.1]])
        def future_loading(t, x={}):
            return {}

        mc_results = pred.predict(samples, future_loading, dt=0.01, events=['falling'], save_freq=1)
        self.assertAlmostEqual(mc_results.time_of_event.mean['falling'], 4.15, 0)
        self.assertTrue('impact' not in mc_results.time_of_event.mean)
        self.assertAlmostEqual(mc_results.times[-1], 4, 1)  # Saving every second, last time should be around the nearest 1s before falling event

    def test_UKP_Battery(self):
        from prog_algs.predictors import UnscentedTransformPredictor
        from prog_algs.uncertain_data import MultivariateNormalDist
        from prog_models.models import BatteryCircuit
        from prog_algs.state_estimators import UnscentedKalmanFilter

        def future_loading(t, x = None):
            # Variable (piece-wise) future loading scheme 
            if (t < 600):
                i = 2
            elif (t < 900):
                i = 1
            elif (t < 1800):
                i = 4
            elif (t < 3000):
                i = 2
            else:
                i = 3
            return {'i': i}

        batt = BatteryCircuit()

        ## State Estimation - perform a single ukf state estimate step
        filt = UnscentedKalmanFilter(batt, batt.parameters['x0'])

        example_measurements = {'t': 32.2, 'v': 3.915}
        t = 0.1
        filt.estimate(t, future_loading(t), example_measurements)

        ## Prediction - Predict EOD given current state
        # Setup prediction
        ut = UnscentedTransformPredictor(batt)

        # Predict with a step size of 0.1
        mc_results = ut.predict(filt.x, future_loading, dt=0.1)
        self.assertAlmostEqual(mc_results.time_of_event.mean['EOD'], 3004, -2)

        # Test Metrics
        from prog_algs.metrics import samples
        s = mc_results.time_of_event.sample(100).key('EOD')
        samples.eol_metrics(s)  # Kept for backwards compatibility

    def test_MC(self):
        from prog_models.models import ThrownObject
        from prog_algs.predictors import MonteCarlo
        m = ThrownObject()
        mc = MonteCarlo(m)
        def future_loading(t = None, x = None):
            return {}
            
        mc_results = mc.predict(m.initialize(), future_loading, dt=0.2, num_samples=3, save_freq=1)

    def test_prediction_mvnormaldist(self):
        from prog_algs.predictors import Prediction as MultivariateNormalDistPrediction
        from prog_algs.uncertain_data import MultivariateNormalDist
        times = list(range(10))
        covar = [[0.1, 0.01], [0.01, 0.1]]
        means = [{'a': 1+i/10, 'b': 2-i/5} for i in range(10)]
        states = [MultivariateNormalDist(means[i].keys(), means[i].values(), covar) for i in range(10)]
        p = MultivariateNormalDistPrediction(times, states)

        self.assertEqual(p.mean, means)
        self.assertEqual(p.snapshot(0), states[0])
        self.assertEqual(p.snapshot(-1), states[-1])
        self.assertEqual(p.time(0), times[0])
        self.assertEqual(p.times[0], times[0])
        self.assertEqual(p.time(-1), times[-1])
        self.assertEqual(p.times[-1], times[-1])

        # Out of range
        try:
            tmp = p.time(10)
            self.fail()
        except Exception:
            pass

        # Test pickle
        import pickle
        p2 = pickle.loads(pickle.dumps(p))
        self.assertEqual(p2, p)

    def test_prediction_uwsamples(self):
        from prog_algs.predictors.prediction import UnweightedSamplesPrediction
        from prog_algs.uncertain_data import UnweightedSamples
        times = list(range(10))
        states = [UnweightedSamples(list(range(10))), 
            UnweightedSamples(list(range(1, 11))), 
            UnweightedSamples(list(range(-1, 9)))]
        p = UnweightedSamplesPrediction(times, states)

        self.assertEqual(p[0], states[0])
        self.assertEqual(p.sample(0), states[0])
        self.assertEqual(p.sample(-1), states[-1])
        self.assertEqual(p.snapshot(0), UnweightedSamples([0, 1, -1]))
        self.assertEqual(p.snapshot(-1), UnweightedSamples([9, 10, 8]))
        self.assertEqual(p.time(0), times[0])
        self.assertEqual(p.times[0], times[0])
        self.assertEqual(p.time(-1), times[-1])

        # Out of range
        try:
            tmp = p[10]
            self.fail()
        except Exception:
            pass

        try:
            tmp = p.sample(10)
            self.fail()
        except Exception:
            pass

        try:
            tmp = p.time(10)
            self.fail()
        except Exception:
            pass

        # Bad type
        try:
            tmp = p.sample('abc')
            self.fail()
        except Exception:
            pass

        # Test pickle
        import pickle
        p2 = pickle.loads(pickle.dumps(p))
        self.assertEqual(p2, p)
    
    def test_prediction_profile(self):
        from prog_algs.predictors import ToEPredictionProfile
        from prog_algs.uncertain_data import ScalarData
        profile = ToEPredictionProfile()
        self.assertEqual(len(profile), 0)

        profile.add_prediction(0, ScalarData({'a': 1, 'b': 2, 'c': -3.2}))
        profile.add_prediction(1, ScalarData({'a': 1.1, 'b': 2.2, 'c': -3.1}))
        profile.add_prediction(0.5, ScalarData({'a': 1.05, 'b': 2.1, 'c': -3.15}))
        self.assertEqual(len(profile), 3)
        for (t_p, t_p_real) in zip(profile.keys(), [0, 0.5, 1]):
            self.assertAlmostEqual(t_p, t_p_real)

        profile[0.75] = ScalarData({'a': 1.075, 'b': 2.15, 'c': -3.125})
        self.assertEqual(len(profile), 4)
        for (t_p, t_p_real) in zip(profile.keys(), [0, 0.5, 0.75, 1]):
            self.assertAlmostEqual(t_p, t_p_real)
        self.assertEqual(profile[0.75], ScalarData({'a': 1.075, 'b': 2.15, 'c': -3.125}))

        del profile[0.5]
        self.assertEqual(len(profile), 3)
        for (t_p, t_p_real) in zip(profile.keys(), [0, 0.75, 1]):
            self.assertAlmostEqual(t_p, t_p_real)
        for ((t_p, toe), t_p_real) in zip(profile.items(), [0, 0.75, 1]):
            self.assertAlmostEqual(t_p, t_p_real)
        try:
            tmp = profile[0.5]
            # 0.5 doesn't exist anymore
        except Exception:
            pass

    def test_pickle_pred_template(self):
        from predictor_template import TemplatePredictor
        m = MockProgModel()
        pred = TemplatePredictor(m)
        import pickle # try pickle'ing
        pickle.dump(pred, open('predictor_test.pkl', 'wb'))
        pickle_converted_result = pickle.load(open('predictor_test.pkl', 'rb'))
        self.assertEqual(pred, pickle_converted_result)

# This allows the module to be executed directly    
def run_tests():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Predictor")
    result = runner.run(l.loadTestsFromTestCase(TestPredictors)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    run_tests()
