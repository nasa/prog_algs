# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
from prog_algs.uncertain_data import UnweightedSamples, MultivariateNormalDist, ScalarData
from prog_algs.metrics import prob_success
from prog_algs.metrics import calc_metrics as toe_metrics


class TestMetrics(unittest.TestCase):
    def test_toe_metrics_prev_name(self):
        # This is kept for backwards compatability
        from prog_algs.metrics import samples
        self.assertIs(samples.eol_metrics, toe_metrics)
        self.assertIs(samples.prob_success, prob_success)

    def test_toe_metrics_list_dict(self):
        # This is kept for backwards compatability

        # Common checks
        def check_metrics(metrics):
            # True for all keys
            for key in keys:
                self.assertAlmostEqual(metrics[key]['min'], 0)
                for percentile in ['0.01', '0.1', '1']:
                    # Not enough samples for these
                    self.assertIsNone(metrics[key]['percentiles'][percentile])

            # Key specific
            self.assertAlmostEqual(metrics['a']['percentiles']['10'], 1)
            self.assertAlmostEqual(metrics['a']['percentiles']['25'], 2)
            self.assertAlmostEqual(metrics['a']['mean'], 4.5)
            self.assertAlmostEqual(metrics['a']['percentiles']['50'], 5)
            self.assertAlmostEqual(metrics['a']['percentiles']['75'], 7)
            self.assertAlmostEqual(metrics['a']['mean absolute deviation'], 2.5)
            self.assertAlmostEqual(metrics['a']['max'], 9)
            self.assertAlmostEqual(metrics['a']['std'], 2.8722813232690143)
            self.assertAlmostEqual(metrics['b']['percentiles']['10'], 1.1)
            self.assertAlmostEqual(metrics['b']['percentiles']['25'], 2.2)
            self.assertAlmostEqual(metrics['b']['mean'], 4.95)
            self.assertAlmostEqual(metrics['b']['percentiles']['50'], 5.5)
            self.assertAlmostEqual(metrics['b']['percentiles']['75'], 7.7)
            self.assertAlmostEqual(metrics['b']['mean absolute deviation'], 2.75)
            self.assertAlmostEqual(metrics['b']['max'], 9.9)
            self.assertAlmostEqual(metrics['b']['std'], 3.159509455595916)
            self.assertAlmostEqual(metrics['c']['percentiles']['10'], 0.04)
            self.assertAlmostEqual(metrics['c']['percentiles']['25'], 0.16)
            self.assertAlmostEqual(metrics['c']['mean'], 1.14)
            self.assertAlmostEqual(metrics['c']['percentiles']['50'], 1.0)
            self.assertAlmostEqual(metrics['c']['percentiles']['75'], 1.96)
            self.assertAlmostEqual(metrics['c']['mean absolute deviation'], 0.928)
            self.assertAlmostEqual(metrics['c']['max'], 3.24)
            self.assertAlmostEqual(metrics['c']['std'], 1.074094967868298)

        u_samples = [{'a': i, 'b': i*1.1, 'c': (i/5)**2} for i in range(10)]
        keys = ['a', 'b', 'c']
        metrics = toe_metrics(u_samples)

        check_metrics(metrics)
        for key in keys:
            for key2 in ['mean absolute percentage error', 'relative accuracy', 'ground truth percentile']:   
                self.assertNotIn(key2, metrics[key])

        metrics = toe_metrics(u_samples, 5.0)
        check_metrics(metrics)
        for key in keys:
            for key2 in ['mean absolute percentage error', 'relative accuracy', 'ground truth percentile']:   
                self.assertIn(key2, metrics[key])
        self.assertAlmostEqual(metrics['a']['mean absolute error'], 2.5)
        self.assertAlmostEqual(metrics['b']['mean absolute error'], 2.75)
        self.assertAlmostEqual(metrics['c']['mean absolute error'], 3.86)

        metrics = toe_metrics(u_samples, ground_truth = {'a': 5.0, 'b': 4.5, 'c': 1.5})
        check_metrics(metrics)
        for key in keys:
            for key2 in ['mean absolute percentage error', 'relative accuracy', 'ground truth percentile']:   
                self.assertIn(key2, metrics[key])
        self.assertAlmostEqual(metrics['a']['mean absolute error'], 2.5)
        self.assertAlmostEqual(metrics['b']['mean absolute error'], 2.75)
        self.assertAlmostEqual(metrics['c']['mean absolute error'], 1.012)

        # Empty Sample Set
        try:
            toe_metrics([]) 
        except ValueError:
            pass

    def test_toe_metrics_mvnd(self):
        # Common checks
        def check_metrics(metrics):
            mean_dict = {key: value for (key, value) in zip(keys, mean)}
            for key in keys:
                self.assertAlmostEqual(metrics[key]['mean'], mean_dict[key])
                self.assertAlmostEqual(metrics[key]['median'], mean_dict[key])
                self.assertAlmostEqual(metrics[key]['percentiles']['50'], mean_dict[key])
                self.assertAlmostEqual(metrics[key]['std'], 1, 1)
        mean = [10, 11, 12]
        keys = ['a', 'b', 'c']
        covar = [
            [1, 0.1, 0.1], 
            [0.1, 1, 0.1], 
            [0.1, 0.1, 1]]
        dist = MultivariateNormalDist(keys, mean, covar)
        metrics = toe_metrics(dist)
        check_metrics(metrics)

        metrics = toe_metrics(dist, 11)
        check_metrics(metrics)
        self.assertAlmostEqual(metrics['a']['ground truth percentile'], 84.3, -1)
        self.assertAlmostEqual(metrics['b']['ground truth percentile'], 50, -1)
        self.assertAlmostEqual(metrics['c']['ground truth percentile'], 15.4, -1)

        # P(success)
        p_success = prob_success(dist, 11)
        self.assertAlmostEqual(p_success['a'], 0.1575, 1)
        self.assertAlmostEqual(p_success['b'], 0.5, 1)
        self.assertAlmostEqual(p_success['c'], 0.8425, 1)

    def test_toe_metrics_scalar(self):
        # Common checks
        def check_metrics(metrics):
            for key in scalar.keys():
                self.assertAlmostEqual(metrics[key]['min'], data[key])
                for value in metrics[key]['percentiles'].values():
                    self.assertAlmostEqual(value, data[key])
                self.assertAlmostEqual(metrics[key]['mean'], data[key])
                self.assertAlmostEqual(metrics[key]['median'], data[key])
                self.assertAlmostEqual(metrics[key]['max'], data[key])
                self.assertAlmostEqual(metrics[key]['std'], 0)
                self.assertAlmostEqual(metrics[key]['mean absolute deviation'], 0)

        data = {
                'a': 10,
                'b': 11,
                'c': 12
            }
        scalar = ScalarData(data)
        metrics = toe_metrics(scalar)
        check_metrics(metrics)

        # Check with ground truth
        metrics = toe_metrics(scalar, 11)
        check_metrics(metrics)
        for key in data.keys():
            for key2 in ['mean absolute percentage error', 'relative accuracy', 'ground truth percentile']:   
                self.assertIn(key2, metrics[key])
        self.assertAlmostEqual(metrics['a']['mean absolute error'], 1)
        self.assertAlmostEqual(metrics['b']['mean absolute error'], 0)
        self.assertAlmostEqual(metrics['c']['mean absolute error'], 1)

        # Check with limited samples
        metrics = toe_metrics(scalar, n_samples = 1000)
        for key in scalar.keys():
            self.assertIsNone(metrics[key]['percentiles']['0.01'])
            metrics[key]['percentiles']['0.01'] = data[key]  # Fill so we can check everything else below
        check_metrics(metrics)

        # Check broken samples
        try:
            toe_metrics(scalar, n_samples = 'abc')
        except TypeError:
            pass

        try:
            toe_metrics(scalar, n_samples = [])
        except TypeError:
            pass

        # P(success)
        p_success = prob_success(scalar, 11)
        self.assertAlmostEqual(p_success['a'], 0)  # After all samples
        self.assertAlmostEqual(p_success['b'], 0)  # Exactly equal
        self.assertAlmostEqual(p_success['c'], 1)  # Before all samples

    def test_toe_metrics_u_samples(self):
        # Common checks
        def check_metrics(metrics):
            # True for all keys
            for key in u_samples.keys():
                self.assertAlmostEqual(metrics[key]['min'], 0)
                for percentile in ['0.01', '0.1', '1']:
                    # Not enough samples for these
                    self.assertIsNone(metrics[key]['percentiles'][percentile])

            # Key specific
            self.assertAlmostEqual(metrics['a']['percentiles']['10'], 1)
            self.assertAlmostEqual(metrics['a']['percentiles']['25'], 2)
            self.assertAlmostEqual(metrics['a']['mean'], 4.5)
            self.assertAlmostEqual(metrics['a']['percentiles']['50'], 5)
            self.assertAlmostEqual(metrics['a']['percentiles']['75'], 7)
            self.assertAlmostEqual(metrics['a']['mean absolute deviation'], 2.5)
            self.assertAlmostEqual(metrics['a']['max'], 9)
            self.assertAlmostEqual(metrics['a']['std'], 2.8722813232690143)
            self.assertAlmostEqual(metrics['b']['percentiles']['10'], 1.1)
            self.assertAlmostEqual(metrics['b']['percentiles']['25'], 2.2)
            self.assertAlmostEqual(metrics['b']['mean'], 4.95)
            self.assertAlmostEqual(metrics['b']['percentiles']['50'], 5.5)
            self.assertAlmostEqual(metrics['b']['percentiles']['75'], 7.7)
            self.assertAlmostEqual(metrics['b']['mean absolute deviation'], 2.75)
            self.assertAlmostEqual(metrics['b']['max'], 9.9)
            self.assertAlmostEqual(metrics['b']['std'], 3.159509455595916)
            self.assertAlmostEqual(metrics['c']['percentiles']['10'], 0.04)
            self.assertAlmostEqual(metrics['c']['percentiles']['25'], 0.16)
            self.assertAlmostEqual(metrics['c']['mean'], 1.14)
            self.assertAlmostEqual(metrics['c']['percentiles']['50'], 1.0)
            self.assertAlmostEqual(metrics['c']['percentiles']['75'], 1.96)
            self.assertAlmostEqual(metrics['c']['mean absolute deviation'], 0.928)
            self.assertAlmostEqual(metrics['c']['max'], 3.24)
            self.assertAlmostEqual(metrics['c']['std'], 1.074094967868298)

        data = [{'a': i, 'b': i*1.1, 'c': (i/5)**2} for i in range(10)]
        u_samples = UnweightedSamples(data)
        metrics = toe_metrics(u_samples)

        check_metrics(metrics)
        for key in u_samples.keys():
            for key2 in ['mean absolute percentage error', 'relative accuracy', 'ground truth percentile']:   
                self.assertNotIn(key2, metrics[key])

        metrics = toe_metrics(u_samples, 5.0)
        check_metrics(metrics)
        for key in u_samples.keys():
            for key2 in ['mean absolute percentage error', 'relative accuracy', 'ground truth percentile']:   
                self.assertIn(key2, metrics[key])
        self.assertAlmostEqual(metrics['a']['mean absolute error'], 2.5)
        self.assertAlmostEqual(metrics['b']['mean absolute error'], 2.75)
        self.assertAlmostEqual(metrics['c']['mean absolute error'], 3.86)

        metrics = toe_metrics(u_samples, ground_truth = {'a': 5.0, 'b': 4.5, 'c': 1.5})
        check_metrics(metrics)
        for key in u_samples.keys():
            for key2 in ['mean absolute percentage error', 'relative accuracy', 'ground truth percentile']:   
                self.assertIn(key2, metrics[key])
        self.assertAlmostEqual(metrics['a']['mean absolute error'], 2.5)
        self.assertAlmostEqual(metrics['b']['mean absolute error'], 2.75)
        self.assertAlmostEqual(metrics['c']['mean absolute error'], 1.012)

        # Empty Sample Set
        try:
            toe_metrics(UnweightedSamples([])) 
        except ValueError:
            pass

        # P(success)
        p_success = prob_success(u_samples, 5.0)
        self.assertAlmostEqual(p_success['a'], 0.4)
        self.assertAlmostEqual(p_success['b'], 0.5)
        self.assertAlmostEqual(p_success['c'], 0)

    def test_toe_metrics_ground_truth(self):
        # Wrong type 
        try:
            toe_metrics(UnweightedSamples([{'a': 1.0}]), ground_truth='abc')
        except TypeError:
            pass

        # Below samples
        toe_metrics(UnweightedSamples([{'a': 1.0}]), ground_truth=0)

        # At sample
        toe_metrics(UnweightedSamples([{'a': 1.0}]), ground_truth=1)

        # Above samples
        toe_metrics(UnweightedSamples([{'a': 1.0}]), ground_truth=2)

        # NaN
        toe_metrics(UnweightedSamples([{'a': 1.0}]), ground_truth=float('nan'))

        # Inf
        toe_metrics(UnweightedSamples([{'a': 1.0}]), ground_truth=float('inf'))

        # -Inf
        toe_metrics(UnweightedSamples([{'a': 1.0}]), ground_truth=-float('inf'))

# This allows the module to be executed directly    
def run_tests():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Metrics")
    result = runner.run(l.loadTestsFromTestCase(TestMetrics)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    run_tests()
