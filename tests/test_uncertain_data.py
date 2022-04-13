# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

import unittest
from prog_algs.uncertain_data import UnweightedSamples, MultivariateNormalDist, ScalarData
from numpy import array


class TestUncertainData(unittest.TestCase):
    def test_unweightedsamples(self):
        empty_samples = UnweightedSamples()
        self.assertEqual(empty_samples.size, 0)
        try:
            empty_samples.sample()
            self.fail() # Cant sample from 0 samples
        except ValueError:
            pass

        empty_samples.append({'a': 1, 'b': 2})
        self.assertEqual(empty_samples.size, 1)
        self.assertDictEqual(empty_samples.mean, {'a': 1, 'b': 2})
        samples = empty_samples.sample()
        self.assertDictEqual(samples[0], {'a': 1, 'b': 2})
        self.assertEqual(samples.size, 1)

        s = UnweightedSamples([{'a': 1, 'b':2}, {'a': 3, 'b':-2}])
        self.assertDictEqual(s.mean, {'a': 2, 'b': 0})
        self.assertEqual(s.size, 2)
        samples = s.sample(10)
        self.assertEqual(samples.size, 10)
        del s[0]
        self.assertEqual(s.size, 1)
        k = s.keys()
        self.assertEqual(len(s.raw_samples()), 1)
        s[0] = {'a': 2, 'b': 10}
        self.assertDictEqual(s[0], {'a': 2, 'b': 10})
        for i in range(50):
            s.append({'a': i, 'b': 9})
        covar = s.cov
        self.assertEqual(len(covar), 2)
        self.assertEqual(len(covar[0]), 2)

        # Test median value
        data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 3, 'b': 1}]
        data = UnweightedSamples(data)
        self.assertEqual(data.median, {'a': 2, 'b': 3})

        # Test percentage in bounds
        self.assertEqual(data.percentage_in_bounds([0, 2.5]), 
            {'a':0.6, 'b': 0.4})
        self.assertEqual(data.percentage_in_bounds({'a': [0, 2.5], 'b': [0, 1.5]}), 
            {'a':0.6, 'b': 0.2})

    def test_multivariatenormaldist(self):
        try: 
            dist = MultivariateNormalDist()
            self.fail()
        except Exception:
            pass
    
        dist = MultivariateNormalDist(['a', 'b'], array([2, 10]), array([[1, 0], [0, 1]]))
        self.assertDictEqual(dist.mean, {'a': 2, 'b':10})
        self.assertDictEqual(dist.median, {'a': 2, 'b':10})
        self.assertEqual(dist.sample().size, 1)
        self.assertEqual(dist.sample(10).size, 10)
        self.assertTrue((dist.cov == array([[1, 0], [0, 1]])).all())
        dist.percentage_in_bounds([0, 10])

    def test_scalardist(self):
        data = {'a': 12, 'b': 14}
        d = ScalarData(data)
        self.assertEqual(d.mean, data)
        self.assertEqual(d.median, data)
        self.assertListEqual(list(d.sample(10)), [data]*10)
        self.assertEqual(d.percentage_in_bounds([13, 20]), {'a': 0, 'b': 1})
        self.assertEqual(d.percentage_in_bounds([0, 10]), {'a': 0, 'b': 0})
        self.assertEqual(d.percentage_in_bounds([0, 20]), {'a': 1, 'b': 1})

    def test_pickle_unweightedsamples(self):
        data = {'a': 12, 'b': 14}
        d = ScalarData(data)
        import pickle # try pickle'ing
        pickle.dump(d, open('data_test.pkl', 'wb'))
        pickle_converted_result = pickle.load(open('data_test.pkl', 'rb'))
        self.assertEqual(d, pickle_converted_result)

    def test_pickle_unweightedsamples(self):
        s = UnweightedSamples([{'a': 1, 'b':2}, {'a': 3, 'b':-2}])
        import pickle # try pickle'ing
        pickle.dump(s, open('data_test.pkl', 'wb'))
        pickle_converted_result = pickle.load(open('data_test.pkl', 'rb'))
        self.assertEqual(s, pickle_converted_result)

    def test_pickle_multivariatenormaldist(self):
        dist = MultivariateNormalDist(['a', 'b'], array([2, 10]), array([[1, 0], [0, 1]]))
        import pickle # try pickle'ing
        pickle.dump(dist, open('data_test.pkl', 'wb'))
        pickle_converted_result = pickle.load(open('data_test.pkl', 'rb'))
        self.assertEqual(dist, pickle_converted_result)

    def test_scalardata_add_override(self):
        data = {'a': 12, 'b': 14}
        d = ScalarData(data)

        # Testing __add__ override
        mod_d = d + 0
        for k in d.keys():
            self.assertEqual(d.mean[k], mod_d.mean[k])
        mod_d = d + 5
        for k in d.keys():
            self.assertEqual(d.mean[k]+5, mod_d.mean[k])
        mod_d = d + -5
        for k in d.keys():
            self.assertEqual(d.mean[k]-5, mod_d.mean[k])
        with self.assertRaises(TypeError):
            # Test adding invalid type
            mod_d = d + []
            mod_d = d + {}
            mod_d = d + "test"
        # Also works with floats
        mod_d = d + 5.5
        for k in d.keys():
            self.assertEqual(d.mean[k]+5.5, mod_d.mean[k])

    def test_scalardata_radd_override(self):
        data = {'a': 12, 'b': 14}
        d = ScalarData(data)

        # Testing __radd__ override
        mod_d = 0 + d
        for k in d.keys():
            self.assertEqual(d.mean[k], mod_d.mean[k])
        mod_d = 5 + d
        for k in d.keys():
            self.assertEqual(d.mean[k]+5, mod_d.mean[k])
        mod_d = -5 + d
        for k in d.keys():
            self.assertEqual(d.mean[k]-5, mod_d.mean[k])
        with self.assertRaises(TypeError):
            # Test adding invalid type
            mod_d = [] + d
            mod_d = {} + d
            mod_d = "test" + d
        # Also works with floats
        mod_d = 5.5 + d
        for k in d.keys():
            self.assertEqual(d.mean[k]+5.5, mod_d.mean[k])

    def test_scalardata_iadd_override(self):
        data = {'a': 12, 'b': 14}
        data_copy = {'a': 12, 'b': 14}
        d = ScalarData(data)

        # Testing __iadd__ override
        d += 0
        for k in d.keys():
            self.assertEqual(d.mean[k], data_copy[k])
        d += 5
        for k in d.keys():
            self.assertEqual(d.mean[k], data_copy[k]+5)
        d += -5
        for k in d.keys():
            self.assertEqual(d.mean[k], data_copy[k])
        with self.assertRaises(TypeError):
            # Test adding invalid type
            d += []
            d += {}
            d += "test"
        # Also works with floats
        d += 5.5
        for k in d.keys():
            self.assertEqual(d.mean[k], data_copy[k] + 5.5)

    def test_scalardata_sub_override(self):
        data = {'a': 12, 'b': 14}
        d = ScalarData(data)

        # Testing __sub__ override
        mod_d = d - 0
        for k in d.keys():
            self.assertEqual(d.mean[k], mod_d.mean[k])
        mod_d = d - 5
        for k in d.keys():
            self.assertEqual(d.mean[k]-5, mod_d.mean[k])
        mod_d = d - -5
        for k in d.keys():
            self.assertEqual(d.mean[k]+5, mod_d.mean[k])
        with self.assertRaises(TypeError):
            # Test adding invalid type
            mod_d = d - []
            mod_d = d - {}
            mod_d = d - "test"
        # Also works with floats
        mod_d = d - 5.5
        for k in d.keys():
            self.assertEqual(d.mean[k]-5.5, mod_d.mean[k])

    def test_scalardata_rsub_override(self):
        data = {'a': 12, 'b': 14}
        d = ScalarData(data)

        # Testing __rsub__ override
        mod_d = 0 - d
        for k in d.keys():
            self.assertEqual(d.mean[k], mod_d.mean[k])
        mod_d = 5 - d
        for k in d.keys():
            self.assertEqual(d.mean[k]-5, mod_d.mean[k])
        mod_d = -5 - d
        for k in d.keys():
            self.assertEqual(d.mean[k]+5, mod_d.mean[k])
        with self.assertRaises(TypeError):
            # Test adding invalid type
            mod_d = [] - d
            mod_d = {} - d
            mod_d = "test" - d
        # Also works with floats
        mod_d = 5.5 - d
        for k in d.keys():
            self.assertEqual(d.mean[k]-5.5, mod_d.mean[k])

    def test_scalardata_isub_override(self):
        data = {'a': 12, 'b': 14}
        data_copy = {'a': 12, 'b': 14}
        d = ScalarData(data)

        # Testing __isub__ override
        d -= 0
        for k in d.keys():
            self.assertEqual(d.mean[k], data_copy[k])
        d -= 5
        for k in d.keys():
            self.assertEqual(d.mean[k], data_copy[k]-5)
        d -= -5
        for k in d.keys():
            self.assertEqual(d.mean[k], data_copy[k])
        with self.assertRaises(TypeError):
            # Test adding invalid type
            d -= []
            d -= {}
            d -= "test"
        # Also works with floats
        d -= 5.5
        for k in d.keys():
            self.assertEqual(d.mean[k], data_copy[k] - 5.5)

    def test_unweightedsamples_add_override(self):
        s = UnweightedSamples([{'a': 1, 'b':2}, {'a': 3, 'b':-2}])

        # Testing __add__ override
        mod_d = s + 0
        self.assertEqual(mod_d.data, [{'a': 1, 'b':2}, {'a': 3, 'b':-2}])
        mod_d = s + 5
        self.assertEqual(mod_d.data, [{'a': 6, 'b':7}, {'a': 8, 'b':3}])
        mod_d = s + -5
        self.assertEqual(mod_d.data, [{'a': -4, 'b':-3}, {'a': -2, 'b':-7}])
        with self.assertRaises(TypeError):
            # Test adding invalid type
            mod_d = s + []
            mod_d = s + {}
            mod_d = s + "test"
        # Also works with floats
        mod_d = s + 5.5
        self.assertEqual(mod_d.data, [{'a': 6.5, 'b':7.5}, {'a': 8.5, 'b':3.5}])

    def test_unweightedsamples_radd_override(self):
        s = UnweightedSamples([{'a': 1, 'b':2}, {'a': 3, 'b':-2}])

        # Testing __radd__ override
        mod_d = 0 + s
        self.assertEqual(mod_d.data, [{'a': 1, 'b':2}, {'a': 3, 'b':-2}])
        mod_d = 5 + s
        self.assertEqual(mod_d.data, [{'a': 6, 'b':7}, {'a': 8, 'b':3}])
        mod_d = -5 + s
        self.assertEqual(mod_d.data, [{'a': -4, 'b':-3}, {'a': -2, 'b':-7}])
        with self.assertRaises(TypeError):
            # Test adding invalid type
            mod_d = [] + s
            mod_d = {} + s
            mod_d = "test" + s
        # Also works with floats
        mod_d = 5.5 + s
        self.assertEqual(mod_d.data, [{'a': 6.5, 'b':7.5}, {'a': 8.5, 'b':3.5}])

    def test_unweightedsamples_iadd_override(self):
        s = UnweightedSamples([{'a': 1, 'b':2}, {'a': 3, 'b':-2}])

        # Testing __iadd__ override
        s += 0
        self.assertEqual(s.data, [{'a': 1, 'b':2}, {'a': 3, 'b':-2}])
        s += 5
        self.assertEqual(s.data, [{'a': 6, 'b':7}, {'a': 8, 'b':3}])
        s += -5
        self.assertEqual(s.data, [{'a': 1, 'b': 2}, {'a': 3, 'b': -2}])
        with self.assertRaises(TypeError):
            # Test adding invalid type
            s += []
            s += {}
            s += "test"
        # Also works with floats
        s += 5.5
        self.assertEqual(s.data, [{'a': 6.5, 'b': 7.5}, {'a': 8.5, 'b': 3.5}])

    def test_unweightedsamples_sub_override(self):
        s = UnweightedSamples([{'a': 1, 'b':2}, {'a': 3, 'b':-2}])

        # Testing __sub__ override
        mod_d = s - 0
        self.assertEqual(mod_d.data, [{'a': 1, 'b':2}, {'a': 3, 'b':-2}])
        mod_d = s - -5
        self.assertEqual(mod_d.data, [{'a': 6, 'b':7}, {'a': 8, 'b':3}])
        mod_d = s - 5
        self.assertEqual(mod_d.data, [{'a': -4, 'b':-3}, {'a': -2, 'b':-7}])
        with self.assertRaises(TypeError):
            # Test adding invalid type
            mod_d = s + []
            mod_d = s + {}
            mod_d = s + "test"
        # Also works with floats
        mod_d = s - 5.5
        self.assertEqual(mod_d.data, [{'a': -4.5, 'b': -3.5}, {'a': -2.5, 'b': -7.5}])

    def test_unweightedsamples_rsub_override(self):
        s = UnweightedSamples([{'a': 1, 'b':2}, {'a': 3, 'b':-2}])

        # Testing __rsub__ override
        mod_d = 0 - s
        self.assertEqual(mod_d.data, [{'a': 1, 'b':2}, {'a': 3, 'b':-2}])
        mod_d = -5 - s
        self.assertEqual(mod_d.data, [{'a': 6, 'b':7}, {'a': 8, 'b':3}])
        mod_d = 5 - s
        self.assertEqual(mod_d.data, [{'a': -4, 'b':-3}, {'a': -2, 'b':-7}])
        with self.assertRaises(TypeError):
            # Test adding invalid type
            mod_d = [] - s
            mod_d = {} - s
            mod_d = "test" - s
        # Also works with floats
        mod_d = 5.5 - s
        self.assertEqual(mod_d.data, [{'a': -4.5, 'b': -3.5}, {'a': -2.5, 'b': -7.5}])

    def test_unweightedsamples_isub_override(self):
        s = UnweightedSamples([{'a': 1, 'b':2}, {'a': 3, 'b':-2}])

        # Testing __isub__ override
        s -= 0
        self.assertEqual(s.data, [{'a': 1, 'b':2}, {'a': 3, 'b':-2}])
        s -= 5
        self.assertEqual(s.data, [{'a': -4, 'b': -3}, {'a': -2, 'b': -7}])
        s -= -5
        self.assertEqual(s.data, [{'a': 1, 'b': 2}, {'a': 3, 'b': -2}])
        with self.assertRaises(TypeError):
            # Test adding invalid type
            s -= []
            s -= {}
            s -= "test"
        # Also works with floats
        s -= 5.5
        self.assertEqual(s.data, [{'a': -4.5, 'b': -3.5}, {'a': -2.5, 'b': -7.5}])

    def test_MultivariateNormalDist_add_override(self):
        dist = MultivariateNormalDist(['a', 'b'], array([2, 10]), array([[1, 0], [0, 1]]))
        
        mod_d = dist + 0
        self.assertEqual(mod_d.mean, {'a': 2, 'b': 10})
        mod_d = dist + 5
        self.assertEqual(mod_d.mean, {'a': 7, 'b': 15})
        mod_d = dist + -5
        self.assertEqual(mod_d.mean, {'a': -3, 'b': 5})
        with self.assertRaises(TypeError):
            # Test adding invalid type
            mod_d = dist + []
            mod_d = dist + {}
            mod_d = dist + "test"
        # Also works with floats
        mod_d = dist + 5.5
        self.assertEqual(mod_d.mean, {'a': 7.5, 'b': 15.5})

# This allows the module to be executed directly    
def run_tests():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Uncertain Data")
    result = runner.run(l.loadTestsFromTestCase(TestUncertainData)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    run_tests()
