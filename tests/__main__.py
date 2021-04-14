# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from .test_state_estimators import TestStateEstimators
from .test_predictors import TestPredictors
from .test_integration import TestIntegration
from .test_uncertain_data import TestUncertainData
from .test_examples import TestExamples
from .test_misc import TestMisc
import unittest
import sys
from examples import basic_example
from io import StringIO 
from timeit import timeit

def run_basic_ex():
    _stdout = sys.stdout
    sys.stdout = StringIO()

    basic_example.run_example()

    # Reset stdout 
    sys.stdout = _stdout

if __name__ == '__main__':
    l = unittest.TestLoader()

    print("\nExample Runtime: ", timeit(run_basic_ex, number=3))

    print('\n\nTesting State Estimators')
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestStateEstimators))
    
    print('\n\nTesting Predictors')
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestPredictors))

    print('\n\nUncertain Data Tests')
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestUncertainData))

    print('\n\nExamples Tests')
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestExamples))

    print('\n\nIntegration Tests')
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestIntegration))

    print('\n\nMisc Tests')
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestMisc))