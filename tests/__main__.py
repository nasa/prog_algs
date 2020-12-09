# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
from .test_state_estimators import TestStateEstimators
from .test_predictors import TestPredictors
from .test_integration import TestIntegration
from .test_misc import TestMisc
import unittest

if __name__ == '__main__':
    l = unittest.TestLoader()

    print('\n\nTesting State Estimators')
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestStateEstimators))
    
    print('\n\nTesting Predictors')
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestPredictors))

    print('\n\nIntegration Tests')
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestIntegration))

    print('\n\nMisc Tests')
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestMisc))