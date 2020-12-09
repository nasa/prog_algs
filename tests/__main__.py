# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
from .test_templates import TestTemplates
from .test_state_estimators import TestStateEstimators
import unittest

if __name__ == '__main__':
    l = unittest.TestLoader()

    print('\n\nTesting Templates')
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestTemplates))

    print('\n\nTesting State Estimators')
    unittest.TextTestRunner().run(l.loadTestsFromTestCase(TestStateEstimators))
    