# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
from testbook import testbook

class TestTutorials(unittest.TestCase):
    def test_tutorial_ipynb(self):
        with testbook('./tutorial.ipynb', execute=False) as tb:
            # Test importing BatteryCircuit, UnscentedKalmanFilter
            tb.execute_cell([0])

def run_tests():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Tutorials")
    result = runner.run(l.loadTestsFromTestCase(TestTutorials)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    run_tests()
