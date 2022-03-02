# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
from testbook import testbook

class TestTutorials(unittest.TestCase):
    def test_tutorial_ipynb(self):
        with testbook('./tutorial.ipynb', execute=False) as tb:
            # TESTING STATE ESTIMATION
            # Test importing and initializing BatteryCircuit, UnscentedKalmanFilter
            tb.execute_cell([0,1,2,3,4,5,6])
            self.assertEqual(tb.ref("m")._type, "BatteryCircuit")
            self.assertEqual(tb.ref("x0"), {'tb': 18.95, 'qb': 7856.3254, 'qcp': 0, 'qcs': 0})
            self.assertEqual(tb.ref("est")._type, "UnscentedKalmanFilter")
            

def run_tests():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Tutorials")
    result = runner.run(l.loadTestsFromTestCase(TestTutorials)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    run_tests()
