# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
from testbook import testbook

class TestTutorials(unittest.TestCase):
    def test_tutorial_ipynb(self):
        with testbook('./tutorial.ipynb', execute=True) as tb:
            # # TESTING STATE ESTIMATION
            # # Test importing and initializing BatteryCircuit, UnscentedKalmanFilter
            # tb.execute_cell([0,1,2,3,4,5,6])
            # self.assertEqual(tb.ref("m")._type, "BatteryCircuit")
            # self.assertEqual(tb.ref("x0"), {'tb': 18.95, 'qb': 7856.3254, 'qcp': 0, 'qcs': 0})
            # self.assertEqual(tb.ref("est")._type, "UnscentedKalmanFilter")
            # # Test estimating the system state; compare against expected print output
            # tb.execute_cell([7,8])
            # state_estimation_prints = [line.strip() for line in tb.cell_output_text(8).splitlines()]
            # self.assertEqual(state_estimation_prints[0], "Prior State: {'tb': 18.95, 'qb': 7856.3254, 'qcp': 0.0, 'qcs': 0.0}")
            # self.assertEqual(state_estimation_prints[1], "SOC:  1.0")
            # self.assertEqual(state_estimation_prints[2], "Posterior State: {'tb': 20.15440859119512, 'qb': 7856.125354781064, 'qcp': 0.20143751745290445, 'qcs': 0.20014829013698857}")
            # self.assertEqual(state_estimation_prints[3], "SOC:  0.9999742773281554")
            # self.assertEqual(state_estimation_prints[4], "<Figure size 1000x900 with 9 Axes><Figure size 1000x900 with 9 Axes>") # Generating figure object

            # # TESTING STATE ESTIMATION
            # # Test importing and initializing MonteCarlo
            # tb.execute_cell([9,10,11,12,13,14])
            # tb.ref("mc")

            # TESTING TO ENSURE EXECUTION WITHOUT FAIL
            self.assertEqual(tb.ref("m")._type, "BatteryCircuit")
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
