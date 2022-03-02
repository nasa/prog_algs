# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
import papermill as pm
from testbook import testbook

class TestTutorials(unittest.TestCase):
    def test_loading_notebook(self):
        with testbook('./tutorial.ipynb', execute=True) as tb:
            ipynb_batt = tb.ref("m")
            


def run_tests():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Tutorials")
    result = runner.run(l.loadTestsFromTestCase(TestTutorials)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    run_tests()
