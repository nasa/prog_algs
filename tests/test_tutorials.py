# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

class TestTutorials(unittest.TestCase):
    def test_tutorial_ipynb(self):
        with open('./tutorial.ipynb') as file:
            file_in = nbformat.read(file, nbformat.NO_CONVERT) 
        process = ExecutePreprocessor(timeout=600, kernel_name='python3')
        file_out = process.preprocess(file_in)

def run_tests():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Tutorials")
    result = runner.run(l.loadTestsFromTestCase(TestTutorials)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    run_tests()
    