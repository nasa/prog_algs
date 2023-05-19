# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

from importlib import import_module
from io import StringIO 
from matplotlib import pyplot as plt
from os.path import dirname, join
import pkgutil
import sys
import unittest
from unittest.mock import patch

sys.path.append(join(dirname(__file__), ".."))  # puts examples in path
from examples import *

skipped_examples = ['playback']

def make_test_function(example):
    def test(self):
        ex = import_module("examples." + example)

        with patch('matplotlib.pyplot.show'):
            ex.run_example()
    return test


class TestExamples(unittest.TestCase):
    def setUp(self):
        # set stdout (so it wont print)
        self._stdout = sys.stdout
        sys.stdout = StringIO()

    def tearDown(self):
        # reset stdout
        sys.stdout = self._stdout

# This allows the module to be executed directly
def run_tests():
    unittest.main()
    
def main():
    # Create tests for each example
    for _, name, _ in pkgutil.iter_modules(['examples']):
        if name in skipped_examples:
            continue
        test_func = make_test_function(name)
        setattr(TestExamples, 'test_{0}'.format(name), test_func)   


    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Examples")
    with patch('matplotlib.pyplot.show'):
        result = runner.run(l.loadTestsFromTestCase(TestExamples)).wasSuccessful()

    plt.close('all')

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
