# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from matplotlib import pyplot as plt
import sys
import unittest
from unittest.mock import patch

"""
This file includes tests that are too long to be run as part of the automated tests. Instead, these tests are run manually as part of the release process.
"""

class TestManual(unittest.TestCase):
    # Test playback example
    def test_playback_example(self):
        from examples import playback
        playback.run_example()

# This allows the module to be executed directly
def run_tests():
    unittest.main()

def main():
    # This ensures that the directory containing ProgModelTemplate is in the python search directory
    from os.path import dirname, join
    sys.path.append(join(dirname(__file__), ".."))

    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Manual")
    
    with patch('matplotlib.pyplot.show'):
        result = runner.run(l.loadTestsFromTestCase(TestManual)).wasSuccessful()
    plt.close('all')

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()

