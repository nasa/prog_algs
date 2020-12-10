# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
import unittest
import sys
from io import StringIO 

class TestExamples(unittest.TestCase):
    def test_main_ex(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        from examples import basic_example
        basic_example.run_example()

        # Reset stdout 
        sys.stdout = _stdout

    def test_benchmarking_ex(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        from examples import benchmarking_example
        benchmarking_example.run_example()

        # Reset stdout 
        sys.stdout = _stdout
