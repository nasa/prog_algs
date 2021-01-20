# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
import unittest
import sys
from io import StringIO 
from examples import *

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

    def test_new_state_est_ex(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        from examples import new_state_estimator_example
        new_state_estimator_example.run_example()

        # Reset stdout 
        sys.stdout = _stdout
    
    def test_measurement_ex(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        from examples import measurement_eqn_example
        measurement_eqn_example.run_example()

        # Reset stdout 
        sys.stdout = _stdout
