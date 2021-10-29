# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
import sys
from io import StringIO 
from examples import *
from unittest.mock import patch


class TestExamples(unittest.TestCase):
    def test_main_ex(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        from examples import basic_example
        with patch('matplotlib.pyplot') as p:
            basic_example.run_example()

        # Reset stdout 
        sys.stdout = _stdout

    def test_benchmarking_ex(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        from examples import benchmarking_example
        with patch('matplotlib.pyplot') as p:
            benchmarking_example.run_example()

        # Reset stdout 
        sys.stdout = _stdout

    def test_new_state_est_ex(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        from examples import new_state_estimator_example
        with patch('matplotlib.pyplot') as p:
            new_state_estimator_example.run_example()

        # Reset stdout 
        sys.stdout = _stdout

    def test_thrown_obj_ex(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        from examples import thrown_object_example
        thrown_object_example.run_example()

        # Reset stdout 
        sys.stdout = _stdout

    def test_thrown_obj_ex(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        from examples import predict_specific_event
        thrown_object_example.run_example()

        # Reset stdout 
        sys.stdout = _stdout
    
    def test_measurement_ex(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        from examples import measurement_eqn_example
        with patch('matplotlib.pyplot') as p:
            measurement_eqn_example.run_example()

        # Reset stdout 
        sys.stdout = _stdout

    def test_utpredictor_ex(self):
        # set stdout (so it wont print)
        _stdout = sys.stdout
        sys.stdout = StringIO()

        from examples import utpredictor
        utpredictor.run_example()

        # Reset stdout 
        sys.stdout = _stdout

# This allows the module to be executed directly
def main():
    # This ensures that the directory containing ProgModelTemplate is in the python search directory
    import sys
    from os.path import dirname, join
    sys.path.append(join(dirname(__file__), ".."))

    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Examples")
    result = runner.run(l.loadTestsFromTestCase(TestExamples)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
