# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from .test_state_estimators import run_tests as state_est_main
from .test_predictors import run_tests as pred_main
from .test_uncertain_data import run_tests as udata_main
from .test_examples import main as examples_main
from .test_metrics import run_tests as metrics_main
from .test_visualize import run_tests as visualize_main
from .test_tutorials import run_tests as tutorials_main

import unittest
import sys
from examples import basic_example
from io import StringIO 
from timeit import timeit
from unittest.mock import patch

def run_basic_ex():
    _stdout = sys.stdout
    sys.stdout = StringIO()
    with patch('matplotlib.pyplot') as p:
        basic_example.run_example()

    # Reset stdout 
    sys.stdout = _stdout

if __name__ == '__main__':
    l = unittest.TestLoader()

    try:
        print("\nExample Runtime: ", timeit(run_basic_ex, number=3))
    except Exception:
        print("\nFailed benchmarking")
        was_successful = False

    was_successful = True
    try:
        state_est_main()
    except Exception:
        was_successful = False
    
    try:
        pred_main()
    except Exception:
        was_successful = False

    try:
        udata_main()
    except Exception:
        was_successful = False

    try:
        visualize_main()
    except Exception:
        was_successful = False

    try:
        examples_main()
    except Exception:
        was_successful = False

    try:
        metrics_main()
    except Exception:
        was_successful = False

    try:
        tutorials_main()
    except Exception:
        was_successful = False

    if not was_successful:
        raise Exception('Tests Failed')
