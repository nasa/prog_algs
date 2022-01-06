from prog_algs.predictors import monte_carlo2
import sys
from examples import basic_example, basic_example2
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

def run_basic_ex2():
    _stdout = sys.stdout
    sys.stdout = StringIO()
    with patch('matplotlib.pyplot') as p:
        basic_example2.run_example()
    # Reset stdout 
    sys.stdout = _stdout

print("\nExample Runtime 1: ", timeit(run_basic_ex, number=3))
print("\nExample Runtime 2: ", timeit(run_basic_ex2, number=3))
