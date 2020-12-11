Getting Started
===============

The NASA Prognostics Algorithms Package is a python framework for defining, building, using, and testing Algorithms for prognostics (computation of remaining useful life) of engineering systems, and provides a set of prognostics algorithms developed within this framework, suitable for use in prognostics applications. It can be used in conjunction for the Prognostics Models Library to perform research in prognostics with prognostics systems. 

Installing Dependencies
-----------------------
You can do this using the included `requriements.txt` file. Use the following command:
    `pip install -r requirements.txt`

Use 
----
See the examples for examples of use. Run examples using the command `python -m examples.[Example name]` command (e.g., `python -m examples.sim_example`). The examples are summarized below:

* :download:`examples.basic_example <../examples/basic_example.py>`
    .. automodule:: examples.basic_example
    |

* :download:`examples.benchmarking_example <../examples/benchmarking_example.py>`
    .. automodule:: examples.benchmarking_example
    |

Extending

New State Estimators and Predictors are created by extending the :class:`prog_algs.state_estimators.state_estimator.StateEstimator` and :class:`prog_algs.predictors.predictor.Predictor` class, respectively.