Getting Started
===============

The NASA Prognostics Algorithms Package is a python framework for defining, building, using, and testing Algorithms for prognostics (computation of remaining useful life) of engineering systems, and provides a set of prognostics algorithms developed within this framework, suitable for use in prognostics applications. It can be used in conjunction for the Prognostics Models Library to perform research in prognostics with prognostics systems. 

A few definitions:

* **events**: some state that can be predicted (e.g., system failure). An event has either occured or not. 

* **event state**: progress towards event occuring. Defined as a number where an event state of 0 indicates the event has occured and 1 indicates no progress towards the event (i.e., fully healthy operation for a failure event). For gradually occuring events (e.g., discharge) the number will progress from 1 to 0 as the event nears. In prognostics, event state is frequently called "State of Health"

* **inputs**: control applied to the system being modeled (e.g., current drawn from a battery)

* **outputs**: measured sensor values from a system (e.g., voltage and temperature of a battery)

* **states**: Internal parameters (typically hidden states) used to represent the state of the system- can be same as inputs/outputs but do not have to be. 

* **process noise**: stochastic process representing uncertainty in the model transition. 

* **measurement noise**: stochastic process representing uncertainty in the measurement process; e.g., sensor sensitivity, sensor misalignements, environmental effects 

Installing Dependencies
-----------------------
You can install dependencies using the included `requirements.txt` file. This file enumerates all the dependencies of this package. Use the following command to install dependencies:
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
* :download:`examples.measurement_eqn_example <../examples/measurement_eqn_example.py>`
    .. automodule:: examples.measurement_eqn_example
    |
* :download:`examples.new_state_estimator_example <../examples/new_state_estimator_example.py>`
    .. automodule:: examples.new_state_estimator_example
    |
* :download:`examples.playback <../examples/playback.py>`
    .. automodule:: examples.playback
    |

Extending
---------
New State Estimators and Predictors are created by extending the :class:`prog_algs.state_estimators.state_estimator.StateEstimator` and :class:`prog_algs.predictors.predictor.Predictor` class, respectively. 

See :download:`examples.new_state_estimator_example <../examples/new_state_estimator_example.py>` for an example of this approach.