Getting Started
===============
.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/nasa/prog_algs/master?labpath=tutorial.ipynb

The NASA Prognostics Algorithms Package is a Python framework for defining, building, using, and testing Algorithms for prognostics (computation of remaining useful life) of engineering systems, and provides a set of prognostics algorithms developed within this framework, suitable for use in prognostics applications. It can be used in conjunction with the Prognostics Models Package (`prog_models`) to perform research in prognostics with prognostics systems.

Installing
-----------------------

Installing from pip (recommended)
********************************************
The latest stable release of `prog_algs` is hosted on PyPi. For most users (unless you want to contribute to the development of `prog_algs`), this version will be adequate. To install from the command line, use the following command:

.. code-block:: console

    $ pip install prog_algs

Installing Pre-Release Versions with GitHub
********************************************
For users who would like to contribute to `prog_algs` or would like to use pre-release features can do so using the 'dev' branch (or a feature branch) on the `prog_algs GitHub repo <https://github.com/nasa/prog_algs>`__. This isn't recommended for most users as this version may be unstable. To use this version, use the following commands:

.. code-block:: console

    $ git clone https://github.com/nasa/prog_algs
    $ cd prog_algs
    $ git checkout dev 
    $ pip install -e .

Summary
---------
A few definitions to get started:

* **events**: something that can be predicted (e.g., system failure). An event has either occurred or not. 

* **event state**: progress towards event occurring. Defined as a number where an event state of 0 indicates the event has occurred and 1 indicates no progress towards the event (i.e., fully healthy operation for a failure event). For gradually occurring events (e.g., discharge) the number will progress from 1 to 0 as the event nears. In prognostics, event state is frequently called "State of Health".

* **inputs**: control applied to the system being modeled (e.g., current drawn from a battery).

* **outputs**: measured sensor values from a system (e.g., voltage and temperature of a battery).

* **performance metrics**: performance characteristics of a system that are a function of system state, but are not directly measured.

* **states**: Internal parameters (typically hidden states) used to represent the state of the system- can be same as inputs/outputs but do not have to be. 

* **process noise**: representing uncertainty in the model transition (e.g., model uncertainty). 

* **measurement noise**: representing uncertainty in the measurement process (e.g., sensor sensitivity, sensor misalignements, environmental effects).

The structure of the packages is illustrated below:

.. image:: images/package_structure.png

Prognostics is performed using `State Estimators <state_estimators.html>`__ and `Predictors <predictors.html>`__. State Estimators are resposible for estimating the current state of the modeled system using sensor data and a prognostics model (see: `prog_models package <https://github.com/nasa/prog_models>`__). The state estimator then produces an estimate of the system state with uncertainty in the form of an `uncertain data object <uncertain_data.html>`__. This state estimate is used by the predictor to predict when events will occur (Time of Event, ToE - returned as an `uncertain data object <uncertain_data.html>`__), and future system states (returned as a `Prediction object <prediction.html#id1>`__).

Data Structures
***************

A few custom data structures are available for storing and manipulating prognostics data of various forms. These structures are listed below and desribed on their respective pages:
 * `SimResult (from prog_models) <https://nasa.github.io/prog_models/simresult.html>`__ : The result of a single simulation (without uncertainty). Can be used to store inputs, outputs, states, event_states, observables, etc. Is returned by the model.simulate_to* methods.
 * `UncertainData <uncertain_data.html>`__ : Used throughout the package to represent data with uncertainty. There are a variety of subclasses of UncertainData to represent data with uncertainty in different forms (e.g., ScalarData, MultivariateNormalDist, UnweightedSamples). Notibly, this is used to represent the output of a StateEstimator's `estimate` method, individual snapshots of a prediction, and the time of event estimate from a predictor's `predict` method.
 * `Prediction <prediction.html#id1>`__ : Prediction of future values (with uncertainty) of some variable (e.g., input, state, output, event_states, etc.). The `predict` method of predictors return this. 
 * `ToEPredictionProfile <prediction.html#toe-prediction-profile>`__ : The result of multiple predictions, including time of prediction. This data structure can be treated as a dictionary of time of prediction to toe prediction. 

Use 
----
The best way to learn how to use `prog_algs` is through the `tutorial <https://mybinder.org/v2/gh/nasa/prog_algs/master?labpath=tutorial.ipynb>`__. There are also a number of examples which show different aspects of the package, summarized and linked below:

* :download:`examples.basic_example <../examples/basic_example.py>`
    .. automodule:: examples.basic_example
    |
* :download:`examples.thrown_object_example <../examples/thrown_object_example.py>`
    .. automodule:: examples.thrown_object_example
    |
* :download:`examples.utpredictor <../examples/utpredictor.py>`
    .. automodule:: examples.utpredictor
    |
* :download:`examples.benchmarking_example <../examples/benchmarking_example.py>`
    .. automodule:: examples.benchmarking_example
    |
* :download:`examples.eol_event <../examples/eol_event.py>`
    .. automodule:: examples.eol_event
    |
* :download:`examples.horizon <../examples/horizon.py>`
    .. automodule:: examples.horizon
    |
* :download:`examples.kalman_filter <../examples/kalman_filter.py>`
    .. automodule:: examples.kalman_filter
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
* :download:`examples.predict_specific_event <../examples/predict_specific_event.py>`
    .. automodule:: examples.predict_specific_event
    |
* :download:`examples.particle_filter_battery_example <../examples/particle_filter_battery_example.py>`
    .. automodule:: examples.particle_filter_battery_example
    |
* :download:`tutorial <../tutorial.ipynb>`
    |

Extending
---------
New State Estimators and Predictors are created by extending the :class:`prog_algs.state_estimators.StateEstimator` and :class:`prog_algs.predictors.Predictor` class, respectively. 

See :download:`examples.new_state_estimator_example <../examples/new_state_estimator_example.py>` for an example of this approach.
