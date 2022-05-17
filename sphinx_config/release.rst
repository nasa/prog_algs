Release Notes
=======================

..  contents:: 
    :backlinks: top

Updates in V1.3
-----------------------
* **New State Estimator Added** :class:`prog_algs.state_estimators.KalmanFilter`. Works with models derived from :class:`prog_models.LinearModel`. See :download:`examples.kalman_filter <../examples/kalman_filter.py>`
* **New Predictor Added** :class:`prog_algs.predictors.UnscentedTransformPredictor`. See :download:`examples.utpredictor <../examples/utpredictor.py>`
* Initial state estimate (x0) can now be passed as `UncertainData` to represent initial state uncertainty. See :download:`examples.playback <../examples/playback.py>`
* Added new metrics for :class:`prog_algs.predictors.ToEPredictionProfile`: Prognostics horizon, Cumulative Relative Accuracy (CRA). See :download:`examples.playback <../examples/playback.py>`
* Added ability to plot :class:`prog_algs.predictors.ToEPredictionProfile`: profile.plot(). See :download:`examples.playback <../examples/playback.py>`
* Added new metric for :class:`prog_algs.predictors.Prediction`: Monotonicity, Relative Accuracy (RA)
* Added new metric for :class:`prog_algs.uncertain_data.UncertainData` (and subclasses): Root Mean Square Error (RMSE)
* Added new describe method for :class:`prog_algs.uncertain_data.UncertainData` (and subclasses)
* Add support for python 3.10
* Various performance improvements and bugfixes


Updates in v1.2
---------------

Note for Existing Users
***********************
This release includes changes to the return format of the MonteCarlo Predictor's `predict` method. These changes were necessary to support non-sample based predictors. The non backwards-compatible changes are listed below:
* times: 
    * previous ```List[List[float]]``` where times[n][m] corresponds to timepoint m of sample n. 
    * new ```List[float]``` where times[m] corresponds to timepoint m for all samples.
* End of Life (EOL)/ Time of Event (ToE) estimates:
    * previous ```List[float]``` where the times correspond to the time that the first event occurs.
    * new ```UnweightedSamples``` where keys correspond to the inidividualevents predicted.
* State at time of event (ToE).
   * previous: element in states.
   * new: member of ToE structure (e.g., ToE.final_state['event1']).

General Updates
***************
* New Feature: Histogram and Scatter Plot of UncertainData.
* New Feature: Vectorized particle filter.
    * Particle Filter State Estimator is now vectorized for vectorized models - this significantly improves performance.
* New Feature: Unscented Transform Predictor.
    * New predictor that propogates sigma points forward to estimate time of event and future states.
* New Feature: `Prediction` class to represent predicted future values.
* New Feature: `ToEPredictionProfile` class to represent and operate on the result of multiple predictions generated at different prediction times.
* Added metrics `percentage_in_bounds` and `metrics` and plots to UncertainData .
* Add support for Python3.9.
* General Bugfixes.
