# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable

from prog_algs.predictors.prediction import PredictionResults
from ..exceptions import ProgAlgTypeError
from ..uncertain_data import UncertainData


class Predictor(ABC):
    """
    Interface class for predictors

    Abstract base class for creating predictors that perform prediction. Predictor subclasses must implement this interface. Equivilant to "Observers" in NASA's Matlab Prognostics Algorithm Library

    Parameters
    ----------
    model : PrognosticsModel
        See: Prognostics Model Package\n
        A prognostics model to be used in prediction
    kwargs : optional, keyword arguments
    """
    default_parameters = {}

    def __init__(self, model, **kwargs):
        if not hasattr(model, 'output'):
            raise ProgAlgTypeError("model must have `output` method")
        if not hasattr(model, 'next_state'):
            raise ProgAlgTypeError("model must have `next_state` method")
        if not hasattr(model, 'inputs'):
            raise ProgAlgTypeError("model must have `inputs` property")
        if not hasattr(model, 'outputs'):
            raise ProgAlgTypeError("model must have `outputs` property")
        if not hasattr(model, 'states'):
            raise ProgAlgTypeError("model must have `states` property")
        if not hasattr(model, 'simulate_to_threshold'):
            raise ProgAlgTypeError("model must have `simulate_to_threshold` property")
        self.model = model

        self.parameters = deepcopy(self.default_parameters)
        self.parameters['events'] = self.model.events.copy()  # Events to predict to
        self.parameters.update(kwargs)

    @abstractmethod
    def predict(self, state : UncertainData, future_loading_eqn : Callable, **kwargs) -> PredictionResults:
        """
        Perform a single prediction

        Parameters
        ----------
        state : UncertainData 
            Distribution representing current state of the system
        future_loading_eqn : function (t, x) -> z
            Function to generate an estimate of loading at future time t, and state x
        options : optional, keyword arguments
            The following configuration parameters are supported: \n
            * dt (float): Simulation step size (s), e.g., 0.1
            * events (list[string]): Events to predict (subset of model.events) e.g., ['event1', 'event2']
            * horizon (float): Prediction horizon (s)
            * save_freq (float): Frequency at which results are saved (s)
            * save_pts (list[float]): Any additional savepoints (s) e.g., [10.1, 22.5]

        Return PredictionResults namedtuple
        ----------
        times : List[float]
            Times for each savepoint such that inputs.snapshot(i), states.snapshot(i), outputs.snapshot(i), and event_states.snapshot(i) are all at times[i]            
        inputs : Prediction
            Inputs at each savepoint such that inputs.snapshot(i) is the input distribution (type UncertainData) at times[i]
        states : Prediction
            States at each savepoint such that states.snapshot(i) is the state distribution (type UncertainData) at times[i]
        outputs : Prediction
            Outputs at each savepoint such that outputs.snapshot(i) is the output distribution (type UncertainData) at times[i]
        event_states : Prediction
            Event states at each savepoint such that event_states.snapshot(i) is the event state distribution (type UncertainData) at times[i]
        time_of_event : UncertainData
            Distribution of predicted Time of Event (ToE) for each predicted event, represented by some subclass of UncertaintData (e.g., MultivariateNormalDist)
        """
