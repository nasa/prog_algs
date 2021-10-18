# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from abc import ABC, abstractmethod
from copy import deepcopy
from ..exceptions import ProgAlgTypeError

class Predictor(ABC):
    """
    Interface class for predictors

    Abstract base class for creating predictors that perform prediction. Predictor subclasses must implement this interface. Equivilant to "Observers" in NASA's Matlab Prognostics Algorithm Library

    Parameters
    ----------
    * model : prog_models.prognostics_model.PrognosticsModel\n
        See: Prognostics Model Package\n
        A prognostics model to be used in prediction
    * options (optional, kwargs): configuration options\n
        Any additional configuration values. See documentation for specific predictor \n
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
        self.parameters.update(kwargs)

    @abstractmethod
    def predict(self, state_samples, future_loading_eqn, **kwargs):
        """
        Perform a single prediction

        Parameters
        ----------
        state_sampler : function (n) -> [x1, x2, ... xn]
            Function to generate n samples of the state. 
            e.g., def f(n): return [x1, x2, x3, ... xn]
        future_loading_eqn : function (t) -> z
            Function to generate an estimate of loading at future time t
        options : keyword arguments, optional
            Any additional configuration values. See default parameters, above

        Return
        ______
        result : recorded values for all samples
        """
        pass
