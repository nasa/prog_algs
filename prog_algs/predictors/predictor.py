# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from abc import ABC, abstractmethod

class Predictor(ABC):
    """
    Interface class for predictors

    Abstract base class for creating predictors that perform prediction. Predictor subclasses must implement this interface. Equivilant to "Observers" in NASA's Matlab Prognostics Algorithm Library
    """
    @abstractmethod
    def predict(self, state_samples, future_loading_eqn, options):
        """
        Perform a single prediction

        Parameters
        ----------
        state_sampler : function (n) -> [x1, x2, ... xn]
            Function to generate n samples of the state. 
            e.g., def f(n): return [x1, x2, x3, ... xn]
        future_loading_eqn : function (t) -> z
            Function to generate an estimate of loading at future time t
        options : dict, optional
            Dictionary of any additional configuration values. See default parameters, above

        Return
        ______
        result : recorded values for all samples
        """
        pass
