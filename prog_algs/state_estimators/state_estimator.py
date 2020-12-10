# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from abc import ABC, abstractmethod, abstractproperty
from ..uncertain_data import UncertainData

class StateEstimator(ABC):
    """
    Interface class for state estimators

    Abstract base class for creating state estimators that perform state estimation. Subclasses must implement this interface. Equivilant to "Observers" in NASA's Matlab Prognostics Algorithm Library
    """

    @abstractmethod
    def estimate(self, t, u, z):
        """
        Perform one state estimation step (i.e., update the state estimate)

        Parameters
        ----------
        t : double
            Current timestamp in seconds (≥ 0.0)
            e.g., t = 3.4
        u : dict
            Measured inputs, with keys defined by model.inputs.
            e.g., u = {'i':3.2} given inputs = ['i']
        z : dict
            Measured outputs, with keys defined by model.outputs.
            e.g., z = {'t':12.4, 'v':3.3} given inputs = ['t', 'v']
        """
        pass

    @abstractproperty
    def x(self) -> UncertainData:
        """
        Getter for property 'x', the current estimated state. 

        Example
        -------
        state = observer.x
        """
        pass
