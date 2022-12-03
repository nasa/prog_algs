# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy

from ..uncertain_data import UncertainData
from ..exceptions import ProgAlgTypeError


class StateEstimator(ABC):
    """
    Interface class for state estimators

    Abstract base class for creating state estimators that perform state estimation. Subclasses must implement this interface. Equivilant to "Observers" in NASA's Matlab Prognostics Algorithm Library

    Args:
        model (PrognosticsModel):
            A prognostics model to be used in state estimation
            See: Prognostics Model Package
        x0 (UncertainData, model.StateContainer, or dict):
            Initial (starting) state, with keys defined by model.states \n
            e.g., x = ScalarData({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
    
    See state-estimator specific documentation for speicfic keyword arguments.
    """

    default_parameters = {
        't0': 0
    }

    def __init__(self, model, x0, **kwargs):
        # Check model
        if not hasattr(model, 'output'):
            raise ProgAlgTypeError("model must have `output` method")
        if not hasattr(model, 'next_state'):
            raise ProgAlgTypeError("model must have `next_state` method")
        if not hasattr(model, 'outputs'):
            raise ProgAlgTypeError("model must have `outputs` property")
        if not hasattr(model, 'states'):
            raise ProgAlgTypeError("model must have `states` property")
        self.model = model

        # Check x0
        for key in model.states:
            if key not in x0:
                raise ProgAlgTypeError("x0 missing state `{}`".format(key))
        
        # Process kwargs (configuration)
        self.parameters = deepcopy(self.default_parameters)
        self.parameters.update(kwargs)

        self.t = self.parameters['t0']  # Initial Time

    @abstractmethod
    def estimate(self, t : float, u, z, **kwargs) -> None:
        """
        Perform one state estimation step (i.e., update the state estimate, filt.x)

        Args
        ----------
        t : float
            Current timestamp in seconds (≥ 0.0)
            e.g., t = 3.4
        u : InputContainer
            Measured inputs, with keys defined by model.inputs.
            e.g., u = m.InputContainer({'i':3.2}) given inputs = ['i']
        z : OutputContainer
            Measured outputs, with keys defined by model.outputs.
            e.g., z = m.OutputContainer({'t':12.4, 'v':3.3}) given outputs = ['t', 'v']

        Note
        ----
        This method updates the state estimate stored in filt.x, but doesn't return the updated estimate. Call filt.x to get the updated estimate.
        """

    def monotonicity(self) -> dict:
        """Calculate monotonicty for a state estimate. 
        Given a state estimate, for each state: go through all events and compare those to the next one.
        Calculates monotonicity for each event key using its associated value.
        
        monotonoicity = |Σsign(i+1 - i) / N-1|
        Where N is number of measurements and sign indicates sign of calculation.
        Coble, J., et. al. (2021). Identifying Optimal Prognostic Parameters from Data: A Genetic Algorithms Approach. Annual Conference of the PHM Society.
        http://www.papers.phmsociety.org/index.php/phmconf/article/view/1404
        Baptistia, M., et. al. (2022). Relation between prognostics predictor evaluation metrics and local interpretability SHAP values. Aritifical Intelligence, Volume 306.
        https://www.sciencedirect.com/science/article/pii/S0004370222000078
        Args:
            None
        Returns:
            dict (str, float): Dictionary where keys represent an event and float is its respective monotonicitiy value between [0, 1].
        """
        pass

    @property
    @abstractproperty
    def x(self) -> UncertainData:
        """
        The current estimated state. 

        Example
        -------
        state = filt.x
        """
