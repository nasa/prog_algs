from abc import ABC, abstractmethod

class Predictor(ABC):
    """
    Interface class for predictors

    Abstract base class for creating predictors that perform prediction. Predictor subclasses must implement this interface. Equivilant to "Observers" in NASA's Matlab Prognostics Algorithm Library
    """
    @abstractmethod
    def predict(self):
        """
        Perform a single prediction
        """
        pass
