# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from abc import ABC, abstractmethod, abstractproperty
from collections import UserList
from warnings import warn

from ..uncertain_data import UnweightedSamples, MultivariateNormalDist


class Prediction(ABC):
    """
    Parent class for the result of a prediction. Is returned by the predict method of a predictor. Defines the interface for operations on a prediciton data object. 

    Note: This class is not intended to be instantiated directly, instead subclasses should be used
    """
    def __init__(self, times, data):
        """
        Args:
            times (array(float)): Times for each data point where times[n] corresponds to data[n]
            data
        """
        self.times = times
        self.data = data

    def __eq__(self, other):
        """Compare 2 Predictions

        Args:
            other (Precition)

        Returns:
            bool: If the two Predictions are equal
        """
        return self.times == other.times and self.data == other.data

    @abstractmethod
    def snapshot(self, time_index):
        """Get all samples from a specific timestep

        Args:
            index (int): Timestep (index number from times)

        Returns:
            UnweightedSamples: Samples for time corresponding to times[timestep]
        """
        pass

    @property
    @abstractproperty
    def mean(self):
        """Mean estimate

        Example:
            mean_value = data.mean
        """
        pass

    def time(self, index):
        """Get time for data point at index `index`

        Args:
            index (int)

        Returns:
            float: Time for which the data point at index `index` corresponds
        """
        warn("Depreciated. Please use prediction.times[index] instead.")
        return self.times[index]


class UnweightedSamplesPrediction(Prediction, UserList):
    """
    Data class for the result of a prediction, where the predictions are stored as UnweightedSamples. Is returned from the predict method of a sample based prediction class (e.g., MonteCarlo).
    """
    def __init__(self, times, data):
        """
        Initialize UnweightedSamplesPrediction

        Args:
            times (array(float)): Times for each data point where times[n] corresponds to data[n]
            data (array(dict)): Data points where data[n] corresponds to times[n]
        """
        super(UnweightedSamplesPrediction, self).__init__(times, data)
        self.__transformed = False  # If transform has been calculated

    def __calculate_tranform(self):
        """
        Calculate tranform of the data from data[sample_id][time_id] to data[time_id][sample_id]. Result is cached as self.__transform and is used in methods which look at a snapshot for a specific time
        """
        # Lazy calculation of tranform - only if needed
        self.__transform = [UnweightedSamples([sample[time_index] for sample in self.data]) for time_index in range(len(self.times))]
        self.__transformed = True

    def __str__(self):
        return "UnweightedSamplesPrediction with {} savepoints".format(len(self.times))

    @property
    def mean(self):
        if not self.__transformed:
            self.__calculate_tranform()
        return [dist.mean for dist in self.__transform]

    def sample(self, sample_id):
        """Get sample by sample_id, equivalent to prediction[index]. Depreciated in favor of prediction[id]

        Args:
            index (int): index of sample

        Returns:
            SimResult: Values for that sample at different times where result[i] corresponds to time[i]
        """
        warn("Depreciated. Please use prediction[sample_id] instead.")
        return self[sample_id]

    def snapshot(self, time_index):
        """Get all samples from a specific timestep

        Args:
            index (int): Timestep (index number from times)

        Returns:
            UnweightedSamples: Samples for time corresponding to times[timestep]
        """
        if not self.__transformed:
            self.__calculate_tranform()
        return self.__transform[time_index]

    def __not_implemented(self, *args, **kw):
        """
        Called for not implemented functions. These functions are not used to make the class immutable
        """
        raise ValueError("UnweightedSamplesPrediction is immutable (i.e., read only)")

    append = pop = __setitem__ = __setslice__ = __delitem__ = __not_implemented


class MultivariateNormalDistPrediction(Prediction):
    """
    Data class for the result of a prediction, where the predictions are stored as MultivariateNormalDist. Is returned from the predict method of a MultivariateNormalDist-based prediction class (e.g., Unscented Kalman Predictor).
    """
    def __str__(self):
        return "MultivariateNormalDistPrediction with {} savepoints".format(len(self.times))

    @property
    def mean(self):
        return [dist.mean for dist in self.data]

    def snapshot(self, time_index):
        """Get all samples from a specific timestep

        Args:
            index (int): Timestep (index number from times)

        Returns:
            UnweightedSamples: Samples for time corresponding to times[timestep]
        """
        return self.data[time_index]
