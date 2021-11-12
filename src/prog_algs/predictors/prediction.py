# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from collections import UserList
from warnings import warn

from ..uncertain_data import UnweightedSamples


class Prediction():
    """
    Class for the result of a prediction. Is returned by the predict method of a predictor.

    Args:
        times (array(float)):
            Times for each data point where times[n] corresponds to data[n]
        data (array(UncertainData)):
            Data points for each time in times 
    """

    def __init__(self, times, data):
        self.times = times
        self.data = data

    def __eq__(self, other):
        """Compare 2 Predictions

        Args:
            other (Precition):

        Returns:
            bool: If the two Predictions are equal
        """
        return self.times == other.times and self.data == other.data

    def snapshot(self, time_index):
        """Get all samples from a specific timestep

        Args:
            index (int):
                Timestep (index number from times)

        Returns:
            UncertainData: Distribution for time corresponding to times[timestep]
        """
        return self.data[time_index]

    @property
    def mean(self):
        """Estimate of the mean value of the prediction at each time

        Returns:
            array(dict): 
                Mean value of the prediction at each time where mean[n] corresponds to the mean value of the prediction at time times[n].\n
                The mean value at each time is a dictionary. \n
                e.g., [{'state1': 1.2, 'state2': 1.3, ...}, ...]

        Example:
            mean_value = data.mean
        """
        return [dist.mean for dist in self.data]

    def time(self, index):
        warn("Deprecated. Please use prediction.times[index] instead.")
        return self.times[index]


class UnweightedSamplesPrediction(Prediction, UserList):
    """
    Immutable data class for the result of a prediction, where the predictions are stored as UnweightedSamples. Is returned from the predict method of a sample based prediction class (e.g., MonteCarlo). Objects of this class can be iterated and accessed like a list (e.g., prediction[0]), where prediction[n] represents a profile for sample n.

    Args:
        times : array(float)
            Times for each data point where times[n] corresponds to data[n]
        data : array(dict)
            Data points where data[n] is a SimResult for sample n
    """

    def __init__(self, times, data):
        super(UnweightedSamplesPrediction, self).__init__(times, data)
        self.__transformed = False  # If transform has been calculated

    def __calculate_tranform(self):
        """
        Calculate tranform of the data from data[sample_id][time_id] to data[time_id][sample_id]. Result is cached as self.__transform and is used in methods which look at a snapshot for a specific time
        """
        # Lazy calculation of tranform - only if needed
        # Note: prediction stops when event is reached, so for the length of all will not be the same. 
        # If the prediction doesn't go this far, then the value is set to None
        self.__transform = [UnweightedSamples([sample[time_index] if len(sample) > time_index else None for sample in self.data]) for time_index in range(len(self.times))]
        self.__transformed = True

    def __str__(self):
        return "UnweightedSamplesPrediction with {} savepoints".format(len(self.times))

    @property
    def mean(self):
        if not self.__transformed:
            self.__calculate_tranform()
        return [dist.mean for dist in self.__transform]

    def sample(self, sample_id):
        warn("Deprecated. Please use prediction[sample_id] instead.")
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
        This function is not implemented. Calling this will raise an error. Is is only included to make the class immutable.

        Raises:
            ValueError: 
        """
        raise ValueError("UnweightedSamplesPrediction is immutable (i.e., read only)")

    append = __not_implemented
    extend = __not_implemented
    clear = __not_implemented
    reverse = __not_implemented
    remove = __not_implemented
    insert = __not_implemented
    pop = __not_implemented
    __setitem__ = __not_implemented
    __setslice__ = __not_implemented
    __delitem__ = __not_implemented
