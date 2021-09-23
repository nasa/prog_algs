from collections import UserList
from ..uncertain_data import UnweightedSamples


class Prediction(UserList):
    """
    Result of a prediction
    """
    __slots__ = ['times', 'data']  # Optimization 

    def __init__(self, times, data):
        """
        Args:
            times (array(float)): Times for each data point where times[n] corresponds to data[n]
            data (array(dict)): Data points where data[n] corresponds to times[n]
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

    def sample(self, index):
        """Get sample by sample_id, equivalent to prediction[index]

        Args:
            index (int): index of sample

        Returns:
            SimResult: Values for that sample at different times where result[i] corresponds to time[i]
        """
        return self[index]

    def snapshot(self, index):
        """Get all samples from a specific timestep

        Args:
            index (int): Timestep (index number from times)

        Returns:
            UnweightedSamples: Samples for time corresponding to times[timestep]
        """
        return UnweightedSamples([sample[index] for sample in self.data])

    def time(self, index):
        """Get time for data point at index `index`

        Args:
            index (int)

        Returns:
            float: Time for which the data point at index `index` corresponds
        """
        return self.times[index]
