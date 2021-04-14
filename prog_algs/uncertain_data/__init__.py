# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from abc import ABC, abstractmethod, abstractproperty
from numpy.random import choice, multivariate_normal
from numpy import array, append, delete, cov

class UncertainData(ABC):
    """
    Abstract base class for data with uncertainty. Any new uncertainty type must implement this class
    """
    @abstractmethod
    def sample(self, nSamples = 1):
        """Generate samples from data

        Args:
            nSamples (int, optional): Number of samples to generate. Defaults to 1.

        Returns:
            samples (array): Array of nSamples samples

        Example:
            samples = data.samples(100)
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

    @property
    @abstractproperty
    def cov(self):
        """Get the covariance matrix

        Returns:
            [[float]]: covariance matrix
        """

    # TODO(CT): Consider median

class ScalarData(UncertainData):
    """
    Data without uncertainty- single value
    """
    def __init__(self, state): 
        """Initialize Scalar Data

        Args:
            state (dict): Single state
        """
        self.__state = state

    @property
    def mean(self):
        return self.__state

    @property
    def cov(self):
        return [[0 for i in self.__state] for j in self.__state]

    def sample(self, num_samples = 1):
        return array([self.__state] * num_samples)

    def __str__(self):
        return 'ScalarData({})'.format(self.__state)


class UnweightedSamples(UncertainData):
    """
    Uncertain Data represented by a set of samples
    """
    def __init__(self, samples : array = array([])):
        """Initialize Unweighted Samples

        Args:
            samples (array, optional): array of samples. Defaults to empty array.
        """
        self.__samples = samples

    def sample(self, num_samples = 1):
        # Completely random resample
        return choice(self.__samples, num_samples)

    @property
    def mean(self):
        mean = {}
        for key in self.__samples[0].keys():
            mean[key] = array([x[key] for x in self.__samples]).mean()
        return mean

    @property
    def cov(self):
        if len(self.__samples) == 0:
            return [[]]
        unlabeled_samples = array([[x[key] for x in self.__samples] for key in self.__samples[0].keys()])
        return cov(unlabeled_samples)

    def __str__(self):
        return 'UnweightedSamples({})'.format(self.__samples)

    # Sample-specific methods
    def append(self, value):
        """Append an additional sample to the unweighted samples

        Args:
            value (dict): Value to be appended
        """
        self.__samples = append(self.__samples, value)

    @property
    def size(self):
        """Get the number of samples

        Returns:
            int: Number of samples
        """
        return self.__samples.size

    def __getitem__(self, index):
        """Get a specific item by index

        Args:
            index (int): Sample index requested

        Returns:
            dict: item requested

        Example:
            sample = samples[index]
        """
        return self.__samples[index]

    def __setitem__(self, index, value):
        """Set a specific item by index

        Args:
            index (int): index to be set
            value (dict): new value

        Example:
            samples[index] = new_value
        """
        self.__samples[index] = value

    def __delitem__(self, index):
        """Delete a specific utem

        Args:
            index (int): Index of item to be deleted

        Example:
            del samples[index]
        """
        self.__samples = delete(self.__samples, index)

    def raw_samples(self):
        """Get raw samples

        Returns:
            np.array(dict): all the samples
        """
        return self.__samples

class MultivariateNormalDist(UncertainData):
    """
    Data represented by a multivariate normal distribution with mean and covariance matrix
    """
    def __init__(self, labels, mean: array, covar : array):
        """Initialize distribution

        Args:
            labels ([str]): Labels for states, in order of mean values
            mean ([float]): Mean values for state in the same order as labels
            covar ([[float]]): Covariance matrix for state
        """
        self.__labels = labels
        self.__mean = mean
        self.__covar = covar

    def sample(self, num_samples = 1):
        if len(self.__mean) != len(self.__labels):
            raise Exception("labels must be provided for each value")
    
        samples = multivariate_normal(self.__mean, self.__covar, num_samples)
        samples = array([{key: value for (key, value) in zip(self.__labels, x)} for x in samples])
        return samples

    @property
    def mean(self):
        return {key: value for (key, value) in zip(self.__labels, self.__mean)}

    def __str__(self):
        return 'MultivariateNormalDist(mean: {}, covar: {})'.format(self.__mean, self.__covar)     

    @property
    def cov(self):
        return self.__covar
    