# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from abc import ABC, abstractmethod, abstractproperty
from numpy.random import choice, multivariate_normal
from numpy import array, append, delete, cov
from prog_algs.visualize import plot_scatter
from collections import UserList
import warnings

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

    @abstractmethod
    def keys(self):
        """Get the keys for the property represented

        Returns:
            [string]: keys
        """
    # TODO(CT): Consider median

    def plot_scatter(self, fig = None, keys = None, num_samples = 100, **kwargs):
        """
        Produce a scatter plot

        Args:
            fig (Figure, optional): Existing figure previously used to plot states. If passed a figure argument additional data will be added to the plot. Defaults to creating new figure
            keys (list of strings, optional): Keys to plot. Defaults to all keys.
            num_samples (int, optional): Number of samples to plot. Defaults to 100
            **kwargs (optional): Additional keyword arguments passed to scatter function.

        Returns:
            Figure

        Example:
            states = UnweightedSamples([1, 2, 3, 4, 5])
            states.plot_scatter() # With 100 samples
            states.plot_scatter(num_samples=5) # Specifying the number of samples to plot
            states.plot_scatter(keys=['state1', 'state2']) # only plot those keys
        """
        samples = self.sample(num_samples)
        return plot_scatter(samples, fig=fig, keys=keys, **kwargs)


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

    def __eq__(self, other):
        return isinstance(other, ScalarData) and other.mean() == self.__state

    @property
    def mean(self):
        return self.__state

    @property
    def cov(self):
        return [[0]]

    def keys(self):
        return self.__state.keys()
        
    def sample(self, num_samples = 1):
        return array([self.__state] * num_samples)

    def __str__(self):
        return 'ScalarData({})'.format(self.__state)


class UnweightedSamples(UncertainData, UserList):
    """
    Uncertain Data represented by a set of samples
    """
    def __init__(self, samples = []):
        """Initialize Unweighted Samples

        Args:
            samples (array, optional): array of samples. Defaults to empty array.
        """
        self.data = samples

    def sample(self, num_samples = 1):
        # Completely random resample
        return choice(self.data, num_samples)

    def keys(self):
        if len(self.__samples) == 0:
            return [[]]
        return self[0].keys()

    @property
    def mean(self):
        mean = {}
        for key in self.data[0].keys():
            mean[key] = array([x[key] for x in self.data]).mean()
        return mean

    @property
    def cov(self):
        if len(self.data) == 0:
            return [[]]
        unlabeled_samples = array([[x[key] for x in self.data] for key in self.data[0].keys()])
        return cov(unlabeled_samples)

    def __str__(self):
        return 'UnweightedSamples({})'.format(self.data)

    @property
    def size(self):
        """Get the number of samples. Note: kept for backwards compatibility, prefer using len() instead.

        Returns:
            int: Number of samples
        """
        return len(self)

    def raw_samples(self):
        warnings.warn("raw_samples is deprecated and will be removed in the future")
        return self.data


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

    def keys(self):
        return self.__labels

    @property
    def mean(self):
        return {key: value for (key, value) in zip(self.__labels, self.__mean)}

    def __str__(self):
        return 'MultivariateNormalDist(mean: {}, covar: {})'.format(self.__mean, self.__covar)     

    @property
    def cov(self):
        return self.__covar
    