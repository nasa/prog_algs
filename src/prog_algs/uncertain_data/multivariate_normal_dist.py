# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from . import UncertainData, UnweightedSamples
from numpy import array
from numpy.random import multivariate_normal


class MultivariateNormalDist(UncertainData):
    """
    Data represented by a multivariate normal distribution with mean and covariance matrix

    Args:
            labels (list[str]): Labels for states, in order of mean values
            mean (array[float]): Mean values for state in the same order as labels
            covar (array[array[float]]): Covariance matrix for state
    """
    def __init__(self, labels, mean: array, covar : array):
        self.__labels = labels
        self.__mean = mean
        self.__covar = covar

    def sample(self, num_samples = 1):
        if len(self.__mean) != len(self.__labels):
            raise Exception("labels must be provided for each value")
    
        samples = multivariate_normal(self.__mean, self.__covar, num_samples)
        samples = [{key: value for (key, value) in zip(self.__labels, x)} for x in samples]
        return UnweightedSamples(samples)

    def keys(self):
        return self.__labels

    @property
    def median(self):
        # For normal distribution medain = mean
        return self.mean

    @property
    def mean(self):
        return {key: value for (key, value) in zip(self.__labels, self.__mean)}

    def __str__(self):
        return 'MultivariateNormalDist(mean: {}, covar: {})'.format(self.__mean, self.__covar)     

    @property
    def cov(self):
        return self.__covar
