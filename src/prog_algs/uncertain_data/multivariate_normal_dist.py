# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from . import ContinuousDist
from numpy import array
from scipy.stats import multivariate_normal


class MultivariateNormalDist(ContinuousDist):
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
        super().__init__(labels, multivariate_normal(mean, covar))
        self.__mean = mean
        self.__covar = covar


    def __str__(self):
        return 'MultivariateNormalDist(mean: {}, covar: {})'.format(self.__mean, self.__covar)  

    @property
    def mean(self):
        return self.__mean   

    @property
    def cov(self):
        return self.__covar
