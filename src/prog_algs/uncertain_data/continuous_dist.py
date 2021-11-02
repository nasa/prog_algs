# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from . import UncertainData, UnweightedSamples
from scipy.stats import gmean


class ContinuousDist(UncertainData):
    """
    Data represented by a continuous distribution
    """
    def __init__(self, labels, dist):
        """Initialize distribution

        Args:
            labels ([str]): Labels for states, in order of mean values
            distribution (scipy.stats distribution): Distribution representing the data
        """
        self.__labels = labels
        self.__dist = dist

    def sample(self, num_samples = 1):
        if len(self.__dist.mean) != len(self.__labels):
            raise Exception("labels must be provided for each value")
    
        samples = self.__dist.rvs(num_samples)
        samples = [{key: value for (key, value) in zip(self.__labels, x)} for x in samples]
        return UnweightedSamples(samples)

    def keys(self):
        return self.__labels

    @property
    def median(self):
        return self.__dist.median

    @property
    def mean(self):
        return {key: value for (key, value) in zip(self.__labels, self.__dist.mean)}

    def __str__(self):
        return 'Continuous Dist (keys: )'.format(self.__labels)     

    @property
    def cov(self):
        return self.sample(1000).cov

    def pdf(self, x):
        return self._dist.pdf(x)

    def cdf(self, x):
        return self._dist.cdf(x)

    def logpdf(self, x):
        return self._dist.logpdf(x)

    def logcdf(self, x):
        return self._dist.logcdf(x)

    def sf(self, x):
        return self._dist.sf(x)

    def logsf(self, x):
        return self._dist.logsf(x)
