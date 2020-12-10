from abc import ABC, abstractmethod, abstractproperty
from numpy.random import choice, multivariate_normal
from numpy import array, append, delete


class UncertainData(ABC):
    @abstractmethod
    def sample(self, nSamples = 1):
        pass

    @property
    @abstractproperty
    def mean(self):
        pass

    # TODO(CT): Consider median

class UnweightedSamples(UncertainData):
    def __init__(self, samples : array = array([])):
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

    # Sample-specific methods
    def append(self, value):
        self.__samples = append(self.__samples, value)

    @property
    def size(self):
        return self.__samples.size

    def __getitem__(self, index):
        return self.__samples[index]

    def __setitem__(self, index, value):
        self.__samples[index] = value

    def __delitem__(self, index):
         self.__samples = delete(self.__samples, index)

    def raw_samples(self):
        return self.__samples

class MultivariateNormalDist(UncertainData):
    def __init__(self, labels, mean: array, covar : array):
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

    # Dist-specific methods
    @property
    def covar(self):
        return self.__covar
    