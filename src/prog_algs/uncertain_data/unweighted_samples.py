# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from . import UncertainData
from collections import UserList
from numpy import array, append, delete, cov, random
import warnings


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
        return UnweightedSamples(random.choice(self.data, num_samples))

    def keys(self):
        if len(self.data) == 0:
            return [[]]
        for sample in self:
            if sample is not None:
                return sample.keys()
        return []

    @property
    def mean(self):
        mean = {}
        for key in self.keys():
            mean[key] = array([x[key] for x in self.data if x is not None]).mean()
        return mean

    @property
    def cov(self):
        if len(self.data) == 0:
            return [[]]
        unlabeled_samples = array([[x[key] for x in self.data if x is not None] for key in self.keys()])
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
