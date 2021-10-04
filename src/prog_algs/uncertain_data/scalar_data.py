# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from . import UncertainData, UnweightedSamples
from numpy import array


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
        return UnweightedSamples([self.__state] * num_samples)

    def __str__(self):
        return 'ScalarData({})'.format(self.__state)
