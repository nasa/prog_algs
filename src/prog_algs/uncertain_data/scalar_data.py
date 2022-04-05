# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from . import UncertainData, UnweightedSamples


class ScalarData(UncertainData):
    """
    Data without uncertainty- single value

    Args:
            state (dict): Single state
    """
    def __init__(self, state): 
        self.__state = state

    def __eq__(self, other):
        return isinstance(other, ScalarData) and other.mean == self.__state

    @property
    def median(self):
        return self.mean
        
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

    def percentage_in_bounds(self, bounds):
        if isinstance(bounds, list):
            bounds = {key: bounds for key in self.keys()}
        if not isinstance(bounds, dict) and all([isinstance(b, list) for b in bounds]):
            raise TypeError("Bounds must be list [lower, upper] or dict (key: [lower, upper]), was {}".format(type(bounds)))
        return {key: (1 if bounds[key][0] < x and bounds[key][1] > x else 0) for (key, x) in self.__state.items()}
