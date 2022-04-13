# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from typing import Union
from numpy import array
from . import UncertainData, UnweightedSamples


class ScalarData(UncertainData):
    """
    Data without uncertainty- single value

    Args:
            state (dict): Single state
    """
    def __init__(self, state): 
        self.__state = state

    def __eq__(self, other : "ScalarData") -> bool:
        return isinstance(other, ScalarData) and other.mean == self.__state

    def __add__(self, other : int) -> "UncertainData":
        new_state = dict()
        for k,v in self.__state.items():
            new_state[k] = v + other
        return ScalarData(new_state)

    def __radd__(self, other : int) -> "UncertainData":
        if other == 0:
            return self
        return self.__add__(other)

    def __iadd__(self, other : int) -> "UncertainData":
        for k in self.__state.keys():
            self.__state[k] += other
        return self

    def __sub__(self, other : int) -> "UncertainData":
        new_state = dict()
        for k,v in self.__state.items():
            new_state[k] = v - other
        return ScalarData(new_state)

    def __rsub__(self, other : int) -> "UncertainData":
        if other == 0:
            return self
        return self.__sub__(other)

    def __isub__(self, other : int) -> "UncertainData":
        for k in self.__state.keys():
            self.__state[k] -= other
        return self

    @property
    def median(self) -> array:
        return self.mean
        
    @property
    def mean(self) -> array:
        return self.__state

    @property
    def cov(self) -> array:
        return [[0]]

    def keys(self):
        return self.__state.keys()
        
    def sample(self, num_samples : int = 1) -> UnweightedSamples:
        return UnweightedSamples([self.__state] * num_samples)

    def __str__(self) -> str:
        return 'ScalarData({})'.format(self.__state)

    def percentage_in_bounds(self, bounds : Union[list, dict]) -> dict:
        if isinstance(bounds, list):
            bounds = {key: bounds for key in self.keys()}
        if not isinstance(bounds, dict) and all([isinstance(b, list) for b in bounds]):
            raise TypeError("Bounds must be list [lower, upper] or dict (key: [lower, upper]), was {}".format(type(bounds)))
        return {key: (1 if bounds[key][0] < x and bounds[key][1] > x else 0) for (key, x) in self.__state.items()}
