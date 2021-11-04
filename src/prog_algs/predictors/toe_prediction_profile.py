# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.
from collections import UserDict

from prog_algs.uncertain_data import UncertainData 

class ToEPredictionProfile(UserDict):
    """
    """
    def add_prediction(self, time_of_prediction: float, toe_prediction: UncertainData):
        """Add a single prediction to the profile

        Args:
            time_of_prediction (float): Time that the prediction was made
            toe_prediction (UncertainData): Distribution of predicted ToEs
        """
        self[time_of_prediction] = toe_prediction

    def __iter__(self):
        return iter(sorted(super(ToEPredictionProfile, self).__iter__()))

    def items(self):
        return iter((k, self[k]) for k in self)

    def keys(self):
        return sorted(super(ToEPredictionProfile, self).keys())

    def values(self):
        return [self[k] for k in self.keys()]
