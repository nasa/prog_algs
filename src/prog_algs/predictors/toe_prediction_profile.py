# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.
from collections import UserDict
from typing import Dict

from prog_algs.uncertain_data import UncertainData 

class ToEPredictionProfile(UserDict):
    """
    Data structure for storing the result of multiple predictions, including time of prediction. This data structure can be treated as a dictionary of time of prediction to Time of Event (ToE) prediction. Iteration of this data structure is in order of increasing time of prediction
    """
    def add_prediction(self, time_of_prediction: float, toe_prediction: UncertainData):
        """Add a single prediction to the profile

        Args:
            time_of_prediction (float): Time that the prediction was made
            toe_prediction (UncertainData): Distribution of predicted ToEs
        """
        self[time_of_prediction] = toe_prediction

    # Functions below are defined to ensure that any iteration is in order of increasing time of prediction
    def __iter__(self):
        return iter(sorted(super(ToEPredictionProfile, self).__iter__()))

    def items(self):
        """
        Get iterators for the items (time_of_prediction, toe_prediction) of the prediction profile
        """
        return iter((k, self[k]) for k in self)

    def keys(self):
        """
        Get iterator for the keys (i.e., time_of_prediction) of the prediction profile
        """
        return sorted(super(ToEPredictionProfile, self).keys())

    def values(self):
        """
        Get iterator for the values (i.e., toe_prediction) of the prediction profile
        """
        return [self[k] for k in self.keys()]

    def alpha_lambda(self, ground_truth : Dict[str, float], lambda_value : float, alpha : float, beta : float, **kwargs) -> Dict[str, bool]:
        """Calculate Alpha lambda metric for the prediction profile

        Args:
            ground_truth (Dict[str, float]):
                Ground Truth time of event for each event (e.g., {'event1': 748, 'event2', 2233, ...})
            lambda_value (float):
                Prediction time at or after which metric is evaluated. Evaluation occurs at this time (if a prediction exists) or the next prediction following.
            alpha (float): 
                percentage bounds around time to event (where 0.2 allows 20% error TtE)
            beta (float):
                portion of prediction that must be within those bounds
            kwargs (optional, keyword arguments):
                configuration arguments. Accepted arge include: \n
                 * keys (list[string]): list of keys to use. If not provided, all keys are used.
                 * print (bool) : If True, print the results. Default is False.

        Returns:
            Dict[str, bool]: If alpha lambda was met for each key (e.g., {'event1': True, 'event2', False, ...})
        """
        from ..metrics import alpha_lambda
        return alpha_lambda(self, ground_truth, lambda_value, alpha, beta, **kwargs)

    def prognostic_horizon(self, criteria_eqn, ground_truth, **kwargs) -> Dict[str, float]:
        """
        Compute prognostic horizon metric, defined as the difference between a time ti, when the predictions meet specified performance criteria, and the time corresponding to the true Time of Event (ToE), for each event.
        PH = ToE - ti
        Args:
            toe_profile (ToEPredictionProfile): A profile of predictions, the combination of multiple predictions
            criteria_eqn (Callable function): A function (toe: UncertainData, ground_truth: dict[str, float]) -> dict[str, bool] calculating whether a prediction in ToEPredictionProfile meets some criteria. \n
                | Args: 
                |  * toe (UncertainData): A single prediction of Time of Event (ToE)
                |  * ground truth (dict[str, float]): Ground truth passed into prognostics_horizon
                | Returns: Map of event names to boolean representing if the event has been met. 
                |   e.g., {'event1': True, 'event2': False}
            ground_truth (dict): Dictionary containing ground truth; specified as key, value pairs for event and its value. E.g, {'event1': 47.3, 'event2': 52.1, 'event3': 46.1}
            kwargs (optional): configuration arguments. Accepted args include:
                * print (bool): Boolean specifying whether the prognostic horizon metric should be printed.

        Returns:
            dict: Dictionary containing prognostic horizon calculations (value) for each event (key). e.g., {'event1': 12.3, 'event2': 15.1}
        """
        from ..metrics import prognostic_horizon
        return prognostic_horizon(self, criteria_eqn, ground_truth, **kwargs)

    def cumulative_relative_accuracy(self, ground_truth, **kwargs) -> Dict[str, float]:
        """
        Compute cumulative relative accuracy for a given profile, defined as the normalized sum of relative prediction accuracies at specific time instances.
        
        CRA = Σ(RA)/N for each event
        Where Σ is summation of all relative accuracies for a given profile and N is the total count of profiles (Journal Prognostics Health Management, Saxena et al.)
        Args:
            ground_truth (dict): Dictionary containing ground truth; specified as key, value pairs for event and its value. E.g, {'event1': 47.3, 'event2': 52.1, 'event3': 46.1}
            kwargs (optional): configuration arguments. Accepted args include:

        Returns:
            dict: Dictionary containing cumulative relative accuracy (value) for each event (key). e.g., {'event1': 12.3, 'event2': 15.1}
        """
        from ..metrics import cumulative_relative_accuracy
        return cumulative_relative_accuracy(self, ground_truth, **kwargs)
