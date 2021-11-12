# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

"""
This file includes functions for calculating metrics given a time of event (ToE) profile (i.e., ToE's calculated at different times of prediction resulting from running prognostics multiple times, e.g., on playback data). The metrics calculated here are specific to multiple ToE estimates (e.g. alpha-lambda metric)
"""
from ..predictors import ToEPredictionProfile

def alpha_lambda(toe_profile : ToEPredictionProfile, ground_truth : dict, lambda_value : float, alpha : float, beta : float, **kwargs): 
    """
    Compute alpha lambda metric, a common metric in prognostics. Alpha-Lambda is met if alpha % of the Time to Event (TtE) distribution is within beta % of the ground truth at prediction time lambda.

    Args:
        toe_profile (ToEPredictionProfile): A profile of predictions, the combination of multiple predictions
        ground_truth (dict): Ground Truth time of event for each event (e.g., {'event1': 748, 'event2', 2233, ...})
        lambda_value (float): Prediction time at or after which metric is evaluated. Evaluation occurs at this time (if a prediction exists) or the next prediction following.
        alpha (float): percentage bounds around time to event (where 0.2 allows 20% error TtE)
        beta (float): portion of prediction that must be within those bounds
        kwargs (optional): configuration arguments. Accepted arge include:
            * keys (list[string], optional): list of keys to use. If not provided, all keys are used.

    Returns:
        bool: if alpha-lambda met
    """
    params = {
        'print': False
    }
    params.update(kwargs)

    for (t_prediction, toe) in toe_profile.items():
        if (t_prediction >= lambda_value):
            # If keys not provided, use all
            keys = params.setdefault('keys', toe.keys())

            result = {}
            for key in keys:
                upper_bound = ground_truth[key] + alpha*(ground_truth[key]-t_prediction)
                lower_bound = ground_truth[key] - alpha*(ground_truth[key]-t_prediction)
                result[key] = toe.percentage_in_bounds([lower_bound, upper_bound])[key] >= beta 
                if params['print']:
                    print('\n', key)
                    print('\ttoe:', toe.key(key))
                    print('\tBounds: [{} - {}]({}%)'.format(lower_bound, upper_bound, toe.percentage_in_bounds([lower_bound, upper_bound])[key]))
            return result
