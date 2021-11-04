# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

"""
This file includes functions for calculating metrics given a time of event (ToE) profile (i.e., ToE's calculated at different times of prediction)
"""
from ..predictors import ToEPredictionProfile

def alpha_lambda(toe_profile : ToEPredictionProfile, ground_truth : float, lambda_value : float, alpha : float, beta : float): 
    """Compute alpha lambda metric

    Args:
        toe_profile (ToEPredictionProfile): A profile of predictions, the combination of multiple predictions
        ground_truth (float): Ground Truth time of event for that event
        lambda_value (float): lambda
        alpha (float): alpha
        beta (float): beta

    Returns:
        bool: if alpha-lambda met
    """

    for (t_prediction, toe) in toe_profile.items():
        if (t_prediction >= lambda_value):
            upper_bound = ground_truth + alpha*(ground_truth-t_prediction)
            lower_bound = ground_truth - alpha*(ground_truth-t_prediction)
            return toe.percentage_in_bounds([lower_bound, upper_bound]) >= beta 
