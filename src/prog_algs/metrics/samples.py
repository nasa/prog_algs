# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from numpy import mean

# For backwards compatability
from .general_metrics import calc_metrics as eol_metrics
from .toe_metrics import prob_success
from warnings import warn

def alpha_lambda(times, toes, ground_truth, lambda_value, alpha, beta): 
    """Compute alpha lambda metrics

    Args:
        times ([float]): Times corresponding to toes (output from predictors)
        toes ([float]): Times of event for a single event, output from predictor
        ground_truth (float): Ground Truth time of event for that event
        lambda_value (float): lambda
        alpha (float): alpha
        beta (float): beta

    Returns:
        bool: if alpha-lambda met
    """
    for (t, toe) in zip(times, toes):
        if (t >= lambda_value):
            upper_bound = ground_truth + alpha*(ground_truth-t)
            lower_bound = ground_truth - alpha*(ground_truth-t)
            return percentage_in_bounds(toe, [lower_bound, upper_bound]) >= beta 


def mean_square_error(values, ground_truth):
    """Mean Square Error

    Args:
        values ([float]): time of event for a single event, output from predictor
        ground_truth (float): Ground truth ToE

    Returns:
        float: mean square error of toe predictions
    """
    return sum([(mean(x) - ground_truth)**2 for x in values])/len(values)

def toe_profile_metrics(toe, ground_truth):
    """Calculate toe profile metrics

    Args:
        toe ([float]): Times of event for a single event, output from predictor
        ground_truth (float): Ground truth toe

    Returns:
        dict: toe Profile Metrics
    """
    return {
        'mean square error': mean_square_error(toe, ground_truth)
    }

def percentage_in_bounds(toe, bounds):
    """Calculate percentage of ToE dist is within specified bounds

    Args:
        toe ([float]): Times of event for a single event, output from predictor
        bounds ([float, float]): Lower and upper bounds

    Returns:
        float: Percentage within bounds (where 1 = 100%)
    """
    warn('percentage_in_bounds has been deprecated in favor of UncertainData.percentage_in_bounds(bounds). This function will be removed in a future release')
    return sum([x < bounds[1] and x > bounds[0] for x in toe])/ len(toe)
