# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from typing import Iterable
from numpy import isscalar, mean, std
from scipy import stats
from ..uncertain_data import UncertainData, UnweightedSamples
from .import toe_metrics

def prob_success(toe, time):
    """Calculate probability of success - i.e., probability that event will not occur within a given time (i.e., success)

    Args:
        toe ([float]): Times of event for a single event, output from predictor
        time ([type]): time for calculation

    Returns:
        float: Probability of success
    """
    return sum([e > time for e in toe])/len(toe)
eol_metrics = toe_metrics  # For backwards compatability

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
    return sum([x < bounds[1] and x > bounds[0] for x in toe])/ len(toe)