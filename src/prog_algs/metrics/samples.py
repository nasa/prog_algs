# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from numpy import mean, std
from scipy import stats

def toe_metrics(toe, ground_truth = None):
    """Calculate all time of event metrics

    Args:
        toe ([double]): Times of event for a single event, output from predictor
        ground_truth (float, optional): Ground truth end of discharge time. Defaults to None.

    Returns:
        dict: collection of metrics
    """
    toe.sort()
    m = mean(toe)
    median = toe[int(len(toe)/2)]
    metrics = {
        'min': toe[0],
        'percentiles': {
            '0.01': toe[int(len(toe)/10000)] if len(toe) >= 10000 else None,
            '0.1': toe[int(len(toe)/1000)] if len(toe) >= 1000 else None,
            '1': toe[int(len(toe)/100)] if len(toe) >= 100 else None,
            '10': toe[int(len(toe)/10)] if len(toe) >= 10 else None,
            '25': toe[int(len(toe)/4)] if len(toe) >= 4 else None,
            '50': median,
            '75': toe[int(3*len(toe)/4)] if len(toe) >= 4 else None,
        },
        'median': median,
        'mean': m,
        'std': std(toe),
        'max': toe[-1],
        'median absolute deviation': sum([abs(x - median) for x in toe])/len(toe),
        'mean absolute deviation':   sum([abs(x - m)   for x in toe])/len(toe),
        'number of samples': len(toe)
    }

    if ground_truth is not None:
        # Metrics comparing to ground truth
        metrics['mean absolute error'] = sum([abs(x - ground_truth) for x in toe])/len(toe)
        metrics['mean absolute percentage error'] = metrics['mean absolute error']/ ground_truth
        metrics['relative accuracy'] = 1 - abs(ground_truth - metrics['mean'])/ground_truth
        metrics['ground truth percentile'] = stats.percentileofscore(toe, ground_truth)

    return metrics

def prob_success(toe, time):
    """Calculate probability of success - i.e., probability that event will not occur within a given time (i.e., success)

    Args:
        toe ([float]): Times of event for a single event, output from predictor
        time ([type]): time for calculation

    Returns:
        fload: Probability of success
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