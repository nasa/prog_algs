# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from numpy import mean, std
from scipy import stats

def eol_metrics(eol, ground_truth = None):
    """Calculate all end of life metrics

    Args:
        eol ([double]): End of life times, output from predictor
        ground_truth (float, optional): Ground truth end of discharge time. Defaults to None.

    Returns:
        dict: collection of metrics
    """
    eol.sort()
    m = mean(eol)
    median = eol[int(len(eol)/2)]
    metrics = {
        'min': eol[0],
        'percentiles': {
            '0.01': eol[int(len(eol)/10000)] if len(eol) >= 10000 else None,
            '0.1': eol[int(len(eol)/1000)] if len(eol) >= 1000 else None,
            '1': eol[int(len(eol)/100)] if len(eol) >= 100 else None,
            '10': eol[int(len(eol)/10)] if len(eol) >= 10 else None,
            '25': eol[int(len(eol)/4)] if len(eol) >= 4 else None,
            '50': median,
            '75': eol[int(3*len(eol)/4)] if len(eol) >= 4 else None,
        },
        'median': median,
        'mean': m,
        'std': std(eol),
        'max': eol[-1],
        'median absolute deviation': sum([abs(x - median) for x in eol])/len(eol),
        'mean absolute deviation':   sum([abs(x - m)   for x in eol])/len(eol),
        'number of samples': len(eol)
    }

    if ground_truth is not None:
        # Metrics comparing to ground truth
        metrics['mean absolute error'] = sum([abs(x - ground_truth) for x in eol])/len(eol)
        metrics['mean absolute percentage error'] = metrics['mean absolute error']/ ground_truth
        metrics['relative accuracy'] = 1 - abs(ground_truth - metrics['mean'])/ground_truth
        metrics['ground truth percentile'] = stats.percentileofscore(eol, ground_truth)

    return metrics

def prob_success(eol, time):
    """Calculate probability of success - i.e., probability that event will not occur within a given time (i.e., success)

    Args:
        eol ([float]): End of life times, output from predictor
        time ([type]): time for calculation

    Returns:
        fload: Probability of success
    """
    return sum([e > time for e in eol])/len(eol)

def alpha_lambda(times, eols, ground_truth, lambda_value, alpha, beta): 
    """Compute alpha lambda metrics

    Args:
        times ([float]): Times corresponding to eols (output from predictors)
        eols ([float]): End of life times, output from predictor
        ground_truth (float): Ground Truth end of life time
        lambda_value (fload): lambda
        alpha (float): alpha
        beta (float): beta

    Returns:
        bool: if alpha-lambda met
    """
    for (t, eol) in zip(times, eols):
        if (t >= lambda_value):
            upper_bound = ground_truth + alpha*(ground_truth-t)
            lower_bound = ground_truth - alpha*(ground_truth-t)
            return percentage_in_bounds(eol, [lower_bound, upper_bound]) >= beta 


def mean_square_error(values, ground_truth):
    """Mean Square Error

    Args:
        values ([float]): End of life times, output from predictor
        ground_truth (float): Ground truth EOL

    Returns:
        float: mean square error of eol predictions
    """
    return sum([(x.mean() - ground_truth)**2 for x in values])/len(values)

def eol_profile_metrics(eol, ground_truth):
    """Calculate eol profile metrics

    Args:
        eol ([float]): End of life times, output from predictor
        ground_truth (float): Ground truth EOL

    Returns:
        dict: EOL Profile Metrics
    """
    # TODO(CT): Consider ground truth optional
    return {
        'mean square error': mean_square_error(eol, ground_truth)
    }

def percentage_in_bounds(eol, bounds):
    """Calculate percentage of EOL dist is within specified bounds

    Args:
        eol ([float]): End of life times, output from predictor
        bounds ([float, float]): Lower and upper bounds

    Returns:
        float: Percentage within bounds (where 1 = 100%)
    """
    return sum([x < bounds[1] and x > bounds[0] for x in eol])/ len(eol)