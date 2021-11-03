from typing import Iterable
from numpy import isscalar, mean, std, array
from scipy import stats
from numbers import Number
from ..uncertain_data import UncertainData, UnweightedSamples, MultivariateNormalDist

def toe_metrics(toe, ground_truth = None, **kwargs):
    """Calculate all time of event metrics

    Args:
        toe (array[float] or UncertainData): Times of event for a single event, output from predictor
        ground_truth (float, optional): Ground truth end of discharge time. Defaults to None.
        **kwargs (optional): Configuration parameters. Supported parameters include:
          * n_samples (int): Number of samples to use for calculating metrics (if toe is not UnweightedSamples). Defaults to 10,000.

    Returns:
        dict: collection of metrics
    """
    params = {
        'n_samples': 10000,  # Default is enough to get every percentile
    }
    params.update(kwargs)
    
    if isinstance(toe, UncertainData):
        if ground_truth and isscalar(ground_truth):
            # If ground truth is scalar, create dict (expected below)
            ground_truth = {key: ground_truth for key in toe.keys()}

        if isinstance(toe, UnweightedSamples):
            samples = toe
        else:
            # Some other distribution besides unweighted samples
            # Generate Samples
            samples = toe.sample(params['n_samples'])

        # If unweighted_samples, calculate metrics for each key
        result = {key: toe_metrics(samples.key(key), 
                ground_truth if not ground_truth else ground_truth[key],  # If ground_truth is a dict, use key
                **kwargs) for key in samples.keys()}

        # Set values specific to distribution
        for key in toe.keys():
            result[key]['mean'] = toe.mean[key]
            result[key]['median'] = toe.median[key]
            result[key]['percentiles']['50'] = toe.median[key]

        return result

    elif isinstance(toe, Iterable):
        if len(toe) == 0:
            raise ValueError('Time of Event must not be empty')
        # Is list or array
        if isscalar(toe[0]):
            # list of numbers - this is the case that we can calculate
            pass
        elif isinstance(toe[0], dict):
            # list of dicts - Supported for backwards compatabilities
            toe = UnweightedSamples(toe)
            return toe_metrics(toe, ground_truth, **kwargs)
        else:
            raise TypeError("ToE must be type Uncertain Data or array of dicts, was {}".format(type(toe)))
    else:
        raise TypeError("ToE must be type Uncertain Data or array of dicts, was {}".format(type(toe)))

    # If we get here then ToE is a list of numbers- calculate metrics for numbers
    toe = array(toe)  # Must be array
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
