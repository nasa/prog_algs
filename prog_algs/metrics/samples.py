# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
from scipy import stats

def eol_metrics(rul, ground_truth = None):
    rul.sort()
    mean = np.mean(rul)
    median = rul[int(len(rul)/2)]
    metrics = {
        'min': rul[0],
        'percentiles': {
            '0.01': rul[int(len(rul)/10000)] if len(rul) >= 10000 else None,
            '0.1': rul[int(len(rul)/1000)] if len(rul) >= 1000 else None,
            '1': rul[int(len(rul)/100)] if len(rul) >= 100 else None,
            '10': rul[int(len(rul)/10)] if len(rul) >= 10 else None,
            '25': rul[int(len(rul)/4)] if len(rul) >= 4 else None,
            '50': median,
            '75': rul[int(3*len(rul)/4)] if len(rul) >= 4 else None,
        },
        'median': median,
        'mean': mean,
        'std': np.std(rul),
        'max': rul[-1],
        'median absolute deviation': sum([abs(x - median) for x in rul])/len(rul),
        'mean absolute deviation':   sum([abs(x - mean)   for x in rul])/len(rul),
        'number of samples': len(rul)
    }

    if ground_truth is not None:
        # Metrics comparing to ground truth
        metrics['mean absolute error'] = sum([abs(x - ground_truth) for x in rul])/len(rul)
        metrics['mean absolute percentage error'] = metrics['mean absolute error']/ ground_truth
        metrics['ground truth percentile'] = stats.percentileofscore(rul, ground_truth)

    return metrics

def percentage_in_bounds(rul, bounds):
    return sum([x < bounds[1] and x > bounds[0] for x in rul])/ len(rul)