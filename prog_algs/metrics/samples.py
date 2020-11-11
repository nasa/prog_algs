# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

def eol_stats(rul, ground_truth = None):
    rul.sort()
    mean = sum(rul)/len(rul)
    median = rul[int(len(rul)/2)]
    stats = {
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
        'max': rul[-1],
        'median absolute deviation': sum([abs(x - median) for x in rul])/len(rul),
        'mean absolute deviation':   sum([abs(x - mean)   for x in rul])/len(rul),
        'number of samples': len(rul)
    }

    if ground_truth is not None:
        # Metrics comparing to ground truth
        stats['mean absolute error'] = sum([abs(x - ground_truth) for x in rul])/len(rul)
        stats['mean absolute percentage error'] = stats['mean absolute error']/ ground_truth
        #TODO(CT): More, percentile?

    return stats

def percentage_in_bounds(rul, bounds):
    return sum([x < bounds[1] and x > bounds[0] for x in rul])/ len(rul)