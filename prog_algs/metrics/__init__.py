# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from timeit import timeit

def eol_stats(rul):
    rul.sort()
    mean = sum(rul)/len(rul)
    median = rul[int(len(rul)/2)]
    return {
        'min': rul[0],
        'median': median,
        'mean': mean,
        'max': rul[-1],
        'median absolute deviation': sum([abs(x - median) for x in rul])/len(rul),
        'mean absolute deviation':   sum([abs(x - mean)   for x in rul])/len(rul)
    }
    