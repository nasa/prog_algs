# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from timeit import timeit

def eol_stats(rul):
    rul.sort()
    return {
        'min': rul[0],
        'mean': rul[int(len(rul)/2)],
        'max': rul[-1]
    }
    