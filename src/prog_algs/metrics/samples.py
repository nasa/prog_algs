# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

# This file is kept for backwards compatability
from .uncertain_data_metrics import calc_metrics as eol_metrics
from .toe_metrics import prob_success
from .toe_profile_metrics import alpha_lambda

from warnings import warn

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
