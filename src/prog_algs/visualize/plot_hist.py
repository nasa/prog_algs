# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import matplotlib.pyplot as plt

def plot_hist(samples, fig = None, keys = None, **kwargs):
    """Create a histogram

    Args:
        samples (Array(Dict)): A set of samples (e.g., states or eol estimates)
        fig (MatPlotLib Figure, optional): Existing histogram figure to be overritten. Defaults to create new figure.
        keys (List(String), optional): Keys to be plotted. Defaults to None.
    """

    # Input checks
    if len(samples) <= 0:
        raise Exception('Must include atleast one sample to plot')
    
    if keys is not None:
        try:
            iter(keys)
        except TypeError:
            raise TypeError("Keys should be a list of strings (e.g., ['state1', 'state2'], was {}".format(type(keys)))
        
        for key in keys:
            if key not in samples[0].keys():
                raise TypeError("Key {} was not present in samples (keys: {})".format(key, list(samples[0].keys())))
    
    # Handle input
    parameters = {  # defaults
        'legend': True
    }
    parameters.update(kwargs)

    if keys is None:
        keys = samples[0].keys()
    keys = list(keys)

    if fig is None:
        # If no figure provided, create one
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.axes[0]
    
    # Plot
    for key in keys:
        ax.hist([sample[key] for sample in samples], label=key, **kwargs)

    # Set legend
    if parameters['legend']:
        ax.legend().remove()  # Remove any existing legend - prevents "ghost effect"
        ax.legend(loc='upper right')