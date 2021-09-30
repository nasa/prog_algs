# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
import matplotlib.pyplot as plt
from statistics import mean

def plot_line(times, data, keys = None, fig = None):
    """Plot Line Chart with Uncertainty Bounds

    Args:
        times ([double]): Times that the data corresponds to.
        data ([type]): [description]
        keys ([type], optional): [description]. Defaults to None.
        fig ([type], optional): [description]. Defaults to None.

    Raises:
        TypeError: [description]
        TypeError: [description]
    """
    parameters = {  # Default parameters
        'legend': True
    }

    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)
    
    if keys is not None:
        try:
            iter(keys)
        except TypeError:
            raise TypeError("Keys should be a list of strings (e.g., ['state1', 'state2'], was {}".format(type(keys)))
        
        for key in keys:
            if key not in data[0][0].keys():
                raise TypeError("Key {} was not present in samples (keys: {})".format(key, list(data[0][0].keys())))
    else:
        keys = data[0][0].keys()

    transposed_data = [data.snapshot(i) for i in range(len(times))]

    for key in keys:
        specific_data = [[sample[key] for sample in snapshot if sample is not None] for snapshot in transposed_data]
        means = [mean(d) for d in specific_data]
        mins = [min(d) for d in specific_data]
        maxs = [max(d) for d in specific_data]
        line = plt.plot(times, means, label=key)[0]
        color = line.get_color()
        plt.fill_between(times, mins, maxs, color=color+"55")

    plt.xlabel('Time (s)')
    plt.ylim(0, 1)
    plt.xlim(times[0], times[-1])

    # Set legend
    if parameters['legend']:
        plt.legend().remove()  # Remove any existing legend - prevents "ghost effect"
        plt.legend(loc='upper right')
    