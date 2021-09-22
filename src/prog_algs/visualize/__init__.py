from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt

def plot_scatter(samples, fig = None, keys = None, **kwargs):
    """
    Produce a scatter plot for a given list of states

    Args:
        samples ([type]): [description]
        keys ([type]): [description]
        fig ([type], optional): [description]. Defaults to None.
        keys ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    # Checks
    if len(samples) <= 0:
        raise Exception('Must include atleast one sample to plot')

    default_parameters = {

    }
    default_parameters.update(kwargs)

    if keys is None:
        keys = samples[0].keys()
    keys = list(keys)

    n = len(keys)
    if fig is None:
        fig = plt.figure()
        axes = [[fig.add_subplot(n-1, n-1, 1 + i + j*(n-1)) for i in range(n-1)] for j in range(n-1)]
    else:
        axes = [[fig.axes[i + j*(n-1)] for i in range(n-1)] for j in range(n-1)]

    for i in range(n-1):
        key1 = keys[i]
        axes[-1][i].set_xlabel(key1)
        axes[i][0].set_ylabel(keys[i+1])
        for j in range(i, n-1): 
            key2 = keys[j+1]
            x1 = [x[key1] for x in samples]
            x2 = [x[key2] for x in samples]
            axes[j][i].scatter(x1, x2, **kwargs)

        # Hide axes not used in plots 
        for j in range(0, i):
            axes[j][i].set_visible(False)
            # axes[j][i].get_yaxis().set_visible(False)

    labels = [thing.get_label() for thing in axes[0][0].get_children()
        if type(thing) is PathCollection]
    if len(labels) > 0:
        fig.legend(labels=labels, loc='upper right')

    return fig