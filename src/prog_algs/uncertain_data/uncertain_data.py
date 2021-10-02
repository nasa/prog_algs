# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from abc import ABC, abstractmethod, abstractproperty
from ..visualize import plot_scatter


class UncertainData(ABC):
    """
    Abstract base class for data with uncertainty. Any new uncertainty type must implement this class
    """
    @abstractmethod
    def sample(self, nSamples = 1):
        """Generate samples from data

        Args:
            nSamples (int, optional): Number of samples to generate. Defaults to 1.

        Returns:
            samples (array): Array of nSamples samples

        Example:
            samples = data.samples(100)
        """
        pass

    @property
    @abstractproperty
    def mean(self):
        """Mean estimate

        Example:
            mean_value = data.mean
        """
        pass

    @property
    @abstractproperty
    def cov(self):
        """Get the covariance matrix

        Returns:
            [[float]]: covariance matrix
        """

    @abstractmethod
    def keys(self):
        """Get the keys for the property represented

        Returns:
            [string]: keys
        """

    def plot_scatter(self, fig = None, keys = None, num_samples = 100, **kwargs):
        """
        Produce a scatter plot

        Args:
            fig (Figure, optional): Existing figure previously used to plot states. If passed a figure argument additional data will be added to the plot. Defaults to creating new figure
            keys (list of strings, optional): Keys to plot. Defaults to all keys.
            num_samples (int, optional): Number of samples to plot. Defaults to 100
            **kwargs (optional): Additional keyword arguments passed to scatter function.

        Returns:
            Figure

        Example:
            states = UnweightedSamples([1, 2, 3, 4, 5])
            states.plot_scatter() # With 100 samples
            states.plot_scatter(num_samples=5) # Specifying the number of samples to plot
            states.plot_scatter(keys=['state1', 'state2']) # only plot those keys
        """
        samples = self.sample(num_samples)
        return plot_scatter(samples, fig=fig, keys=keys, **kwargs)
