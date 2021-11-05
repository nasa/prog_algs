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

    @property
    @abstractproperty
    def median(self):
        """Median estimate

        Example:
            median_value = data.median
        """

    @property
    @abstractproperty
    def mean(self):
        """Mean estimate

        Example:
            mean_value = data.mean
        """

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

    def percentage_in_bounds(self, bounds, keys = None):
        """Calculate percentage of dist is within specified bounds

        Args:
            bounds ([float, float]): Lower and upper bounds
            keys (list of strings, optional): Keys to analyze. Defaults to all keys.

        Returns:
            float: Percentage within bounds (where 0.5 = 50%)
        """
        return self.sample(1000).percentage_in_bounds(bounds)

    def metrics(self, **kwargs):
        """Calculate Metrics for this dist

        Args
            **kwargs (optional): Configuration parameters. Supported parameters include:
             * ground_truth (int or dict): Ground truth value. Defaults to None.
             * n_samples (int): Number of samples to use for calculating metrics (if not UnweightedSamples)
             * keys (list of strings): Keys to calculate metrics for. Defaults to all keys.

        Returns:
            dict: Dictionary of metrics
        """
        from ..metrics import calc_metrics
        return calc_metrics(self, **kwargs)

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
        if keys is None:
            keys = self.keys()
        samples = self.sample(num_samples)
        return plot_scatter(samples, fig=fig, keys=keys, **kwargs)
