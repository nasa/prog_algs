# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from abc import ABC, abstractmethod, abstractproperty
from ..visualize import plot_scatter, plot_hist


class UncertainData(ABC):
    """
    Abstract base class for data with uncertainty. Any new uncertainty type must implement this class
    """
    @abstractmethod
    def sample(self, nSamples : int = 1):
        """Generate samples from data

        Args:
            nSamples (int, optional): Number of samples to generate. Defaults to 1.

        Returns:
            samples (UnweightedSamples): Array of nSamples samples

        Example:
            samples = data.samples(100)
        """

    @property
    @abstractproperty
    def median(self):
        """The median of the UncertainData distribution or samples 

        Returns:
            Dict[str, float]: Mean value. e.g., {'key1': 23.2, ...}

        Example:
            median_value = data.median
        """

    @property
    @abstractproperty
    def mean(self):
        """The mean of the UncertainData distribution or samples 

        Returns:
            Dict[str, float]: Mean value. e.g., {'key1': 23.2, ...}

        Example:
            mean_value = data.mean
        """

    @property
    @abstractproperty
    def cov(self):
        """The covariance matrix of the UncertiantyData distribution or samples in order of keys (i.e., cov[1][1] is the standard deviation for key keys()[1])

        Returns:
            array[array[float]]: Covariance matrix
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

        Keyword Args:
            ground_truth (int or dict, optional): Ground truth value. Defaults to None.
            n_samples (int, optional): Number of samples to use for calculating metrics (if not UnweightedSamples)
            keys (List[str], optional): Keys to calculate metrics for. Defaults to all keys.

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
            states = UnweightedSamples([1, 2, 3, 4, 5])\n
            states.plot_scatter() # With 100 samples\n
            states.plot_scatter(num_samples=5) # Specifying the number of samples to plot\n
            states.plot_scatter(keys=['state1', 'state2']) # only plot those keys
        """
        if keys is None:
            keys = self.keys()
        samples = self.sample(num_samples)
        return plot_scatter(samples, fig=fig, keys=keys, **kwargs)

    def plot_hist(self, fig = None, keys = None, num_samples = 100, **kwargs):
        """Create a histogram

        Args:
            fig (MatPlotLib Figure, optional): Existing histogram figure to be overritten. Defaults to create new figure.
            num_samples (int, optional): Number of samples to plot. Defaults to 100
            keys (List(String), optional): Keys to be plotted. Defaults to None.
        """
        if keys is None:
            keys = self.keys()
        samples = self.sample(num_samples)
        return plot_hist(samples, fig=fig, keys=keys, **kwargs)
