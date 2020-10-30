import numpy as np

def generate_mean_cov_random_sampler(labels, means, Q):
    """
    Generate a multiviarate sampler using mean and covariance matrix

    Parameters
    ----------
    labels : array of strings
        Labels corresponding to each value in x/Q
    means : array of numbers
        array of mean values
    Q : 2D array of numbers
        covariance matrix
    

    Return
    ------
    sampler : function 
        Function that maps between num_samples -> samples (array of dicts)
    """
    if len(means) != len(labels):
        raise Exception("labels must be provided for each value")
 
    def sampler(num_samples):
        samples = np.random.multivariate_normal(means, Q, num_samples)
        samples = [{key: value for (key, value) in zip(labels, x)} for x in samples]
        return samples
    return sampler