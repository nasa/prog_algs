import numpy as np

def generate_mean_cov_random_sampler(model, x_array, Q):
    def sampler(num_samples):
        samples = np.random.multivariate_normal(x_array, Q, num_samples)
        samples = [{key: value for (key, value) in zip(model.states, x)} for x in samples]
        return samples
    return sampler