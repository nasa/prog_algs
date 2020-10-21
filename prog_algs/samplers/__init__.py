import numpy as np

def generate_mean_cov_random_sampler(model, x, Q):
    if type(x) is dict:
        x = np.array(list(x.values()))
    def sampler(num_samples):
        samples = np.random.multivariate_normal(x, Q, num_samples)
        samples = [{key: value for (key, value) in zip(model.states, x)} for x in samples]
        return samples
    return sampler