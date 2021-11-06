# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from prog_models.models.thrown_object import ThrownObject
from prog_algs import *
import matplotlib.pyplot as plt  # For plotting
from prog_algs.visualize import plot_line

def run_example():
    ## Setup
    def future_loading(t, x = None):
        return {}
    m = ThrownObject(process_noise = 0)


    ## Prediction - Predict EOD given current state
    # Setup prediction
    mc = predictors.MonteCarlo(m)

    # Predict with a step size of 0.1
    mean = m.initialize({}, {})
    dist = uncertain_data.MultivariateNormalDist(['x', 'v'], list(mean.values()), [[0.01, 0], [0, 1e-4]])
    samples = dist.sample(100)
    print(samples)
    print([m.event_state(x) for x in samples])
    (times, inputs, states, outputs, event_states, eol) = mc.predict(samples, future_loading, dt=0.1)
    
    # Plot result
    plot_line(times[0], event_states)
    plt.show()

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
