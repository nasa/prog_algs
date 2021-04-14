# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import sys
sys.path.insert(1, "../prog_models/")
from prog_models.models.battery_circuit import BatteryCircuit
from prog_algs import *

def run_example():
    ## Setup
    def future_loading(t, x={}):
        # Variable (piece-wise) future loading scheme 
        if (t < 600):
            i = 2
        elif (t < 900):
            i = 1
        elif (t < 1800):
            i = 4
        elif (t < 3000):
            i = 2
        else:
            i = 3
        return {'i': i}

    batt = BatteryCircuit()

    ## State Estimation - perform a single ukf state estimate step
    # This example is a little different. Here we are saying that we dont have the complete output. Instead we're only measuring voltage. We use the measurement eqn to remove temp

    def measure(x):
        output = batt.output(x)
        del output['t'] # Not measuring temperature
        return output

    # Setup particle filter to use measurement eqn
    filt = state_estimators.particle_filter.ParticleFilter(batt, batt.parameters['x0'], measurement_eqn = measure)

    # Simulate results
    print("Prior State:", filt.x.mean)
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])
    t = 0.1
    load = future_loading(t)
    filt.estimate(t, load, {'v': 3.915})
    print("Posterior State:", filt.x.mean)
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])

    t = 0.2
    load = future_loading(t)
    filt.estimate(t, load, {'v': 3.91})
    print("Posterior State (t={}):".format(t), filt.x.mean)
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()