# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

"""
This example performs a state estimation with uncertainty given a Prognostics Model for a system in which not all output values are measured. 
 
Method: An instance of the BatteryCircuit model in prog_models is created. We assume that we are only measuring one of the output values, and we define a measurement_eqn to remove the other output value.  
        Estimation of the current state is performed at various time steps, using the defined state_estimator. The state_estimator takes the measurement_eqn as input, to account for the missing output information. 

Results: 
    i) Estimate of the current state given various times
    ii) Display of results, such as prior and posterior state estimate values and SOC
"""

from prog_models.models import BatteryCircuit as Battery
# VVV Uncomment this to use Electro Chemistry Model VVV
# from prog_models.models import BatteryElectroChem as Battery

from prog_algs import *

def run_example():
    # Step 1: Setup model & future loading
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

    batt = Battery()
    x0 = batt.parameters['x0']

    # Step 2: Define Measurement Equation
    # This example is a little different. Here we are saying that we dont have the complete output. Instead we're only measuring voltage. We use the measurement eqn to remove temp
    def measure(x):
        output = batt.output(x)
        del output['t'] # Not measuring temperature
        return output

    # Step 3: Setup particle filter to use measurement eqn
    filt = state_estimators.ParticleFilter(batt, x0, measurement_eqn = measure)

    # Step 4: Run step and print results
    print('Running state estimation step with only one of 2 outputs measured')

    # Print Prior
    print("\nPrior State:", filt.x.mean)
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])

    # Estimate Step
    t = 0.1
    load = future_loading(t)
    filt.estimate(t, load, {'v': 3.915})

    # Print Posterior
    print("\nPosterior State:", filt.x.mean)
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])

    # Another Estimate Step
    t = 0.2
    load = future_loading(t)
    filt.estimate(t, load, {'v': 3.91})

    # Print Posterior Again
    print("\nPosterior State (t={}):".format(t), filt.x.mean)
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])

    # Note that the particle filter was still able to perform state estimation.
    # The measurement_eqn can be used for any case where the measurement doesn't match the model outputs
    # For example, when units are different, or when the measurement is some combination of the outputs

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
