# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

"""
This example performs state estimation and prediction using playback data. 
 
Method: An instance of the BatteryCircuit model in prog_models is created, the state estimation is set up by defining a state_estimator, and the prediction method is set up by defining a predictor.
        Prediction is then performed using playback data. For each data point:
        1) The necessary data is extracted (time, current load, output values) and corresponding values defined (t, i, and z)
        2) The current state estimate is performed and samples are drawn from this distribution
        3) Prediction performed to get future states (with uncertainty) and the times at which the event threshold will be reached
    
Results: 
    i) Predicted future values (inputs, states, outputs, event_states) with uncertainty from prediction
    ii) Time event is predicted to occur (with uncertainty)
    iii) Various prediction metrics
    iv) Figures illustrating results
"""

from prog_models.models import BatteryCircuit as Battery
# VVV Uncomment this to use Electro Chemistry Model VVV
# from prog_models.models import BatteryElectroChem as Battery

from prog_algs.state_estimators import ParticleFilter as StateEstimator
# VVV Uncomment this to use UnscentedKalmanFilter instead VVV
# from prog_algs.state_estimators import UnscentedKalmanFilter as StateEstimator

from prog_algs.predictors import MonteCarlo, ToEPredictionProfile
from prog_algs.metrics import samples as metrics

import csv
import matplotlib.pyplot as plt

from prog_algs.uncertain_data.multivariate_normal_dist import MultivariateNormalDist 
import numpy as np

# Constants
NUM_SAMPLES = 10
TIME_STEP = 1
PREDICTION_UPDATE_FREQ = 5 # Number of steps between prediction update

def run_example():
    # Setup Model
    batt = Battery()

    # Setup State Estimation with uncertainty
    x0 = batt.initialize()
    PROCESS_NOISE = 1e-4
    MEASUREMENT_NOISE = 1e-4
    X0_STD_DEV = 1 # Standard deviation as a percentage of state
    batt.parameters['process_noise'] = {key: PROCESS_NOISE * value for key, value in x0.items()}
    z0 = batt.output(x0)
    batt.parameters['measurement_noise'] = {key: MEASUREMENT_NOISE * value for key, value in z0.items()}
    x0 = MultivariateNormalDist(x0.keys(), list(x0.values()), np.diag([max(1e-9, X0_STD_DEV * abs(x)) for x in x0.values()]))
    filt = StateEstimator(batt, x0)

    # Setup Prediction
    def future_loading(t, x=None):
        return {'i': 2.35}
    mc = MonteCarlo(batt)

    # Prepare SOC Plot
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.grid()
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('SOC')
    xdata, ydata = [], []
    fig.show()

    # Prepare RUL Plot
    rul_fig, rulax = plt.subplots()
    rul_line, = rulax.plot([], [])
    rulax.grid()
    rulax.set_xlabel('Time (s)')
    rulax.set_ylabel('RUL (s)')
    rul_x, rul_y = [], []
    rul_fig.show()

    # Run Playback
    step = 0
    profile = ToEPredictionProfile()
    
    with open('examples/data_const_load.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            step += 1
            print("{} s: {} W, {} C, {} V".format(*row))
            t = float(row[0])
            i = {'i': float(row[1])/float(row[3])}
            z = {'t': float(row[2]), 'v': float(row[3])}

            # State Estimation Step
            filt.estimate(t, i, z) 
            eod = batt.event_state(filt.x.mean)['EOD']
            print("  - Event State: ", eod)

            # Update Plot
            xdata.append(t)
            ydata.append(eod)
            xmin, xmax = ax.get_xlim()

            if t >= xmax:
                ax.set_xlim(xmin, 2*xmax)
                rulax.set_xlim(xmin, 2*xmax)
            line.set_data(xdata, ydata)
            fig.canvas.draw()

            # Prediction Step (every PREDICTION_UPDATE_FREQ steps)
            if (step%PREDICTION_UPDATE_FREQ == 0):
                mc_results = mc.predict(filt.x, future_loading, n_samples=NUM_SAMPLES, dt=TIME_STEP)
                m = mc_results.time_of_event.metrics()
                print('  - ToE: {} (sigma: {})'.format(m['EOD']['mean'], m['EOD']['std']))

                # Update Plot
                rul_x.append(t)
                rul_y.append(m['EOD']['mean']-t)
                ymin, ymax = rulax.get_ylim()
                if m['EOD']['mean']-t > ymax:
                    rulax.set_ylim(0, (m['EOD']['mean']-t)*1.1)
                rul_line.set_data(rul_x, rul_y)
                rul_fig.canvas.draw()
                profile.add_prediction(t, mc_results.time_of_event)

    input('Press any key to exit')

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
