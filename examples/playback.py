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

import csv
import matplotlib.pyplot as plt

# Constants
NUM_SAMPLES = 10
TIME_STEP = 1
PREDICTION_UPDATE_FREQ = 50 # Number of steps between prediction update
PLOT = True

def run_example():
    # Setup Model
    batt = Battery()

    # Setup State Estimation
    filt = StateEstimator(batt, batt.parameters['x0'])

    # Setup Prediction
    def future_loading(t, x=None):
        return {'i': 2.35}
    mc = MonteCarlo(batt)

    if PLOT:
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

            if PLOT:
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
                mc_results = mc.predict(filt.x, future_loading, t0 = t, n_samples=NUM_SAMPLES, dt=TIME_STEP)
                m = mc_results.time_of_event.metrics()
                print('  - ToE: {} (sigma: {})'.format(m['EOD']['mean'], m['EOD']['std']))

                # Update Plot
                rul_x.append(t)
                rul_y.append(m['EOD']['mean']-t)
                _, ymax = rulax.get_ylim()
                if m['EOD']['mean']-t > ymax:
                    rulax.set_ylim(0, (m['EOD']['mean']-t)*1.1)
                rul_line.set_data(rul_x, rul_y)
                rul_fig.canvas.draw()
                profile.add_prediction(t, mc_results.time_of_event)

        # Calculating Prognostic Horizon once for loop complete
        from prog_algs.uncertain_data.uncertain_data import UncertainData
        from prog_algs.metrics import samples as metrics, prognostic_horizon

        ground_truth = {'EOD':3306.5}
        def criteria_eqn(tte : UncertainData, ground_truth_tte : dict) -> dict:
            """
            Sample criteria equation for playback. 
            # UPDATE THIS CRITERIA EQN AND WHAT IS CALCULATED

            Args:
                tte : UncertainData
                    Time to event in UncertainData format.
                ground_truth_tte : dict
                    Dictionary of ground truth of time to event.
            """
            # given ground truth x, y% of distribution between x+a and x-a where a is percentage of tte
            # example: Given a t_p of 100, a ground truth of 500, and predictions of type unweighted samples of 445, 470, 495, 520, and 545, 
            # if the beta is 0.6 (i.e., 60% must be within alpha bounds), an alpha of 0.1 would be met. 
            # Because true TtE is 400, and 60% is within 40 (i.e. 0.1 * 400), but an alpha of 0.05 would not be met, because only 40% is within 20 of 500
            result = {}
            for key, value in ground_truth_tte.items():
                result[key] = abs(tte.mean[key] - value) < 10 # GROUND TRUTH VALUE HERE 
                # because it only needs to meet EOD, returns 1 value 
                # Prognostic Horizon: {'EOD': 2207.5}  because first value met when abs(tte.mean[key] - value) < 10
            return result

        ph = prognostic_horizon(profile, criteria_eqn, ground_truth)
        print(f"Prognostic Horizon: {ph}")
        # printing for single event, but ph calculates for all predicted events MAKE COMMENT NICER

    input('Press any key to exit')

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
