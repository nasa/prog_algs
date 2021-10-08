# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

"""
This example performs a state estimation and prediction using playback data. 
 
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

# Constants
num_samples = 10
time_step = 0.1
prediction_update_freq = 5 # Number of steps between prediction update

def run_example():
    from prog_models.models import BatteryCircuit
    from prog_algs.metrics import samples as metrics
    from prog_algs import state_estimators, predictors

    import csv
    import matplotlib.pyplot as plt 

    # Setup Model
    batt = BatteryCircuit()

    # Setup State Estimation
    filt = state_estimators.UnscentedKalmanFilter(batt, batt.parameters['x0'])

    # Setup Prediction
    def future_loading(t, x=None):
        return {'i': 2.35}
    mc = predictors.MonteCarlo(batt)

    # Prepare Plot
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.grid()
    xdata, ydata = [], []
    plt.show()

    # Run Playback
    step = 0
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
                ax.figure.canvas.draw()
            line.set_data(xdata, ydata)

            # Prediction Step (every prediction_update_freq steps)
            if (step%prediction_update_freq == 0):
                samples = filt.x.sample(10)
                (times, inputs, states, outputs, event_states, eol) = mc.predict(samples, future_loading, dt=1.0)
                m = metrics.eol_metrics(eol)
                print('  - RUL: {} (sigma: {})'.format(m['mean'], m['std']))

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
