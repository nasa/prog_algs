# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example using playback data. Builds a Battery Model, runs prognostics using the playback data. 
"""

# Constants
num_samples = 10
time_step = 0.1
prediction_update_freq = 5 # Number of steps between prediction update

def run_example():
    import csv
    from sys import path

    path.insert(1, "../prog_models/")
    from prog_models.models.battery_circuit import BatteryCircuit
    from prog_algs.metrics import samples as metrics
    from prog_algs import state_estimators, predictors

    import matplotlib.pyplot as plt 

    # Setup Model
    batt = BatteryCircuit()

    # Setup State Estimation
    filt = state_estimators.unscented_kalman_filter.UnscentedKalmanFilter(batt, batt.parameters['x0'])

    # Setup Prediction
    def future_loading(t, x=None):
        return {'i': 2.35}
    mc = predictors.monte_carlo.MonteCarlo(batt)

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
