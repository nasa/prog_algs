from prog_models.models import battery_circuit
from prog_algs.state_estimators import unscented_kalman_filter

## Setup
def future_loading(t):
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

batt = battery_circuit.BatteryCircuit()

## State Estimation - perform a single ukf state estimate step
ukf = unscented_kalman_filter.UnscentedKalmanFilter(batt, future_loading, batt.parameters['x0'])
print("Prior State:", ukf.x, '\n\tlog-likelihood', ukf.filter.log_likelihood)
print('\tSOC: ', batt.event_state(ukf.t, ukf.x)['EOD'])
ukf.estimate(0.1, {'t': 32.2, 'v': 3.915})
print("Posterior State:", ukf.x, '\n\tlog-likelihood', ukf.filter.log_likelihood)
print('\tSOC: ', batt.event_state(ukf.t, ukf.x)['EOD'])

## Prediction - Predict EOD given current state
from prog_algs.predictors import monte_carlo
from prog_algs import samplers
mc = monte_carlo.MonteCarlo(batt)
state_sampler = samplers.generate_mean_cov_random_sampler(batt.states, list(ukf.x.values()), ukf.Q)
prediction_config = {'dt': 0.025, 'num_samples':5}
(times, inputs, states, outputs, event_states, eol) = mc.predict(state_sampler, future_loading, prediction_config)

# Print Prediction Results
print("\nEOD Predictions (s):")
print('\t', eol)