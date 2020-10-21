from prog_algs.observers import unscented_kalman_filter
import sys
sys.path.insert(1, "/Users/cteubert/Desktop/python-prognostics-models-package/")
from prog_models.models import battery_circuit
batt = battery_circuit.BatteryCircuit()

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

ukf = unscented_kalman_filter.UnscentedKalmanFilter(batt, {'input_eqn': future_loading, 'x0':batt.parameters['x0']})
print(ukf.filter.x)
ukf.step(0.1, {'t': 32.2, 'v': 3.915})
print(ukf.x, 'log-likelihood', ukf.filter.log_likelihood)
print('event_state', batt.event_state(ukf.t, ukf.x))

from prog_algs.predictors import monte_carlo
mc = monte_carlo.MonteCarlo(batt)
def state_sampler(num_samples):
    return [ukf.x]
results = mc.predict(state_sampler, future_loading)

for i in range(len(results[0]['t'])):
    print(results[0]['t'][i], results[0]['event_state'][i])