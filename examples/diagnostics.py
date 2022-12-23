import matplotlib.pyplot as plt
import numpy as np
from prog_algs.state_estimators import ParticleFilter
from prog_algs.uncertain_data import UnweightedSamples
from prog_models.models import PneumaticValveBase
import warnings

STEP = 0.1
DT = 0.05
NUM_PARTICLES = 100e3
N_STEPS = 20
PROCESS_NOISE_BASE = 2.5e-4
MEASURE_NOISE_BASE = 8e-3
config = {
        'dt': DT,
        'save_freq': STEP,
    }
valv = PneumaticValveBase(process_noise = 0)
cycle_time = 20
def future_loading(t, x=None):
    t = t % cycle_time
    if t < cycle_time/2:
        return valv.InputContainer({
            'pL': 3.5e5,
            'pR': 2.0e5,
            # Open Valve
            'uTop': False,
            'uBot': True
        })
    return valv.InputContainer({
        'pL': 3.5e5,
        'pR': 2.0e5,
        # Close Valve
        'uTop': True,
        'uBot': False
    })

# Starting position (x, v == 0) with unknown other parameters
states = [{'Aeb': aeb, 'Aet': aet, 'Ai': ai, 'k': k, 'mBot': mbot, 'mTop': mtop, 'r': r, 'v': 0, 'x': 0, 'pDiff': 150000}
    for aeb in [0 + i * 2.5e-5 for i in range(3)]
    for aet in [0 + i * 2.5e-5 for i in range(3)]
    for ai in [0 + i * 1e-6 for i in range(3)]
    for k in [4.8e4 - i * 5000 for i in range(3)]
    for mbot in [0.04 * i for i in range(3)]
    for mtop in [0.04 * i for i in range(3)]
    for r in [1.5e6 * i for i in range(3)]]

x0 = UnweightedSamples(states, _type=valv.StateContainer)

# Generate some fake data
# In this case, there's a leak in the bottom of the valve
# it is slightly above the limit for the bottom leak event
results = []

MAX_T = STEP*N_STEPS+1
valv.parameters['x0']['Aeb'] = 5e-5
results.append(("Bottom Leak", valv.simulate_to(MAX_T, future_loading, **config), 'Aeb'))
valv.parameters['x0']['Aeb'] = valv.default_parameters['x0']['Aeb']

valv.parameters['x0']['Aet'] = 5e-5
results.append(("Top Leak", valv.simulate_to(MAX_T, future_loading, **config), 'Aet'))
valv.parameters['x0']['Aet'] = valv.default_parameters['x0']['Aet']

valv.parameters['x0']['Ai'] = 2e-6
results.append(("Internal Leak", valv.simulate_to(MAX_T, future_loading, **config), 'Ai'))
valv.parameters['x0']['Ai'] = valv.default_parameters['x0']['Ai']

valv.parameters['x0']['k'] = 3.5e4
results.append(("Spring Failure", valv.simulate_to(MAX_T, future_loading, **config), 'k'))
valv.parameters['x0']['k'] = valv.default_parameters['x0']['k']

valv.parameters['x0']['r'] = 4.5e6
results.append(("Friction Failure", valv.simulate_to(MAX_T, future_loading, **config), 'r'))
valv.parameters['x0']['r'] = valv.default_parameters['x0']['r']

process_noise = {
    'Aeb': PROCESS_NOISE_BASE, 
    'Aet': PROCESS_NOISE_BASE, 
    'Ai': PROCESS_NOISE_BASE/25, 
    'k': PROCESS_NOISE_BASE * 3e8, 
    'mBot': PROCESS_NOISE_BASE*1e2, 
    'mTop': PROCESS_NOISE_BASE*1e2, 
    'r': PROCESS_NOISE_BASE * 2e10, 
    'v': PROCESS_NOISE_BASE * 1e-7, 
    'x': 0.01, 
    'pDiff': 1e-99}

measurement_noise = {
    'Q': MEASURE_NOISE_BASE, 
    'iB': 1e99, 
    'iT': 1e99, 
    'pB': MEASURE_NOISE_BASE,
    'pT': MEASURE_NOISE_BASE, 
    'x': MEASURE_NOISE_BASE*1e-3}
valv = PneumaticValveBase(process_noise = process_noise, measurement_noise = 0)

# State estimation 
for failure, simulated_results, state_key in results:
    print('\nActive Failure: ', failure)
    filt = ParticleFilter(valv, x0, num_particles=NUM_PARTICLES, measurement_noise= measurement_noise)

    fig = plt.figure()
    gs = fig.add_gridspec(3, 3)
    fig.suptitle(failure)
    (ax00, ax01, ax02), (ax10, ax11, ax12), (ax20, ax21, ax22) = gs.subplots(sharex='col', sharey=False)
    axs = [ax00, ax01, ax02, ax10, ax11, ax12, ax20, ax21, ax22]
    for key, ax in zip(valv.states, axs):
        x = filt.x.key(key)
        ax.scatter([0]*len(x), np.array(x)-np.array([simulated_results.states[0][key]]*len(x)), label=key, color='blue')
        ax.set_ylabel(key)
        if key == state_key:
            ax.set_facecolor('pink')

    SKIP = True
    for i in range(N_STEPS):
        t = simulated_results.times[i]
        u = simulated_results.inputs[i]
        z = simulated_results.outputs[i]
        gt = simulated_results.states[i]

        if SKIP:
            # Skip the first few step
            SKIP = False
            continue

        with warnings.catch_warnings():
            # Ignore warnings about state limiting
            warnings.simplefilter("ignore")
            filt.estimate(t, u, z, dt = DT)

        for key, ax in zip(gt.keys(), axs):
            x = filt.x.key(key)
            ax.scatter([t]*len(x), np.array(x)-np.array([gt[key]]*len(x)), label=key, color='orange')
        
        t_met = {key: [] for key in valv.events}
        for x in filt.x:
            tm = valv.threshold_met(x)
            for key in tm.keys():
                t_met[key].append(tm[key])
        
        print(f't = {t}', {
            key: sum(t_met[key])/len(t_met[key]) for key in t_met.keys()
        })
    fig.show()
print('done')