

### High-fidelity simulation: 
from prog_models.models import BatteryElectroChemEOD as Battery
# from prog_models.models import BatteryCircuit as Battery

from prog_algs import *
# import timeit
import cProfile, pstats 

def run_example():
    # Set up model 
    R_vars = {
        't': 2, 
        'v': 0.02
    }
    batt = Battery(measurement_noise = R_vars)
    batt.parameters['process_noise'] = 0

    # Define future loading functions for DMD training data 
    def future_loading_1(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 500):
            i = 3
        elif (t < 1000):
            i = 2
        elif (t < 1500):
            i = 0.5
        else:
            i = 4.5
        return batt.InputContainer({'i': i})
    
    def future_loading_2(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 300):
            i = 2
        elif (t < 800):
            i = 3.5
        elif (t < 1300):
            i = 4
        elif (t < 1600):
            i = 1.5
        else:
            i = 5
        return batt.InputContainer({'i': i})
    
    load_functions = [future_loading_1, future_loading_2]

    ## Step 3: generate surrogate model 
    # Simulation options for training data and surrogate model generation
    # Note: here dt is less than save_freq. This means the model will iterate forward multiple steps per saved point.
    # This is commonly done to ensure accuracy. 
    options_surrogate = {
        'save_freq': 1, # For DMD, this value is the time step for which the surrogate model is generated
        'dt': 0.1, # For DMD, this value is the time step of the training data
        'trim_data_to': 0.7 # Value between 0 and 1 that determines the fraction of data resulting from simulate_to_threshold that is used to train DMD surrogate model
    }

    # Generate surrogate model  
    surrogate = batt.generate_surrogate(load_functions,**options_surrogate)

    # Define loading profile for prediction
    def future_loading(t, x=None):
        if (t < 600):
            i = 3
        elif (t < 1000):
            i = 2
        elif (t < 1500):
            i = 1.5
        else:
            i = 4
        return batt.InputContainer({'i': i})

    # Step 3: Demonstrating Predictor
    print("\n\nPerforming Prediction Step")

    # Step 3a: Setup Predictor
    mc_surrogate = predictors.MonteCarlo(surrogate)
    mc_hf = predictors.MonteCarlo(batt)

    # Step 3b: Perform a prediction
    surrogate.parameters['process_noise'] = 0.002 # or percentage of values 
    batt.parameters['process_noise'] = 0.002
    NUM_SAMPLES = 10
    STEP_SIZE = 1
    options = {
        'n_samples': NUM_SAMPLES,
        'save_freq': STEP_SIZE,
        'horizon': 5000
    }

    profiler = cProfile.Profile()
    x0_surrogate = surrogate.initialize()
    profiler.enable() 
    mc_results_surrogate = mc_surrogate.predict(x0_surrogate, future_loading, **options)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(0.1)

    profiler = cProfile.Profile()
    x0_hf = batt.initialize()
    profiler.enable() 
    mc_results_hf = mc_hf.predict(x0_hf, future_loading, **options)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(0.1)

    # Adding large step size/fast surrogate 
    # options['save_freq'] = options_surrogate['save_freq'] = 100
    options['step_size'] = options_surrogate['save_freq'] = 100
    batt.parameters['process_noise'] = 0

    surrogate_large_step = batt.generate_surrogate(load_functions,**options_surrogate)

    surrogate_large_step.parameters['process_noise'] = 2e-5

    mc_surrogate_large_step = predictors.MonteCarlo(surrogate_large_step)

    x0_surrogate_large_step = surrogate_large_step.initialize()

    profiler = cProfile.Profile()
    profiler.enable()
    mc_results_surrogate_large_step = mc_surrogate_large_step.predict(x0_surrogate_large_step, future_loading, **options)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(0.1)

    debug = 1


# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
