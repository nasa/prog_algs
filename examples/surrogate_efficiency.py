

### High-fidelity simulation: 
from prog_models.models import BatteryElectroChemEOD as Battery
# from prog_models.models import BatteryCircuit as Battery

from prog_algs import *
import timeit
import cProfile, pstats 

def run_example():
    # Set up model 
    R_vars = {
        't': 2, 
        'v': 0.02
    }
    batt = Battery(measurement_noise = R_vars)
    batt.parameters['process_noise'] = 0 #2e-05
    batt.parameters['process_noise']['qpS'] = 0.25
    batt.parameters['process_noise']['qpB'] = 0.25

    # Define future loading functions for DMD training data 
    def future_loading_1(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 600):
            i = 2
        elif (t < 900):
            i = 1
        elif (t < 1800):
            i = 4
        else:
            i = 2
        return batt.InputContainer({'i': i})
    
    def future_loading_2(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 200):
            i = 4
        elif (t < 500):
            i = 2
        elif (t < 1300):
            i = 1
        else:
            i = 6
        return batt.InputContainer({'i': i})
    
    def future_loading_3(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 1000):
            i = 2
        else:
            i = 6.5
        return batt.InputContainer({'i': i})
    
    def future_loading_4(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 700):
            i = 5.5
        elif (t < 1400):
            i = 1.3
        elif (t < 1800):
            i = 3.8
        else:
            i = 4
        return batt.InputContainer({'i': i})

    def future_loading_5(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 300):
            i = 1.2
        elif (t < 600):
            i = 1.8
        elif (t < 800):
            i = 4
        elif (t < 1000):
            i = 2.5
        elif (t < 1200):
            i = 5.3
        elif (t < 1600):
            i = 3.8
        else:
            i = 6.2
        return batt.InputContainer({'i': i})

    def future_loading_6(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 500):
            i = 2.7
        elif (t < 800):
            i = 5
        elif (t < 1200):
            i = 3.9
        elif (t < 1500):
            i = 2.5
        else:
            i = 6
        return batt.InputContainer({'i': i})

    def future_loading_7(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 200):
            i = 0.5
        elif (t < 400):
            i = 1.2
        elif (t < 600):
            i = 3.9
        elif (t < 800):
            i = 2.3
        elif (t < 1000):
            i = 0.8
        elif (t < 1400):
            i = 4.7
        elif (t < 1800):
            i = 3.1
        elif (t < 2200):
            i = 1
        else:
            i = 3
        return batt.InputContainer({'i': i})

    def future_loading_8(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 200):
            i = 2.5
        elif (t < 400):
            i = 3
        elif (t < 600):
            i = 1.7
        elif (t < 800):
            i = 2.3
        elif (t < 1000):
            i = 0.5
        elif (t < 1400):
            i = 2.2
        elif (t < 1800):
            i = 3.6
        elif (t < 2200):
            i = 2
        else:
            i = 3
        return batt.InputContainer({'i': i})

    def future_loading_9(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 800):
            i = 2.5
        else:
            i = 7
        return batt.InputContainer({'i': i})
    
    def future_loading_10(t, x=None):
        # Variable (piece-wise) future loading scheme 
        i = 1.5
        return batt.InputContainer({'i': i})

    def future_loading_11(t, x=None):
        # Variable (piece-wise) future loading scheme 
        i = 3
        return batt.InputContainer({'i': i})

    def future_loading_12(t, x=None):
        # Variable (piece-wise) future loading scheme 
        i = 4.5
        return batt.InputContainer({'i': i})

    def future_loading_13(t, x=None):
        # Variable (piece-wise) future loading scheme 
        i = 6
        return batt.InputContainer({'i': i})

    def future_loading_14(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 500):
            i = 2.5
        elif (t < 900):
            i = 3
        elif (t < 1400):
            i = 4
        elif (t < 2000):
            i = 1.3
        else:
            i = 2.6
        return batt.InputContainer({'i': i})

    def future_loading_15(t, x=None):
        if (t < 650):
            i = 1.5
        elif (t < 900):
            i = 3.9
        elif (t < 1600):
            i = 2.7
        elif (t < 2000):
            i = 1
        else:
            i = 3
        return batt.InputContainer({'i': i})

    def future_loading_16(t, x=None):
        if (t < 1000):
            i = 4.3
        elif (t < 1500):
            i = 2.9
        elif (t < 2000):
            i = 1
        else:
            i = 5
        return batt.InputContainer({'i': i})

    def future_loading_17(t, x=None):
        if (t < 1000):
            i = 2
        elif (t < 1500):
            i = 3
        elif (t < 2000):
            i = 1
        else:
            i = 4.5
        return batt.InputContainer({'i': i})

    def future_loading_18(t, x=None):
        if (t < 500):
            i = 2.5
        elif (t < 1000):
            i = 1.5
        elif (t < 2000):
            i = 3
        else:
            i = 5
        return batt.InputContainer({'i': i})

    def future_loading_19(t, x=None):
        if (t < 800):
            i = 3.8
        elif (t < 1000):
            i = 6
        else:
            i = 2.3
        return batt.InputContainer({'i': i})

    def future_loading_20(t, x=None):
        if (t < 800):
            i = 2
        elif (t < 1000):
            i = 4.5
        else:
            i = 1.5
        return batt.InputContainer({'i': i})
    
    load_functions = [future_loading_1, future_loading_2, future_loading_3, future_loading_4, future_loading_5, future_loading_6, 
                   future_loading_7, future_loading_8, future_loading_9, future_loading_10, future_loading_11, future_loading_12]

    ## Step 3: generate surrogate model 
    # Simulation options for training data and surrogate model generation
    # Note: here dt is less than save_freq. This means the model will iterate forward multiple steps per saved point.
    # This is commonly done to ensure accuracy. 
    options_surrogate = {
        'save_freq': 1, # For DMD, this value is the time step for which the surrogate model is generated
        'dt': 0.1, # For DMD, this value is the time step of the training data
        'trim_data_to': 0.8, # Value between 0 and 1 that determines the fraction of data resulting from simulate_to_threshold that is used to train DMD surrogate model
        'training_noise': 0.01,
        'outputs': ['v']
    }

    # Generate surrogate model  
    surrogate = batt.generate_surrogate(load_functions,**options_surrogate)

    # Define loading profile for prediction
    future_loading = future_loading_16

    # Step 3: Demonstrating Predictor
    print("\n\nPerforming Prediction Step")

    # Step 3a: Setup Predictor
    mc_surrogate = predictors.MonteCarlo(surrogate)
    mc_hf = predictors.MonteCarlo(batt)

    # Step 3b: Perform a prediction
    surrogate.parameters['process_noise'] = 0.002 # or percentage of values 
    batt.parameters['process_noise'] = 0.002
    NUM_SAMPLES = 50
    STEP_SIZE = 1
    options_approx = {
        'n_samples': NUM_SAMPLES,
        'save_freq': STEP_SIZE,
        'horizon': 5000
    }
    options_hf = {
        'n_samples': NUM_SAMPLES,
        'save_freq': STEP_SIZE,
        'dt': STEP_SIZE,
        'horizon': 5000
    }

    ### Time with profiler
    # profiler = cProfile.Profile()
    # x0_surrogate = surrogate.initialize()
    # profiler.enable() 
    # mc_results_surrogate = mc_surrogate.predict(x0_surrogate, future_loading, **options)
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats(0.1)

    # profiler = cProfile.Profile()
    # x0_hf = batt.initialize()
    # profiler.enable() 
    # mc_results_hf = mc_hf.predict(x0_hf, future_loading, **options)
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats(0.1)

    ### Time with timeit
    x0_surrogate = surrogate.initialize()
    x0_hf = batt.initialize()
    def time_surrogate_fcn():
        mc_results_surrogate = mc_surrogate.predict(x0_surrogate, future_loading, **options_approx)
    def time_hf_fcn():
        mc_results_hf = mc_hf.predict(x0_hf, future_loading, **options_hf)

    time_surrogate = timeit.timeit(time_surrogate_fcn,number=1)
    time_hf = timeit.timeit(time_hf_fcn,number=1)

    # mc_results_surrogate.time_of_event.plot_hist()
    # mc_results_hf.time_of_event.plot_hist()

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
