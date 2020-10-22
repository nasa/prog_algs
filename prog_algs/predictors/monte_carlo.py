from . import predictor

class MonteCarlo(predictor.Predictor):
    """
    Class for performing model-based prediction using sampling. 

    This class defines logic for performing model-based state prediction using sampling methods. A Predictor is constructed using a PrognosticsModel object, (See Prognostics Model Package). The states are simulated until either a specified time horizon is met, or the threshold is reached for all samples, as defined by the threshold equation. A provided future loading equation is used to compute the inputs to the system at any given time point. 
    """
    parameters = { # Default Parameters
        'dt': 0.5,          # Timestep, seconds
        'horizon': 4000,    # Prediction horizon, seconds
        'num_samples': 100, # Number of samples used
        'save_freq': 10     # Frequency at which results are saved
    }

    def __init__(self, model):
        """
        Construct a MonteCarlo Predictor

        Parameters
        ----------
        model : prog_models.prognostics_model.PrognosticsModel
            See: Prognostics Model Package
            A prognostics model to be used in prediction
        """
        self._model = model

    def predict(self, state_sampler, future_loading_eqn, options: dict = {}):
        """
        Perform a single prediction

        Parameters
        ----------
        state_sampler : function (n) -> [x1, x2, ... xn]
            Function to generate n samples of the state. 
            e.g., def f(n): return [x1, x2, x3, ... xn]
        future_loading_eqn : function (t) -> z
            Function to generate an estimate of loading at future time t
        options : dict, optional
            Dictionary of any additional configuration values. See default parameters, above

        Return
        ______
        result : recorded values for all samples
        """
        params = self.parameters # copy default parameters
        params.update(options)

        state_samples = state_sampler(params['num_samples'])
        results = []
        for x in state_samples:
            first_output = self._model.output(0, x)
            params['x'] = x
            result = self._model.simulate_to_threshold(future_loading_eqn, first_output, params)
            if (self._model.threshold_met(result['t'][-1], result['x'][-1])):
                result['EOL'] = result['t'][-1]
            else:
                result['EOL'] = None
            results.append(result)
            # TODO(CT): Add noise
        return results
        # TODO(CT): Consider other return types (e.g., table)
            
        
