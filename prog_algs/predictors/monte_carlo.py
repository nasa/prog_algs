from . import predictor

class MonteCarlo(predictor.Predictor):
    """
    """
    parameters = { # Default Parameters
        'dt': 0.5,
        'horizon': 4000,
        'num_samples': 100,
        'save_freq': 10
    }

    def __init__(self, model):
        self._model = model

    def predict(self, state_sampler, future_loading_eqn, options: dict = {}):
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
            
        
