from . import predictor

class MonteCarlo(predictor.Predictor):
    """
    """
    parameters = { # Default Parameters
        'dt': 0.1,
        'horizon': 3000,
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
            results.append(result)
            # TODO(CT): Add noise
        return results
            
        
