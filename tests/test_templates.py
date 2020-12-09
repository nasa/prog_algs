# Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
from state_estimator_template import TemplateStateEstimator
from predictor_template import TemplatePredictor

class TestTemplates(unittest.TestCase):
    def test_state_est_template(self):
        se = TemplateStateEstimator(None)

    def test_pred_template(self):
        pred = TemplatePredictor(None)