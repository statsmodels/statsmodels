import numpy as np
import statsmodels.api as sm
import summary_stats as ss

# need to put in init_args and fit_args. Then do error checking for
# model_class, args, and y

class SurveyModel(object):

    def __init__(self, design, model_class, y, init_args=None, fit_args=None):
        self.design = design
        self.model = model_class
        self.init_args = init_args
        self.fit_args = fit_args
        self.y = y

    def fit(self, data):
        self.results = self.model(self.y, data).fit()

