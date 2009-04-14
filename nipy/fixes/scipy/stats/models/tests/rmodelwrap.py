#!/bin/env python
''' Wrapper for R models to allow comparison to scipy models '''

import numpy as np

from rpy import r

from exampledata import x, y

class RModel(object):
    ''' Class gives R models scipy.models -like interface '''
    def __init__(self, y, design, model_type=r.lm):
        ''' Set up and estimate R model with data and design '''
        self.y = y
        self.design = design
        self.model_type = model_type
        self._design_cols = ['x.%d' % (i+1) for i in range(
            self.design.shape[1])]
        # Note the '-1' for no intercept - this is included in the design
        self.formula = r('y ~ %s-1' % '+'.join(self._design_cols))
        self.frame = r.data_frame(y=y, x=self.design)
        self.results = self.model_type(self.formula,
                                    data = self.frame)
        # Provide compatible interface with scipy models
        coeffs = self.results['coefficients']
        self.beta = np.array([coeffs[c] for c in self._design_cols])
        self.resid = self.results['residuals']
        self.predict = self.results['fitted.values']
        self.df_resid = self.results['df.residual']


