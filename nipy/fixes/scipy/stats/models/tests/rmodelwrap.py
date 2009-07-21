''' Wrapper for R models to allow comparison to scipy models '''

import numpy as np

import rpy
from rpy import r
# MASS contains
# negative binomial family for GLM
# rlm

#TODO: write a check_key wrapper for these

class RModel(object):
    ''' Class gives R models scipy.models -like interface '''
    def __init__(self, y, design, model_type=r.lm, **kwds):
        ''' Set up and estimate R model with data and design '''
        rpy.set_default_mode(rpy.NO_CONVERSION)
        self.y = y
        self.design = design
        self.model_type = model_type
        self._design_cols = ['x.%d' % (i+1) for i in range(
            self.design.shape[1])]
        # Note the '-1' for no intercept - this is included in the design
        self.formula = r('y ~ %s-1' % '+'.join(self._design_cols))
        self.frame = r.data_frame(y=y, x=self.design)
        r.library('MASS')
        results = self.model_type(self.formula,
                                    data = self.frame, **kwds)
        rpy.set_default_mode(rpy.BASIC_CONVERSION)
        rsum = r.summary(results)
        self.rsum = rsum
        # Provide compatible interface with scipy models
        self.results = results.as_py()
#        coeffs = self.results['coefficients']
#        self.beta0 = np.array([coeffs[c] for c in self._design_cols])
        self.nobs = len(self.results['residuals'])
        self.resid = self.results['residuals']
        self.predict = self.results['fitted.values']
        #hasattr(rglmtestpoiss_res.resid, 'keys')
        #self.resid = [self.results['residuals'][str(k)] for k in range(1, 1+self.nobs)]
        #self.predict = [self.results['fitted.values'][str(k)] for k in range(1, 1+self.nobs)]
        self.df_resid = self.results['df.residual']
        self.params = rsum['coefficients'][:,0]
        self.bse = rsum['coefficients'][:,1]
        self.bt = rsum['coefficients'][:,2]
        self.bpval = rsum['coefficients'][:,3]
        self.R2 = rsum.setdefault('r.squared', None)
        self.adjR2 = rsum.setdefault('adj.r.squared', None)
        self.aic_R = rsum.setdefault('aic', None)
        self.F = rsum.setdefault('fstatistic', None)
        self.df = rsum['df']     # which df is this?
        self.bcov_unscaled = rsum['cov.unscaled']
        self.bcov = rsum.setdefault('cov.scaled', None)
#        self.df_resid = self.results['df.residual']    #RLM

        if rsum.has_key('sigma'):
            self.scale = rsum['sigma']
        elif rsum.has_key('dispersion'):
            self.scale = rsum['dispersion']
        else:
            self.scale = None
        self.llf = r.logLik(results)

        if model_type == r.glm:
            self.getglm()
        if model_type == r.rlm:
            self.getrlm()

    def getglm(self):
        self.deviance = self.rsum['deviance']
        self.resid = [self.results['residuals'][str(k)] \
                for k in range(1, 1+self.nobs)]
        self.predict = [self.results['linear.predictors'][str(k)] \
                for k in range(1, 1+self.nobs)]
        self.predictedy = [self.results['fitted.values'][str(k)] \
                for k in range(1, 1+self.nobs)]
        self.weights = [self.results['weights'][str(k)] \
                for k in range(1, 1+self.nobs)]
        self.resid_dev = self.rsum['deviance.resid']
        self.null_deviance = self.rsum['null.deviance']

    def getrlm(self):
        self.k2 = None
        self.weights = None
        self.stddev = None # is this SER?


#TODO:
# function to write Rresults to results file, so this is a developers tool
# and not a test dependency
def RModelConvert(model, sec_title=None, results_title=None):
    import os
    if not results_title:
        raise AttributeError, "You need to specify a results title"
    outfile = open('./model_results.py', 'a')
    outfile.write('class '+results_title)
    outfile.write(' '*4)    # handle indents

