''' Wrapper for R models to allow comparison to scipy models '''

import numpy as np

import rpy
from check_for_rpy import skip_rpy
skipR = skip_rpy()
if not skipR:
    from rpy import r

# MASS contains
# negative binomial family for GLM
# rlm

#TODO: write a check_key wrapper for these

class RModel(object):
    ''' Class gives R models scipy.models -like interface '''
    def __init__(self, y, design, model_type=r.lm, **kwds):
        ''' Set up and estimate R model with data and design '''
        r.library('MASS') # still needs to be in test, but also here for
                          # logical tests at the end not to show an error
        self.y = np.array(y)
        self.design = np.array(design)
        self.model_type = model_type
        self._design_cols = ['x.%d' % (i+1) for i in range(
            self.design.shape[1])]
        # Note the '-1' for no intercept - this is included in the design
        self.formula = r('y ~ %s-1' % '+'.join(self._design_cols))
        self.frame = r.data_frame(y=y, x=self.design)
        rpy.set_default_mode(rpy.NO_CONVERSION)
        results = self.model_type(self.formula,
                                    data = self.frame, **kwds)
        self.robj = results # keep the Robj model so it can be
                            # used in the tests
        rpy.set_default_mode(rpy.BASIC_CONVERSION)
        rsum = r.summary(results)
        self.rsum = rsum
        # Provide compatible interface with scipy models
        self.results = results.as_py()

#        coeffs = self.results['coefficients']
#        self.beta0 = np.array([coeffs[c] for c in self._design_cols])
        self.nobs = len(self.results['residuals'])
        if isinstance(self.results['residuals'], dict):
            self.resid = np.zeros((len(self.results['residuals'].keys())))
            for i in self.results['residuals'].keys():
                self.resid[int(i)-1] = self.results['residuals'][i]
        else:
            self.resid = self.results['residuals']
        self.fittedvalues = self.results['fitted.values']
        self.df_resid = self.results['df.residual']
        self.params = rsum['coefficients'][:,0]
        self.bse = rsum['coefficients'][:,1]
        self.bt = rsum['coefficients'][:,2]
        try:
            self.pvalues = rsum['coefficients'][:,3]
        except: pass
        self.rsquared = rsum.setdefault('r.squared', None)
        self.rsquared_adj = rsum.setdefault('adj.r.squared', None)
        self.aic_R = rsum.setdefault('aic', None)
        self.fvalue = rsum.setdefault('fstatistic', None)
        if self.fvalue and isinstance(self.fvalue, dict):
            self.fvalue = self.fvalue.setdefault('value', None) # for wls
        df = rsum.setdefault('df', None)
        if df:  # for RLM, works for other models?
            self.df_model = df[0]-1 # R counts intercept
            self.df_resid = df[1]
        self.bcov_unscaled = rsum.setdefault('cov.unscaled', None)
        self.bcov = rsum.setdefault('cov.scaled', None)
        if 'sigma' in rsum:
            self.scale = rsum['sigma']
        elif 'dispersion' in rsum:
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
        if isinstance(self.resid, dict):
            tmp = np.zeros(len(self.resid))
            for i in self.resid.keys():
                tmp[int(i)-1] = self.resid[i]
            self.resid = tmp
        self.predict = [self.results['linear.predictors'][str(k)] \
                for k in range(1, 1+self.nobs)]
        self.fittedvalues = [self.results['fitted.values'][str(k)] \
                for k in range(1, 1+self.nobs)]
        self.weights = [self.results['weights'][str(k)] \
                for k in range(1, 1+self.nobs)]
        self.resid_deviance = self.rsum['deviance.resid']
        if isinstance(self.resid_deviance, dict):
            tmp = np.zeros(len(self.resid_deviance))
            for i in self.resid_deviance.keys():
                tmp[int(i)-1] = self.resid_deviance[i]
            self.resid_deviance = tmp
        self.null_deviance = self.rsum['null.deviance']

    def getrlm(self):
        self.k2 = self.results['k2']
        if isinstance(self.results['w'], dict):
            tmp = np.zeros((len(self.results['w'].keys())))
            for i in self.results['w'].keys():
                tmp[int(i)-1] = self.results['w'][i]
            self.weights = tmp
        else: self.weights = self.results['w']
        self.stddev = self.rsum['stddev'] # Don't know what this is yet
        self.wresid = None # these equal resids always?

#TODO:
# function to write Rresults to results file, so this is a developers tool
# and not a test dependency?
def RModelConvert(model, sec_title=None, results_title=None):
    import os
    if not results_title:
        raise AttributeError("You need to specify a results title")
    outfile = open('./model_results.py', 'a')
    outfile.write('class '+results_title)
    outfile.write(' '*4)    # handle indents

