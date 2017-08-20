# -*- coding: utf-8 -*-
"""

Created on Fri Jun 28 14:19:26 2013

Author: Josef Perktold
"""


import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel

from numpy.testing import assert_array_less, assert_almost_equal, assert_allclose

class MyPareto(GenericLikelihoodModel):
    '''Maximum Likelihood Estimation pareto distribution

    first version: iid case, with constant parameters
    '''

    def initialize(self):   #TODO needed or not
        super(MyPareto, self).initialize()
        #start_params needs to be attribute
        self.start_params = np.array([1.5, self.endog.min() - 1.5, 1.])


    #copied from stats.distribution
    def pdf(self, x, b):
        return b * x**(-b-1)

    def loglike(self, params):
        return -self.nloglikeobs(params).sum(0)

    # TODO: design start_params needs to be an attribute,
    # so it can be overwritten
#    @property
#    def start_params(self):
#        return np.array([1.5, self.endog.min() - 1.5, 1.])

    def nloglikeobs(self, params):
        #print params.shape
        if not self.fixed_params is None:
            #print 'using fixed'
            params = self.expandparams(params)
        b = params[0]
        loc = params[1]
        scale = params[2]
        #loc = np.dot(self.exog, beta)
        endog = self.endog
        x = (endog - loc)/scale
        logpdf = np.log(b) - (b+1.)*np.log(x)  #use np_log(1 + x) for Pareto II
        logpdf -= np.log(scale)
        #lb = loc + scale
        #logpdf[endog<lb] = -inf
        #import pdb; pdb.set_trace()
        logpdf[x<1] = -10000 #-np.inf
        return -logpdf


class CheckGenericMixin(object):
    #mostly smoke tests for now


    def test_summary(self):
        self.res1.summary()
        #print self.res1.summary()

    def test_ttest(self):
        self.res1.t_test(np.eye(len(self.res1.params)))

    def test_params(self):
        params = self.res1.params

        params_true = np.array([2,0,2])
        if self.res1.model.fixed_paramsmask is not None:
            params_true = params_true[self.res1.model.fixed_paramsmask]
        assert_allclose(params, params_true, atol=1.5)
        assert_allclose(params, np.zeros(len(params)), atol=4)

        assert_allclose(self.res1.bse, np.zeros(len(params)), atol=0.5)
        if not self.skip_bsejac:
            assert_allclose(self.res1.bse, self.res1.bsejac, rtol=0.05,
                            atol=0.15)
            # bsejhj is very different from the other two
            # use huge atol as sanity check for availability
            assert_allclose(self.res1.bsejhj, self.res1.bsejac,
                            rtol=0.05, atol=1.5)







class TestMyPareto1(CheckGenericMixin):

    @classmethod
    def setup_class(cls):
        params = [2, 0, 2]
        nobs = 100
        np.random.seed(1234)
        rvs = stats.pareto.rvs(*params, **dict(size=nobs))

        mod_par = MyPareto(rvs)
        mod_par.fixed_params = None
        mod_par.fixed_paramsmask = None
        mod_par.df_model = 3
        mod_par.df_resid = mod_par.endog.shape[0] - mod_par.df_model
        mod_par.data.xnames = ['shape', 'loc', 'scale']

        cls.mod = mod_par
        cls.res1 = mod_par.fit(disp=None)

        # Note: possible problem with parameters close to min data boundary
        # see issue #968
        cls.skip_bsejac = True

    def test_minsupport(self):
        # rough sanity checks for convergence
        params = self.res1.params
        x_min = self.res1.endog.min()
        p_min = params[1] + params[2]
        assert_array_less(p_min, x_min)
        assert_almost_equal(p_min, x_min, decimal=2)

class TestMyParetoRestriction(CheckGenericMixin):


    @classmethod
    def setup_class(cls):
        params = [2, 0, 2]
        nobs = 50
        np.random.seed(1234)
        rvs = stats.pareto.rvs(*params, **dict(size=nobs))

        mod_par = MyPareto(rvs)
        fixdf = np.nan * np.ones(3)
        fixdf[1] = -0.1
        mod_par.fixed_params = fixdf
        mod_par.fixed_paramsmask = np.isnan(fixdf)
        mod_par.start_params = mod_par.start_params[mod_par.fixed_paramsmask]
        mod_par.df_model = 2
        mod_par.df_resid = mod_par.endog.shape[0] - mod_par.df_model
        mod_par.data.xnames = ['shape', 'scale']

        cls.mod = mod_par
        cls.res1 = mod_par.fit(disp=None)

        # Note: loc is fixed, no problems with parameters close to min data
        cls.skip_bsejac = False
