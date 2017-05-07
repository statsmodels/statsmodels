# -*- coding: utf-8 -*-
"""
Created on Sat May  6 22:56:22 2017

Author: Josef Perktold
"""

import numpy as np
from scipy import optimize

from statsmodels.base.model import Model, Results
from statsmodels.regression.linear_model import OLS
from statsmodels.nonparametric.smoothers_lowess import lowess


class OLSNonNegative(Model):
    """Basic prediction model for OLS with nonnegativity constraints

    This model currently does not have inference.
    The inference is non-standard if we don't know which inequality
    constraints are binding.

    Parameter index_unconstrained not yet supported
    """


    def __init__(self, endog, exog, index_unconstrained=None):
        super(OLSNonNegative, self).__init__(endog, exog)


    def fit(self):
        params, rnorm = optimize.nnls(self.exog, self.endog)
        return Results(self, params, method='OLSNonNegative')


    def predict(self, params, exog):
        predicted = exog.dot(params)
        return predicted


class OLSSimplexConstrained(Model):
    """Basic prediction model for OLS with simplex constraints

    This constrains params to be nonnegative and sum to one.

    This model currently does not have inference.
    The inference is non-standard if we don't know which inequality
    constraints are binding.

    Parameter index_unconstrained not yet supported
    """


    def __init__(self, endog, exog, index_unconstrained=None):
        super(OLSSimplexConstrained, self).__init__(endog, exog)


    def fit(self, w=None):
        """fit the model with constraints using slsqp

        weights w is for experimenting, should go into init if WLS
        """
        # w = 0.9 ** (nobs_pre - np.arange(nobs_pre) - 1) in experiment
        if w is not None:
            endog_ = w * self.endog
            exog_ = w * self.exog
        else:
            # we don't need a copy
            endog_ = self.endog
            exog_ = self.exog

        params0, rnorm = optimize.nnls(exog_, endog_)

        #Define minimisation function
        def fn(params, en, ex):
            return np.linalg.norm(ex.dot(params) - en)

        #Define constraints and bounds
        cons = {'type': 'eq', 'fun': lambda x:  np.sum(x)-1}
        bounds = [[0., 1]] * exog_.shape[1]

        #Call minimisation subject to these values
        params_start = params0.copy() + 0.01
        params_start /= params0.sum()

        minout = optimize.minimize(fn, params_start, args=(endog_, exog_),
                                   method='SLSQP', bounds=bounds,
                                   constraints=cons)
        params = minout.x

        return Results(self, params, method='OLSSimplexConstrained')


    def predict(self, params, exog):
        predicted = exog.dot(params)
        return predicted



class PanelATTBasic(object):
    """Basic class for treatment effect estimation in panel data

    This is a simple version to get started.
    Assumes only one treated unit

    Currently assumes treatment lasts from treatment_index to the end of the
    series. TODO: add optional end point

    endog, exog is for general prediction
    We don't separate yet matching variables from prediction, i.e. extra
    covariates in computing weights as in synthetic control (Abadie et al.)

    This works now in the basic case, but the design has problems to add
    cross-validation. It might be better to delegate more to the prediction
    model, e.g. add supporting function for OLS and fit-regularized to do
    the variable or penalization search.

    Also we need basic prediction models that use nnls and slsqp with
    constraints, so we can delegate and get required method like predict.

    """

    def __init__(self, endog, exog, treatment_index):
        self.endog = np.asarray(endog)
        self.exog = np.asarray(exog)
        self.treatment_index = treatment_index
        self.nobs_pre = treatment_index
        self.slice_pre = slice(None, treatment_index)
        self.slice_treat = slice(treatment_index, None)


    def fit(self, constraints=None, regularization=None, add_const=True):
        """fit the prediction model

        constraints and regularization are currently mutually exclusive, only
        one of the too can be different from None.

        I'm adding `add_const` option separately from the constraints because
        it can be used with OLS and with regularized.
        (just implmentation leak?)

        """
        self.add_const = add_const
        y0_pre = self.endog[self.slice_pre]
        y1_pre = self.exog[self.slice_pre]
        if constraints is not None and regularization is not None:
            raise ValueError("only one of constraints and regularization can"
                             " be used")

        if add_const:
            exog = np.column_stack((np.ones(y0_pre.shape[0]), y1_pre))
        else:
            exog = y1_pre



        if constraints is not None:
            #print('******** using constraints:', constraints)
            if ('nonneg' in constraints) or ('nn' in constraints) :
                res_fit = OLSNonNegative(y0_pre, exog).fit()

            elif 'simplex' in constraints:
                #print('******** using simplex')
                res_fit = OLSSimplexConstrained(y0_pre, exog).fit()
            else:
                raise ValueError('constraints not recognized, use nonneg or simplex')
            #raise NotImplementedError('not yet, WIP')

        elif regularization is not None:
            model = OLS(y0_pre, exog)
            alpha, L1_wt = regularization[-2:]
            res_fit = model.fit_regularized(alpha=alpha, L1_wt=L1_wt)
        else:
            model = OLS(y0_pre, exog)
            res_fit = model.fit()



        res_att =  PanelATTResults(model=self, res_fit=res_fit)
                                   #predicted=predicted)
        return res_att

    def predict(self, result, exog):
        # TODO: this doesn't work properly, we need already the fit instance
        # or we just use linear prediction directly
        if self.add_const:
            exog = np.column_stack((np.ones(exog.shape[0]), exog))
        return result.res_fit.predict(exog)


class PanelATTResults(object):
    """Results class for PanelATT

    API, signature unclear, What will be required?
    currently attach anything as optional

    """
    def __init__(self, model, **kwds):
        self.__dict__.update(kwds)
        self.model = model
        self.predicted = self.model.predict(self, self.model.exog)
        self.prediction_error = self.model.endog - self.predicted
        self.treatment_effect = self.prediction_error[self.model.slice_treat]

    @property
    def att(self):
        return self.treatment_effect.mean(0)

    def get_smoothed(self, connected=False, frac=0.25):
        y0 = self.endog
        y0_pre = self.endog[self.slice_pre]
        y0_treat = self.endog[self.slice_treat]
        trend = np.arange(y0.shape[0])
        if connected:
            smoothed = lowess(y0, trend, frac=0.25, return_sorted=False)
            return smoothed
        else:
            smoothed0 = lowess(y0_pre, trend[self.model.slice_pre], frac=0.25, return_sorted=False)
            smoothed1 = lowess(y0_treat, trend[self.model.slice_treat], frac=0.25, return_sorted=False)
            return smoothed0, smoothed1

    def plot(self, loc_legend=None, fig=None):
        """plot time series graph,

        This does not yet conform to statsmodels API standard
        This needs an option to plot scatter plus lowess instead of lines.
        """
        # TODO: use graphics helper function
        import matplotlib.pyplot as plt
        nobs_pre = self.model.treatment_index
        if fig is None:
            fig = plt.figure()
        ax0 = fig.add_subplot(3,1,1)
        ax0.plot(self.model.endog, lw=2, label="treated")
        ax0.plot(self.model.exog, lw=2, alpha=0.5)
        ax0.vlines(nobs_pre - 0.5, *ax0.get_ylim())
        ax0.legend(loc=loc_legend)
        ax0.set_title('Observed series Treated and Controls')

        ax1 = fig.add_subplot(3,1,2)
        ax1.plot(self.model.endog, lw=2, label="observed")
        ax1.plot(self.predicted, lw=2, label="predicted")
        ax1.vlines(nobs_pre - 0.5, *ax1.get_ylim())
        ax1.legend(loc=loc_legend)
        ax1.set_title("Observed and Predicted")

        ax2 = fig.add_subplot(3,1,3)
        ax2.plot(self.model.endog - self.predicted, lw=2)
        ax2.plot(np.zeros(len(self.model.endog)), lw=2)
        ax2.vlines(nobs_pre - 0.5, *ax2.get_ylim())
        #ax2.legend(loc="lower left")
        ax2.set_title("Prediction Error and Treatment Effect")
        return fig


