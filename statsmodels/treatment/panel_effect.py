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
        params, _ = optimize.nnls(self.exog, self.endog)
        return Results(self, params, method='OLSNonNegative')


    def predict(self, params, exog):
        if exog is None:
            exog = self.exog
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

        params0, _ = optimize.nnls(exog_, endog_)

        # minimisation function
        def fn(params, en, ex):
            return np.linalg.norm(ex.dot(params) - en)

        # constraints and bounds
        cons = {'type': 'eq', 'fun': lambda x:  np.sum(x)-1}
        bounds = [[0., 1]] * exog_.shape[1]

        params_start = params0.copy() + 0.01
        params_start /= params0.sum()

        minout = optimize.minimize(fn, params_start, args=(endog_, exog_),
                                   method='SLSQP', bounds=bounds,
                                   constraints=cons)
        params = minout.x

        return Results(self, params, method='OLSSimplexConstrained')


    def predict(self, params, exog):
        if exog is None:
            exog = self.exog
        predicted = exog.dot(params)
        return predicted


class PanelATTBasic(object):
    """Basic class for treatment effect estimation in panel data

    This is a simple version to get started.
    Assumes only one treated unit and several controls or predictors.

    Parameters
    ----------
    endog : array_like
        outcome series of the treated unit
    exog : array_like
        outcome variable of the control or untreated units,
        observation in rows, and control units in columns
    treatment_index : int
        index of the first treatment period.
        The average treatment effect is computed using all periods from
        treatment_index to the end.
        Estimation of the prediction model uses observations up to the
        treatment_index (excluding the treatment_index period)

    Notes
    -----
    Currently assumes treatment lasts from treatment_index to the end of the
    series. TODO: add optional end point

    endog, exog is for general prediction
    We don't separate yet matching variables from prediction, i.e. extra
    covariates in computing weights as in synthetic control (Abadie et al.)

    This works now in the basic case, but the design has problems to add
    cross-validation. It might be better to delegate more to the prediction
    model, e.g. add supporting function for OLS and fit-regularized to do
    the variable or penalization search.

    Status: experimental, options and API are still changing
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

        Parameters
        ----------
        constraints : string or None
            If constraints is None, then no constraints on the parameters are
            imposed. The implemented constraints are
            - "nonneg" or "nn": all parameters are nonnegative
            - "simplex" : all parameters are nonnegative and sum to one
        regularization : None or tuple (alpha, L1_wt)
            This uses the elastic net `fit_regularized` method of OLS. If it is
            not None, then the tuple needs to specify the elastic net
            penalization parameters `alpha` and `L1_wt`.
            Note, regularization cannot be specified at the same time
            as constraints
        add_const : bool
            Wether to add a constant to the prediction regression.
            Warning: the constant is currently subject to the same constraints
            or penalization as slope parameters.
            This will change, so that a constant will not be penalized or
            subject to constraints.

        Returns
        -------
        res : Results instance


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
            if ('nonneg' in constraints) or ('nn' in constraints) :
                res_fit = OLSNonNegative(y0_pre, exog).fit()

            elif 'simplex' in constraints:
                res_fit = OLSSimplexConstrained(y0_pre, exog).fit()
            else:
                raise ValueError('constraints not recognized, use nonneg or simplex')

        elif regularization is not None:
            model = OLS(y0_pre, exog)
            alpha, L1_wt = regularization[-2:]
            res_fit = model.fit_regularized(alpha=alpha, L1_wt=L1_wt)
        else:
            model = OLS(y0_pre, exog)
            res_fit = model.fit()



        res_att = PanelATTResults(model=self, res_fit=res_fit)
        return res_att


    def predict(self, result, exog):
        """predict for given exog using the results instance

        Note, this differs from the usual predict method in that it
        requires the results instance instead of `params`.

        This might change. Currently this class does not hold on to the
        prediction model.

        """
        # TODO: this doesn't work properly, we need already the fit instance
        # or we could just use linear prediction directly
        if self.add_const:
            exog = np.column_stack((np.ones(exog.shape[0]), exog))
        return result.res_fit.predict(exog)


class PanelATTResults(object):
    """Results class for PanelATT

    This class computes the prediction and treatment effect.

    API, signature unclear, What will be required?
    Currently only model is required and anything else is attached
    as optional.

    Attributes
    ----------
    model : PanelATT model instance
    res_fit : the results instance of the prediction model. This is currently
        optional but the currently used PanelATTBasic class provides this.
    att : average treatment effect, mean over treatment periods
    treatment_effect : prediction error for treatment periods
    predicted : predicted values for all periods
    prediction_error : prediction error for all periods

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
        """get a lowess fit for the outcome variable of the treated

        Not used yet

        Parameters
        ----------
        connected : bool
            If true, then lowess is fit to the entire all observations.
            If false, then lowess is fit separately to the pre-treatment
            and treatment periods.
            This method returns one array if false, and two arrays if true.
        frac : float
            lowess smoothing parameter, The same value is used for eacg
            part if connected is false.

        Returns
        -------
        smoothed: array or tuple of two arrays
            The smoothed series. See connected parameter.

        """
        y0 = self.endog

        trend = np.arange(y0.shape[0])
        if connected:
            smoothed = lowess(y0, trend, frac=0.25, return_sorted=False)
            return smoothed
        else:
            y0_pre = self.endog[self.slice_pre]
            y0_treat = self.endog[self.slice_treat]
            smoothed0 = lowess(y0_pre, trend[self.model.slice_pre], frac=0.25, return_sorted=False)
            smoothed1 = lowess(y0_treat, trend[self.model.slice_treat], frac=0.25, return_sorted=False)
            return smoothed0, smoothed1


    def plot(self, loc_legend=None, fig=None):
        """plot time series graph of data, prediction and prediction error

        This returns a figure with 3 axis.

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
        ax2.plot(np.zeros(len(self.model.endog)), color='k', lw=1)
        ax2.vlines(nobs_pre - 0.5, *ax2.get_ylim())
        #ax2.legend(loc="lower left")
        ax2.set_title("Prediction Error and Treatment Effect")
        return fig
