# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:29:18 2014

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import stats
import pandas as pd


# this is based on PredictionResults, copied and adjusted
class PredictionResultsBase(object):

    def __init__(self, predicted, var_pred, func=None, deriv=None,
                 df=None, dist=None, row_labels=None, **kwds):
        self.predicted = predicted
        self.var_pred = var_pred
        self.func = func
        self.deriv = deriv
        self.df = df
        self.row_labels = row_labels
        self.__dict__.update(kwds)

        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    @property
    def se(self):
        return np.sqrt(self.var_pred)

    @property
    def tvalues(self):
        return self.predicted / self.se

    def t_test(self, value=0, alternative='two-sided'):
        '''z- or t-test for hypothesis that mean is equal to value

        Parameters
        ----------
        value : array_like
            value under the null hypothesis
        alternative : str
            'two-sided', 'larger', 'smaller'

        Returns
        -------
        stat : ndarray
            test statistic
        pvalue : ndarray
            p-value of the hypothesis test, the distribution is given by
            the attribute of the instance, specified in `__init__`. Default
            if not specified is the normal distribution.

        '''
        # assumes symmetric distribution
        stat = (self.predicted_mean - value) / self.se_mean

        if alternative in ['two-sided', '2-sided', '2s']:
            pvalue = self.dist.sf(np.abs(stat), *self.dist_args)*2
        elif alternative in ['larger', 'l']:
            pvalue = self.dist.sf(stat, *self.dist_args)
        elif alternative in ['smaller', 's']:
            pvalue = self.dist.cdf(stat, *self.dist_args)
        else:
            raise ValueError('invalid alternative')
        return stat, pvalue

    def _conf_int_generic(self, center, se, alpha, dist_args=None):
        """internal function to avoid code duplication
        """
        if dist_args is None:
            dist_args = ()

        q = self.dist.ppf(1 - alpha / 2., *dist_args)
        lower = center - q * se
        upper = center + q * se
        ci = np.column_stack((lower, upper))
        # if we want to stack at a new last axis, for lower.ndim > 1
        # np.concatenate((lower[..., None], upper[..., None]), axis=-1)
        return ci

    def conf_int(self, *, alpha=0.05, **kwds):
        """
        Returns the confidence interval of the value, `effect` of the
        constraint.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        kwds : extra keyword arguments
            currently ignored, only for compatibility, consistent signature

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """

        ci = self._conf_int_generic(self.predicted, self.se, alpha,
                                    dist_args=self.dist_args)
        return ci

    def summary_frame(self, alpha=0.05):
        """Summary frame"""
        ci = self.conf_int(alpha=alpha)
        to_include = {}
        to_include['predicted'] = self.predicted
        to_include['se'] = self.se
        to_include['ci_lower'] = ci[:, 0]
        to_include['ci_upper'] = ci[:, 1]

        self.table = to_include
        # pandas dict does not handle 2d_array
        # data = np.column_stack(list(to_include.values()))
        # names = ....
        res = pd.DataFrame(to_include, index=self.row_labels,
                           columns=to_include.keys())
        return res


# this is based on PredictionResults, copied and adjusted
class PredictionResultsMonotonic(object):

    def __init__(self, predicted, var_pred, linpred=None, linpred_se=None,
                 func=None, deriv=None, df=None, dist=None, row_labels=None):
        # TODO: is var_resid used? drop from arguments?
        self.predicted = predicted
        self.var_pred = var_pred
        self.linpred = linpred
        self.linpred_se = linpred_se
        self.func = func
        self.deriv = deriv
        self.df = df
        self.row_labels = row_labels

        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    @property
    def se(self):
        return np.sqrt(self.var_pred)

    @property
    def tvalues(self):
        return self.predicted / self.se

    def t_test(self, value=0, alternative='two-sided'):
        '''z- or t-test for hypothesis that mean is equal to value

        Parameters
        ----------
        value : array_like
            value under the null hypothesis
        alternative : str
            'two-sided', 'larger', 'smaller'

        Returns
        -------
        stat : ndarray
            test statistic
        pvalue : ndarray
            p-value of the hypothesis test, the distribution is given by
            the attribute of the instance, specified in `__init__`. Default
            if not specified is the normal distribution.

        '''
        # assumes symmetric distribution
        stat = (self.predicted_mean - value) / self.se_mean

        if alternative in ['two-sided', '2-sided', '2s']:
            pvalue = self.dist.sf(np.abs(stat), *self.dist_args)*2
        elif alternative in ['larger', 'l']:
            pvalue = self.dist.sf(stat, *self.dist_args)
        elif alternative in ['smaller', 's']:
            pvalue = self.dist.cdf(stat, *self.dist_args)
        else:
            raise ValueError('invalid alternative')
        return stat, pvalue

    def _conf_int_generic(self, center, se, alpha, dist_args=None):
        """internal function to avoid code duplication
        """
        if dist_args is None:
            dist_args = ()

        q = self.dist.ppf(1 - alpha / 2., *dist_args)
        lower = center - q * se
        upper = center + q * se
        ci = np.column_stack((lower, upper))
        # if we want to stack at a new last axis, for lower.ndim > 1
        # np.concatenate((lower[..., None], upper[..., None]), axis=-1)
        return ci

    def conf_int(self, method='endpoint', alpha=0.05, **kwds):
        """
        Returns the confidence interval of the value, `effect` of the
        constraint.

        This is currently only available for t and z tests.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        kwds : extra keyword arguments
            currently ignored, only for compatibility, consistent signature

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """
        tmp = np.linspace(0, 1, 6)
        # TODO: drop check?
        is_linear = (self.func(tmp) == tmp).all()
        if method == 'endpoint' and not is_linear:
            ci_linear = self._conf_int_generic(self.linpred, self.linpred_se,
                                               alpha,
                                               dist_args=self.dist_args)
            ci = self.func(ci_linear)
        elif method == 'delta' or is_linear:
            ci = self._conf_int_generic(self.predicted, self.se, alpha,
                                        dist_args=self.dist_args)

        return ci

    def summary_frame(self, alpha=0.05):
        """Summary frame"""
        ci = self.conf_int(alpha=alpha)
        to_include = {}
        to_include['predicted'] = self.predicted
        to_include['se'] = self.se
        to_include['ci_lower'] = ci[:, 0]
        to_include['ci_upper'] = ci[:, 1]

        self.table = to_include
        # pandas dict does not handle 2d_array
        # data = np.column_stack(list(to_include.values()))
        # names = ....
        res = pd.DataFrame(to_include, index=self.row_labels,
                           columns=to_include.keys())
        return res


class PredictionResultsDelta(PredictionResultsBase):

    def __init__(self, results_delta, **kwds):

        predicted = results_delta.predicted()
        var_pred = results_delta.var()

        super().__init__(predicted, var_pred, **kwds)


# this is similar to ContrastResults after t_test, partially copied, adjusted
class PredictionResultsMean(object):

    def __init__(self, predicted_mean, var_pred_mean, var_resid=None,
                 df=None, dist=None, row_labels=None, linpred=None, link=None):
        # TODO: is var_resid used? drop from arguments?
        self.predicted_mean = predicted_mean
        self.var_pred_mean = var_pred_mean
        self.df = df
        self.var_resid = var_resid
        self.row_labels = row_labels
        self.linpred = linpred
        self.link = link

        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    @property
    def se_mean(self):
        return np.sqrt(self.var_pred_mean)

    @property
    def tvalues(self):
        return self.predicted_mean / self.se_mean

    def t_test(self, value=0, alternative='two-sided'):
        '''z- or t-test for hypothesis that mean is equal to value

        Parameters
        ----------
        value : array_like
            value under the null hypothesis
        alternative : str
            'two-sided', 'larger', 'smaller'

        Returns
        -------
        stat : ndarray
            test statistic
        pvalue : ndarray
            p-value of the hypothesis test, the distribution is given by
            the attribute of the instance, specified in `__init__`. Default
            if not specified is the normal distribution.

        '''
        # assumes symmetric distribution
        stat = (self.predicted_mean - value) / self.se_mean

        if alternative in ['two-sided', '2-sided', '2s']:
            pvalue = self.dist.sf(np.abs(stat), *self.dist_args)*2
        elif alternative in ['larger', 'l']:
            pvalue = self.dist.sf(stat, *self.dist_args)
        elif alternative in ['smaller', 's']:
            pvalue = self.dist.cdf(stat, *self.dist_args)
        else:
            raise ValueError('invalid alternative')
        return stat, pvalue

    def conf_int(self, method='endpoint', alpha=0.05, **kwds):
        """
        Returns the confidence interval of the value, `effect` of the
        constraint.

        This is currently only available for t and z tests.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        kwds : extra keyword arguments
            currently ignored, only for compatibility, consistent signature

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """
        tmp = np.linspace(0, 1, 6)
        is_linear = (self.link.inverse(tmp) == tmp).all()
        if method == 'endpoint' and not is_linear:
            ci_linear = self.linpred.conf_int(alpha=alpha, obs=False)
            ci = self.link.inverse(ci_linear)
        elif method == 'delta' or is_linear:
            se = self.se_mean
            q = self.dist.ppf(1 - alpha / 2., *self.dist_args)
            lower = self.predicted_mean - q * se
            upper = self.predicted_mean + q * se
            ci = np.column_stack((lower, upper))
            # if we want to stack at a new last axis, for lower.ndim > 1
            # np.concatenate((lower[..., None], upper[..., None]), axis=-1)

        return ci

    def summary_frame(self, alpha=0.05):
        """Summary frame"""
        # TODO: finish and cleanup
        ci_mean = self.conf_int(alpha=alpha)
        to_include = {}
        to_include['mean'] = self.predicted_mean
        to_include['mean_se'] = self.se_mean
        to_include['mean_ci_lower'] = ci_mean[:, 0]
        to_include['mean_ci_upper'] = ci_mean[:, 1]

        self.table = to_include
        # pandas dict does not handle 2d_array
        # data = np.column_stack(list(to_include.values()))
        # names = ....
        res = pd.DataFrame(to_include, index=self.row_labels,
                           columns=to_include.keys())
        return res


def _get_exog_predict(self, exog=None, transform=True, row_labels=None):
    """
    compute prediction results

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.


    Returns
    -------
    exog : ndarray
        Prediction exog
    row_labesls : list of str
        Labels or pandas index for rows of prediction
    """

    # prepare exog and row_labels, based on base Results.predict
    if transform and hasattr(self.model, 'formula') and exog is not None:
        from patsy import dmatrix
        if isinstance(exog, pd.Series):
            exog = pd.DataFrame(exog)
        exog = dmatrix(self.model.data.design_info, exog)

    if exog is not None:
        if row_labels is None:
            row_labels = getattr(exog, 'index', None)
            if callable(row_labels):
                row_labels = None

        exog = np.asarray(exog)
        if exog.ndim == 1 and (self.model.exog.ndim == 1 or
                               self.model.exog.shape[1] == 1):
            exog = exog[:, None]
        exog = np.atleast_2d(exog)  # needed in count model shape[1]
    else:
        exog = self.model.exog

        if row_labels is None:
            row_labels = getattr(self.model.data, 'row_labels', None)
    return exog, row_labels


def get_prediction_glm(self, exog=None, transform=True,
                       row_labels=None, linpred=None, link=None,
                       pred_kwds=None):
    """
    compute prediction results

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    *args :
        Some models can take additional arguments. See the
        predict method of the model for the details.
    **kwargs :
        Some models can take additional keyword arguments. See the
        predict method of the model for the details.

    Returns
    -------
    prediction_results : generalized_linear_model.PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction of the mean and of new observations.
    """

    # prepare exog and row_labels, based on base Results.predict
    exog, row_labels = _get_exog_predict(
        self,
        exog=exog,
        transform=transform,
        row_labels=row_labels,
        )

    if pred_kwds is None:
        pred_kwds = {}

    pred_kwds['linear'] = False
    predicted_mean = self.model.predict(self.params, exog, **pred_kwds)

    covb = self.cov_params()

    link_deriv = self.model.family.link.inverse_deriv(linpred.predicted_mean)
    var_pred_mean = link_deriv**2 * (exog * np.dot(covb, exog.T).T).sum(1)
    var_resid = self.scale  # self.mse_resid / weights

    # TODO: check that we have correct scale, Refactor scale #???
    # special case for now:
    if self.cov_type == 'fixed scale':
        var_resid = self.cov_kwds['scale']

    dist = ['norm', 't'][self.use_t]
    return PredictionResultsMean(
        predicted_mean, var_pred_mean, var_resid,
        df=self.df_resid, dist=dist,
        row_labels=row_labels, linpred=linpred, link=link)


def get_prediction_monotonic(self, exog=None, transform=True,
                             row_labels=None, link=None,
                             pred_kwds=None, index=None):
    """
    compute prediction results

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    *args :
        Some models can take additional arguments. See the
        predict method of the model for the details.
    **kwargs :
        Some models can take additional keyword arguments. See the
        predict method of the model for the details.

    Returns
    -------
    prediction_results : generalized_linear_model.PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction of the mean and of new observations.
    """

    # prepare exog and row_labels, based on base Results.predict
    exog, row_labels = _get_exog_predict(
        self,
        exog=exog,
        transform=transform,
        row_labels=row_labels,
        )

    if pred_kwds is None:
        pred_kwds = {}

    if link is None:
        link = self.model.family.link

    func_deriv = link.inverse_deriv

    # get linear prediction and standard errors
    covb = self.cov_params(column=index)
    linpred_var = (exog * np.dot(covb, exog.T).T).sum(1)
    pred_kwds_linear = pred_kwds.copy()
    pred_kwds_linear["which"] = "linear"
    linpred = self.model.predict(self.params, exog, **pred_kwds_linear)

    predicted = self.model.predict(self.params, exog, **pred_kwds)
    link_deriv = func_deriv(linpred)
    var_pred = link_deriv**2 * linpred_var

    dist = ['norm', 't'][self.use_t]
    res = PredictionResultsMonotonic(predicted, var_pred,
                                     df=self.df_resid, dist=dist,
                                     row_labels=row_labels, linpred=linpred,
                                     linpred_se=np.sqrt(linpred_var),
                                     func=link.inverse, deriv=func_deriv)
    return res


def get_prediction_delta(
        self,
        exog=None,
        which="mean",
        use_mean=False,
        transform=True,
        row_labels=None,
        pred_kwds=None
        ):
    """
    compute prediction results

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    which : str
        The statistic that is prediction. Which statistics are available
        depends on the model.predict method.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    *args :
        Some models can take additional arguments. See the
        predict method of the model for the details.
    **kwargs :
        Some models can take additional keyword arguments. See the
        predict method of the model for the details.

    Returns
    -------
    prediction_results : generalized_linear_model.PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction of the mean and of new observations.
    """

    # prepare exog and row_labels, based on base Results.predict
    exog, row_labels = _get_exog_predict(
        self,
        exog=exog,
        transform=transform,
        row_labels=row_labels,
        )

    def f_pred(p):
        """Prediction function as function of params
        """
        pred = self.model.predict(p, exog, which=which, **pred_kwds)
        if use_mean:
            pred = pred.mean(0)
        return pred

    nlpm = self._get_wald_nonlinear(f_pred)
    # TODO: currently returns NonlinearDeltaCov
    res = PredictionResultsDelta(nlpm)
    return res


def params_transform_univariate(params, cov_params, link=None, transform=None,
                                row_labels=None):
    """
    results for univariate, nonlinear, monotonicaly transformed parameters

    This provides transformed values, standard errors and confidence interval
    for transformations of parameters, for example in calculating rates with
    `exp(params)` in the case of Poisson or other models with exponential
    mean function.
    """

    from statsmodels.genmod.families import links
    if link is None and transform is None:
        link = links.Log()

    if row_labels is None and hasattr(params, 'index'):
        row_labels = params.index

    params = np.asarray(params)

    predicted_mean = link.inverse(params)
    link_deriv = link.inverse_deriv(params)
    var_pred_mean = link_deriv**2 * np.diag(cov_params)
    # TODO: do we want covariance also, or just var/se

    dist = stats.norm

    # TODO: need ci for linear prediction, method of `lin_pred
    linpred = PredictionResultsMean(
        params, np.diag(cov_params), dist=dist,
        row_labels=row_labels, link=links.identity())

    res = PredictionResultsMean(
        predicted_mean, var_pred_mean, dist=dist,
        row_labels=row_labels, linpred=linpred, link=link)

    return res
