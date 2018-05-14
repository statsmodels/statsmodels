# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:29:18 2014

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import stats

# this is similar to ContrastResults after t_test, partially copied and adjusted
class PredictionResults(object):

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
    def se_obs(self):
        raise NotImplementedError
        return np.sqrt(self.var_pred_mean + self.var_resid)

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
        alternative : string
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
        # from statsmodels.stats.weightstats

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
        Returns the confidence interval of the value, `effect` of the constraint.

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


    def summary_frame(self, what='all', alpha=0.05):
        # TODO: finish and cleanup
        import pandas as pd
        from statsmodels.compat.collections import OrderedDict
        #ci_obs = self.conf_int(alpha=alpha, obs=True) # need to split
        ci_mean = self.conf_int(alpha=alpha)
        to_include = OrderedDict()
        to_include['mean'] = self.predicted_mean
        to_include['mean_se'] = self.se_mean
        to_include['mean_ci_lower'] = ci_mean[:, 0]
        to_include['mean_ci_upper'] = ci_mean[:, 1]


        self.table = to_include
        #OrderedDict doesn't work to preserve sequence
        # pandas dict doesn't handle 2d_array
        #data = np.column_stack(list(to_include.values()))
        #names = ....
        res = pd.DataFrame(to_include, index=self.row_labels,
                           columns=to_include.keys())
        return res


def get_prediction_glm(self, exog=None, transform=True, weights=None,
                   row_labels=None, linpred=None, link=None, pred_kwds=None):
    """
    compute prediction results

    Parameters
    ----------
    exog : array-like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    weights : array_like, optional
        Weights interpreted as in WLS, used for the variance of the predicted
        residual.
    args, kwargs :
        Some models can take additional arguments or keywords, see the
        predict method of the model for the details.

    Returns
    -------
    prediction_results : generalized_linear_model.PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction of the mean and of new observations.

    """

    ### prepare exog and row_labels, based on base Results.predict
    if transform and hasattr(self.model, 'formula') and exog is not None:
        from patsy import dmatrix
        exog = dmatrix(self.model.data.design_info.builder,
                       exog)

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
        if weights is None:
            weights = getattr(self.model, 'weights', None)

        if row_labels is None:
            row_labels = getattr(self.model.data, 'row_labels', None)

    # need to handle other arrays, TODO: is delegating to model possible ?
    if weights is not None:
        weights = np.asarray(weights)
        if (weights.size > 1 and
           (weights.ndim != 1 or weights.shape[0] == exog.shape[1])):
            raise ValueError('weights has wrong shape')

    ### end

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

    if weights is not None:
        var_resid /= weights

    dist = ['norm', 't'][self.use_t]
    return PredictionResults(predicted_mean, var_pred_mean, var_resid,
                             df=self.df_resid, dist=dist,
                             row_labels=row_labels, linpred=linpred, link=link)


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
    linpred = PredictionResults(params, np.diag(cov_params), dist=dist,
                             row_labels=row_labels, link=links.identity())

    res = PredictionResults(predicted_mean, var_pred_mean, dist=dist,
                             row_labels=row_labels, linpred=linpred, link=link)

    return res
