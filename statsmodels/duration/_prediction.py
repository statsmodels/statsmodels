"""
This is a bare-bones version of _prediction from regression for
duration
"""

import numpy as np
from scipy import stats

# this is similar to ContrastResults after t_test, partially copied and adjusted
class PredictionResults(object):

    def __init__(self, predicted_mean, var_pred_mean,
                 df=None, dist=None, row_labels=None):
        self.predicted_mean = predicted_mean
        self.var_pred_mean = var_pred_mean
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
    def se_mean(self):
        return np.sqrt(self.var_pred_mean)

    def conf_int(self, obs=False, alpha=0.05):
        """
        Returns the confidence interval of the value, `effect` of the constraint.

        This is currently only available for t and z tests.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.

        """

        se = self.se_mean

        q = self.dist.ppf(1 - alpha / 2., *self.dist_args)
        lower = self.predicted_mean - q * se
        upper = self.predicted_mean + q * se
        return np.column_stack((lower, upper))


    def summary_frame(self, what='all', alpha=0.05):
        # TODO: finish and cleanup
        import pandas as pd
        from statsmodels.compat.collections import OrderedDict
        ci_obs = self.conf_int(alpha=alpha, obs=True) # need to split
        ci_mean = self.conf_int(alpha=alpha, obs=False)
        to_include = OrderedDict()
        to_include['mean'] = self.predicted_mean
        to_include['mean_se'] = self.se_mean
        to_include['mean_ci_lower'] = ci_mean[:, 0]
        to_include['mean_ci_upper'] = ci_mean[:, 1]
        to_include['obs_ci_lower'] = ci_obs[:, 0]
        to_include['obs_ci_upper'] = ci_obs[:, 1]

        self.table = to_include
        #OrderedDict doesn't work to preserve sequence
        # pandas dict doesn't handle 2d_array
        #data = np.column_stack(list(to_include.values()))
        #names = ....
        res = pd.DataFrame(to_include, index=self.row_labels,
                           columns=to_include.keys())
        return res


def get_prediction(self, exog=None, transform=True, row_labels=None,
                   cov_params=None, endog=None, strata=None,
                   offset=None, pred_type=None, pred_kwds=None):
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
    args, kwargs :
        Some models can take additional arguments or keywords, see the
        predict method of the model for the details.

    Returns
    -------
    prediction_results : instance
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
            if hasattr(exog, 'index'):
                row_labels = exog.index
            else:
                row_labels = None

        exog = np.asarray(exog)
        if exog.ndim == 1 and (self.model.exog.ndim == 1 or
                               self.model.exog.shape[1] == 1):
            exog = exog[:, None]
        exog = np.atleast_2d(exog)  # needed in count model shape[1]
    else:
        exog = self.model.exog

    if endog is None:
        endog = self.model.endog

        if row_labels is None:
            row_labels = getattr(self.model.data, 'row_labels', None)

    ### end

    if pred_kwds is None:
        pred_kwds = {}
    predicted_mean = self.model.predict(self.params, exog=exog,
                                        cov_params=cov_params, endog=endog,
                                        strata=strata, offset=offset,
                                        pred_type=pred_type, **pred_kwds)

    if pred_type == "lhr":
        # TODO fix the handling of this
        if cov_params is None:
            cov_params = self.cov_params()
        mat = np.dot(exog, cov_params)
        var_pred_mean = (mat * exog).sum(1)
    
    else:
        msg = "Type %s does not support get_prediction" % pred_type
        raise ValueError(msg)

    dist = ['norm', 't'][self.use_t]
    return PredictionResults(predicted_mean, var_pred_mean,
                             df=self.df_resid, dist=dist,
                             row_labels=row_labels)
