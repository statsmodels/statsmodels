"""
Conditional logistic regression
"""

import numpy as np
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
import collections

class ConditionalLogit(base.LikelihoodModel):
    """
    Fit a conditional logistic regression model to grouped data.

    Every group is implicitly given an intercept, but the model is fit using
    a conditional likelihood in which the intercepts are not present.  Thus,
    intercept estimates are not given, but the other parameter estimates can
    be interpreted as being adjusted for any group-level confounders.
    """

    def __init__(self, endog, exog, groups, missing='none', **kwargs):

        super(ConditionalLogit, self).__init__(endog, exog, missing=missing, **kwargs)

        self.k_params = exog.shape[1]

        # Get the row indices for each group
        row_ix = collections.OrderedDict()
        for i, g in enumerate(groups):
            if g not in row_ix:
                row_ix[g] = []
            row_ix[g].append(i)

        # Split the data into groups
        endog, exog = np.asarray(endog), np.asarray(exog)
        self._endog_grp = []
        self._exog_grp = []
        self._groupsize = []
        for g, ix in row_ix.items():
            y = endog[ix].flat
            if np.std(y) == 0:
                continue
            self._endog_grp.append(y)
            self._groupsize.append(len(y))
            self._exog_grp.append(exog[ix, :])

        # Number of groups
        self._n_groups = len(self._endog_grp)

        # These are the sufficient statistics
        self._xy = []
        self._n1 = []
        for g in range(self._n_groups):
            self._xy.append(np.dot(self._endog_grp[g], self._exog_grp[g]))
            self._n1.append(np.sum(self._endog_grp[g]))

    def loglike(self, params):

        ll = 0
        for g in range(len(self._endog_grp)):
            ll += self.loglike_grp(g, params)

        return ll

    def score(self, params):

        score = 0
        for g in range(self._n_groups):
            score += self.score_grp(g, params)

        return score

    def _denom(self, grp, params):

        exb = np.exp(np.dot(self._exog_grp[grp], params))

        memo = {}

        def f(t, k):
            if t < k:
                return 0
            if k == 0:
                return 1

            try:
                return memo[(t, k)]
            except KeyError:
                pass

            v = f(t-1, k) + f(t-1, k-1) * exb[t-1]
            memo[(t, k)] = v

            return v

        return f(self._groupsize[grp], self._n1[grp])

    def _denom_grad(self, grp, params):

        exb = np.exp(np.dot(self._exog_grp[grp], params))

        memo = {}

        def s(t, k):

            if t < k:
                return 0, np.zeros(self.k_params)
            if k == 0:
                return 1, 0

            try:
                return memo[(t, k)]
            except KeyError:
                pass

            h = exb[t-1]
            a, b = s(t-1, k)
            c, e = s(t-1, k-1)
            d = c * h * self._exog_grp[grp][t-1, :]

            u, v = a + c * h, b + d + e * h
            memo[(t, k)] = (u, v)

            return u, v

        return s(self._groupsize[grp], self._n1[grp])

    def loglike_grp(self, grp, params):

        return np.dot(self._xy[grp], params) - np.log(self._denom(grp, params))

    def score_grp(self, grp, params):

        #d = self._denom(grp, params)
        d, h = self._denom_grad(grp, params)

        return self._xy[grp] - h / d

    def hessian(self, params):

        from statsmodels.tools.numdiff import approx_fprime
        hess = approx_fprime(params, self.score)
        hess = np.atleast_2d(hess)
        return hess

    def fit(self, start_params=None, maxiter=100, method='BFGS'):

        rslt = super(ConditionalLogit, self).fit(start_params=start_params, maxiter=maxiter,
                     method=method)

        crslt = ConditionalLogitResults(self, rslt.params, rslt.cov_params(), 1)
        crslt.method = method
        crslt.nobs = len(self.endog)
        crslt.n_groups = self._n_groups
        crslt._group_stats = ["%d" % min(self._groupsize),
                              "%d" % max(self._groupsize),
                              "%.1f" % np.mean(self._groupsize)]
        return ConditionalLogitResultsWrapper(crslt)


    # Override to allow groups and time to be passed as variable
    # names.
    @classmethod
    def from_formula(cls, formula, groups, data, *args, **kwargs):

        if type(groups) == str:
            groups = data[groups]

        model = super(ConditionalLogit, cls).from_formula(
                   formula, data=data, groups=groups, *args, **kwargs)

        return model


class ConditionalLogitResults(base.LikelihoodModelResults):


    def __init__(self, model, params, normalized_cov_params, scale):

        super(ConditionalLogitResults, self).__init__(
              model,
              params,
              normalized_cov_params=normalized_cov_params,
              scale=scale)


    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """
        Summarize the fitted model.

        Parameters
        -----------
        yname : string, optional
            Default is `y`
        xname : list of strings, optional
            Default is `var_##` for ## in p the number of regressors
        title : string, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results

        """

        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Log-Likelihood:', None),
                    ('Method:', [self.method]),
                    ('Date:', None),
                    ('Time:', None),
                    ]

        top_right = [('No. Observations:', None),
                     ('No. groups:', [self.n_groups]),
                     ('Min group size:', [self._group_stats[0]]),
                     ('Max group size:', [self._group_stats[1]]),
                     ('Mean group size:', [self._group_stats[2]]),
                     ]

        if title is None:
            title = "Conditional Logit Model Regression Results"

        # create summary tables
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,  # [],
                             yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha,
                              use_t=self.use_t)

        return smry


class ConditionalLogitResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(ConditionalLogitResultsWrapper, ConditionalLogitResults)
