import scipy
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.tools import chain_dot
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import (RegressionModel,
                                                 RegressionResults)
from statsmodels.tools.grouputils import Grouping
from statsmodels.panel.panel_model import PanelModel

def _check_method_compat(method, effects):
    if method not in ['within', 'pooling', 'between', 'swar', 'mle', 'random']:
        raise ValueError("method %s not understood" % method)
    if effects not in ['oneway', 'unit', 'time', 'twoway']:
        raise ValueError("effects %s not understood" % effects)
    if method == 'within' and not effects in ['oneway', 'twoway', 'time']:
        raise ValueError("effects for within must be oneway, twoway, or time")
    elif method == 'between' and not effects in ['oneway', 'time']:
        raise ValueError("effects for between must be oneway or time")
    #Need to do any checking for other methods?


class PanelLM(PanelModel, RegressionModel):
    r'''
    Panel Data Linear Regression Model

    Parameters
    ----------
    y : array-like
        The endogenous variable. See Notes.
    X : array-like
        The exogenous variables. See Notes.
    panel : array-like, optional
        If `y` and `X` are array-like, then `panel` must be specified and must
        be of the same length as them.
    time : array-like, optional
        If `y` and `X` are array-like, then `time` must be specified and must
        be of the same length as them.
    effects : str
        The type of model to be estimated.:

        * oneway
        * twoway
        * time
    method : str
        The type of model to be estimated.:

        * pooling
        * within - Also known as the fixed-effects estimator. This uses OLS.
        * between - Between effects model
        * swar - The small sample Swamy-Arora estimator of individual-level
          variance components should be used.
        * random - GLS random-effects model
        * mle - Maximum Likelihood random-effects model
    %(extra_parameters)s

    Notes
    -----
    If a Series or DataFrame is given for both `y` and `X` it is assumed that
    the indices of both are equivalent MultiIndex. The first level must be the
    panel index and the second unit must be the time index.

    These are the different models assumed. The `effects` keyword will affect
    the subscripts, but the same models hold.

    Within or fixed-effects model

    .. math:

       (y_{it} - \bar{y}_i) = (X_{it}-\bar{X}_i)\beta + (\epsilon_{it} - \bar{\epsilon}_i)

    Between-effects model

    .. math:

       \bar{y}_i = \alpha + \bar{x}_i\beta + \upsilon_i + \bar{\epsilon}_i

    The random-effects models is weighted average of the two.

    .. math:

        (y_{it} - \theta\bar{y}_i) = (1-\theta)\alpha + (X_{it} - \theta \bar{X}_{i})\beta + [(1-\theta)\upsilon_i + (\epsilon_{it} - \theta \bar{\epsilon}_i)]

    where :math:`\theta` is a function of :math:`\sigma_{\upsilon}^2` and
    :math:`\sigma_{\epsilon}^2`

    '''
    #TODO: make sure hasconst works
    def __init__(self, y, X, method='pooling', effects='oneway', panel=None,
                 time=None, hasconst=None, missing='none'):
        _check_method_compat(method, effects)
        self.method = method
        self.effects = effects

        super(PanelLM, self).__init__(y, X, missing=missing, time=time,
                                      panel=panel, hasconst=hasconst)


    def initialize(self, unit=None, time=None):
        self.wexog = self.whiten(self.exog)
        self.wendog = self.whiten(self.endog)
        self.nobs = float(self.wexog.shape[0])
        self.rank = np.linalg.matrix_rank(self.exog)
        #NOTE: These should also depend on effects - needs to be a function
        if self.method == 'within':
            # -1 because n_panel is full rank
            self.df_model = float(self.rank + self.data.n_panel - 1)
            # N(T-1) - K
            # -K doesn't matter asymptotically (for calc of sigma_u)
            self.df_resid = self.nobs - self.df_model - 1
        else:
            self.df_model = float(self.rank - self.k_constant)
            self.df_resid = self.nobs - self.df_model - self.k_constant

    def whiten(self, data):
        g = self.data.groupings
        method = self.method
        if method == 'pooling':
            return data
        elif method == 'swar':
            # do this here so endog and exog have been through data handling
            idx = g.index
            panel, time = (idx.get_level_values(0), idx.get_level_values(1))
            self.var_u, self.var_e, self.theta = swar_ercomp(self.endog,
                                                             self.exog,
                                                             panel,
                                                             time)
            out = g.transform_slices(array=data, function=swar_transform,
                                     theta=self.theta)
            return out

        elif method == 'within':
            func = lambda x : x - x.mean()
        elif method == "between":
            func = lambda x : x.mean()

        effects = self.effects
        if effects in ['oneway', 'unit']: #document/keep unit?
            levels = [0]
        elif effects == 'time':
            levels = [1]
        elif effects in ['twoway']:
            levels = [0, 1]

        for level in levels: #TODO: this should copy but be sure
            data = g.transform_array(data, func, level)
        return data


    def fit(self, method="pinv", **kwargs):
        wexog = self.wexog
        self.pinv_wexog = pinv_wexog = np.linalg.pinv(wexog)
        normalized_cov_params = np.dot(pinv_wexog, pinv_wexog.T)
        beta = np.dot(self.pinv_wexog, self.wendog)
        if self.method == "within":
            return PanelLMWithinResults(self, beta,
                    normalized_cov_params=normalized_cov_params)
        elif self.method == "between":
            return PanelLMBetweenResults(self, beta,
                    normalized_cov_params=normalized_cov_params)
        else:
            return PanelLMResults(self, beta,
                    normalized_cov_params=normalized_cov_params)

#TODO: hook this into the data handling so panel and time are optional
#      but for now it's used internally on plain arrays, so needs these
def swar_ercomp(y, X, panel, time):
    '''Swamy-Arora error decomposition'''
    b = PanelLM(y, X, panel=panel, time=time, method='between').fit()
    w = PanelLM(y, X, panel=panel, time=time, method='within').fit()
    w.model.data.groupings.count_categories(level=0)
    Ts = w.model.data.groupings.counts
    Th = scipy.stats.mstats.hmean(Ts)
    var_e = w.ssr / (X.shape[0] - w.model.data.n_panel - X.shape[1] + 1)
    var_u = b.ssr / (b.model.data.n_panel - X.shape[1]) - var_e / Th
    var_u = max(var_u, 0)
    Ts = np.concatenate([np.repeat(x,x) for x in Ts])
    theta = 1 - np.sqrt(var_e / (Ts * var_u + var_e))
    return var_e, var_u, np.array(theta)

def swar_transform(subset, position, theta):
    '''Apply to a sub-group of observations'''
    n = subset.shape[0]
    B = np.ones((n,n)) / n
    out = subset - chain_dot(np.diag(theta[position]), B, subset)
    return out

class PanelLMResults(RegressionResults):
    def __init__(self, model, params, normalized_cov_params=None):
        super(PanelLMResults, self).__init__(model, params)
        self.normalized_cov_params = normalized_cov_params

        # overwrite scale set in RegressionResults. This smells.
        # might want to define other things than scale and leave it as 1.
        # or just deprecate scale
        wresid = self.wresid
        self.scale = np.dot(wresid.T, wresid) / self.df_resid

    @cache_readonly
    def ssr(self):
        return np.sum(self.wresid**2)

    def conf_int(self, alpha=.05, cols=None):
        """
        Returns the confidence interval of the fitted parameters.

        Parameters
        ----------
        alpha : float, optional
            The `alpha` level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.
        cols : array-like, optional
            `cols` specifies which confidence intervals to return

        Notes
        -----
        The confidence interval is based on Student's t-distribution.
        """
        from scipy import stats

        bse = self.bse
        params = self.params
        if self.model.method == "swar":
            dist = stats.norm
            q = dist.ppf(1 - alpha / 2)
        else:
            dist = stats.t
            q = dist.ppf(1 - alpha / 2, self.df_resid)

        if cols is None:
            lower = self.params - q * bse
            upper = self.params + q * bse
        else:
            cols = np.asarray(cols)
            lower = params[cols] - q * bse[cols]
            upper = params[cols] + q * bse[cols]
        return np.asarray(zip(lower, upper))

    @cache_readonly
    def rsquared_overall(self):
        return np.corrcoef(self.fittedvalues, self.model.endog)[0,1] ** 2

    @cache_readonly
    def resid(self):
        model = self.model
        if (model.method == 'within') and not (model.data.is_balanced):
            Xb_bar = np.dot(model.exog.mean(axis=0), self.params)
            alph = np.mean(model.endog) - Xb_bar
            pred = alph + np.dot(model.exog, self.params)
            resid = model.endog - pred
        else:
            resid = model.wendog - model.predict(self.params, model.wexog)
        return resid

    def predict(self):
        return self.model.predict(self.params, self.model.wexog)

    @cache_readonly
    def fvalue(self):
        if self.model.method == 'within':
            #TODO: not sure where the 2 degrees of freedom comes from
            #      check literature
            f = self.ess / self.ssr * self.df_resid / 2.
        else:
            f = self.mse_model/self.mse_resid
        return f

class PanelLMWithinResults(PanelLMResults):
    @cache_readonly
    def _fixed_effects(self): # make public?
        model = self.model
        grouped_y = model.data.groupings.transform_array(model.endog,
                                                lambda x : x.mean(), 0)
        grouped_X = model.data.groupings.transform_array(model.exog,
                                                lambda x : x.mean(), 0)
        #TODO: Make fixed_effects a property?
        return grouped_y - model.predict(self.params, grouped_X)

    @cache_readonly
    def constant(self): # attach to params instead - handle up in fit?
        return self._fixed_effects.mean()

    #NOTE: this makes fittedvalues different from predict()
    @cache_readonly
    def fittedvalues(self): # defined as deviations from the mean fixed effect
        model = self.model
        return model.predict(self.params, model.exog) + self.constant

    #TODO: better names for std_devs?
    @cache_readonly
    def std_dev_resid(self): # e_it the overall error
        return self.scale ** .5

    @cache_readonly
    def std_dev_groups(self):
        # sd. of res. within groups
        return self._fixed_effects.std(ddof=1)

    @cache_readonly
    def resid_groups(self): # u_i, the fixed-error component
        # easier for unbalanced case to use pandas alignment here.
        # this should maybe be a method for Grouping class though.
        # repeat maybe
        import pandas as pd
        #TODO: update for time effects
        data = self.model.data
        idx = data.groupings.index.get_level_values(0)
        idx_uniq = idx.unique()
        resid = pd.DataFrame(self._fixed_effects - self.constant,
                             index=idx_uniq)
        return resid.reindex(idx).values.squeeze()

    @cache_readonly
    def resid_combined(self):
        return self.resid_groups + self.resid

    @cache_readonly
    def corr(self): # the correlation of u_i with XB_{it}
        return np.corrcoef(self.resid_groups, self.fittedvalues)[0,1]

    @cache_readonly
    def rho(self):
        # fraction of variance due to differences across panels
        # aka intraclass correlation
        return self.std_dev_groups ** 2 / (self.std_dev_groups**2 +
                                           self.std_dev_resid**2)

    @cache_readonly
    def fitted_with_effects(self):
        return self.fittedvalues + self.resid_groups

    @cache_readonly
    def std_dev_overall(self):
        return (self.std_dev_groups**2 + self.std_dev_resid**2)**.5

    @cache_readonly
    def rsquared_within(self):
        return self.rsquared

    @cache_readonly
    def rsquared_between(self):
        model = self.model
        grouped_y = model.data.groupings.transform_array(model.endog,
                                                lambda x : x.mean(), 0)
        grouped_X = model.data.groupings.transform_array(model.exog,
                                                lambda x : x.mean(), 0)
        grouped_y_hat = model.predict(self.params, grouped_X)
        return np.corrcoef(grouped_y, grouped_y_hat)[0,1] ** 2

    @cache_readonly
    def rsquared_adj(self): # this is not a good measure of fit.
        r2 = self.rsquared  # prefer F-test / know what you're doing.
        nobs = self.nobs
        return 1 - (1 - r2) * (nobs - 1.)/(self.df_resid)

    @cache_readonly
    def centered_tss(self): #TODO: centered tss of _original_ y_{it} right?
        return self.model.endog.var() * self.nobs

    #NOTE: r2 depends on this, so it might be why it's off
    #@cache_readonly
    #def uncentered_tss(self):
    #    # better safe than wrong
    #    raise NotImplementedError


class PanelLMBetweenResults(PanelLMResults):
    @cache_readonly
    def rsquared_between(self):
        return self.rsquared

    @cache_readonly
    def rsquared_within(self):
        model = self.model
        within_y = model.data.groupings.transform_array(model.endog,
                                                lambda x : x - x.mean(), 0)
        within_X = model.data.groupings.transform_array(model.exog,
                                                lambda x : x - x.mean(), 0)

        within_y_hat = model.predict(self.params, within_X)
        return np.corrcoef(within_y, within_y_hat)[0,1] ** 2

    @cache_readonly
    def rmse(self): #TODO: rename this. std_dev_overall? s.d.(u_i + avg(e_i))
        return self.scale ** .5

    @cache_readonly
    def resid_overall(self):
        model = self.model
        within_y = model.data.groupings.transform_array(model.endog,
                                                lambda x : x - x.mean(), 0)
        within_X = model.data.groupings.transform_array(model.exog,
                                                lambda x : x - x.mean(), 0)
        within_y_hat = model.predict(self.params, within_X)
        return within_y - within_y_hat

    @cache_readonly
    def resid_groups(self):
        return self.resid

    @cache_readonly
    def resid_combined(self):
        data = self.model.data
        idx = data.groupings.index.get_level_values(0)
        idx_uniq = idx.unique()
        resid_groups = pd.DataFrame(self.resid_groups, index=idx_uniq)
        return resid_groups.reindex(idx).values.squeeze() + self.resid_overall

def pooltest(endog, exog):
    '''Chow poolability test: F-test of joint significance for the unit dummies
    in a LSDV model

    Returns
    -------

    F statistic for the null hypothesis that unit effects are zero.
    p value
    '''
    # TODO: Does this assume balanced panels?
    unrestricted = PanelLM(endog, exog, 'within').fit()
    restricted = PanelLM(endog, exog, 'pooling').fit()
    N = unrestricted.model.panel_N
    T = unrestricted.model.panel_T
    K = unrestricted.model.exog.shape[1]
    urss = unrestricted.ssr
    rrss = restricted.ssr
    F = ((rrss - urss) / (N-1)) / (urss / (N*T - N - K))
    p = 1 - scipy.stats.distributions.f.cdf(F, N-1, N*(T-1)-K)
    return F, p


if __name__ == "__main__":
    import statsmodels.api as sm
    dta = sm.datasets.get_rdataset("EmplUK", "plm").data
    dta.set_index(['firm', 'year'], inplace=True)
    y = dta["emp"]
    dta["const"] = 1
    X = dta[["const", "wage", "capital"]]
    mod1 = PanelLM(y, X, method='pooling').fit()
    mod2 = PanelLM(y, X, method='between').fit()
    mod3 = PanelLM(y, X.ix[:,1:], method='within').fit()
    mod4 = PanelLM(y, X.ix[:,1:], method='within', effects='time').fit()
    mod5 = PanelLM(y, X.ix[:,1:], method='within', effects='twoway').fit()
    mod6 = PanelLM(y, X, method='swar').fit()

    from patsy import dmatrices
    from panel import PanelLM
    from statsmodels.datasets import grunfeld
    data = grunfeld.load_pandas().data
    data.firm = data.firm.apply(lambda x: x.lower())
    data = data.set_index(['firm', 'year'])
    data = data.sort()
    y, X = dmatrices("invest ~ value + capital - 1", data=data,
        return_type='dataframe')

    within = PanelLM(y, X, method='within').fit(disp=0)

    y, X = dmatrices("invest ~ value + capital", data=data,
        return_type='dataframe')
    between = PanelLM(y, X, method='between').fit(disp=0)
    swar = PanelLM(y, X, method="swar").fit()
    pooling = PanelLM(y, X, method="pooling").fit()



'''
from statsmodels.iolib.summary import summary_col
from patsy import dmatrices
def test(y, X):
    mod1 = PanelLM(y, X, method='pooling').fit()
    mod2 = PanelLM(y, X, method='between').fit()
    mod3 = PanelLM(y, X.ix[:,1:], method='within').fit()
    mod4 = PanelLM(y, X.ix[:,1:], method='within', effects='time').fit()
    mod5 = PanelLM(y, X.ix[:,1:], method='within', effects='twoway').fit()
    mod6 = PanelLM(y, X, 'swar').fit()
    mn = ['OLS', 'Between', 'Within N', 'Within T', 'Within 2w', 'RE-SWAR']
    out = summary_col([mod1, mod2, mod3, mod4, mod5, mod6], model_names=mn, stars=False)
    return out
url = 'http://vincentarelbundock.github.com/Rdatasets/csv/plm/EmplUK.csv'
url = 'EmplUK.csv'
gasoline = pd.read_csv(url).set_index(['firm', 'year'])
f = 'emp~wage+capital'
y, X = dmatrices(f, gasoline, return_type='dataframe')

# Statsmodels

================================================================
            OLS    Between  Within N Within T Within 2w RE-SWAR
----------------------------------------------------------------
Intercept  10.3598  10.5060                               9.0390
          (1.2023) (3.3055)                             (1.1015)
capital     2.1049   2.2673   0.8015   2.0999    0.7854   1.1564
          (0.0441) (0.1136) (0.0641) (0.0470)  (0.0622) (0.0592)
wage       -0.3238  -0.3371  -0.1436  -0.3334   -0.0955  -0.1589
          (0.0487) (0.1338) (0.0328) (0.0527)  (0.0356) (0.0338)
----------------------------------------------------------------
N             1031      140     1031     1031      1031     1031
R2           0.693    0.748    0.164    0.695     0.155    0.283
================================================================
Standard errors in parentheses.

# Stata replication

wget http://vincentarelbundock.github.com/Rdatasets/csv/plm/EmplUK.csv

insheet using EmplUK.csv, clear
xtset firm year
xtreg emp wage capital, fe

# R replication

library(plm)
data(EmplUK)
dat = EmplUK
dat = pdata.frame(dat, c('firm', 'year'))
mod = plm(emp ~ wage + capital, data=dat, model='within')

'''
