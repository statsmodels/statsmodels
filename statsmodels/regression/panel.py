import scipy
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.tools import chain_dot
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import (RegressionModel,
                                                 RegressionResults)
from statsmodels.tools.grouputils import Grouping

class PanelLM(RegressionModel):
    '''Assumes the first level of the index is unit and the second it time'''
    def __init__(self, endog, exog, method='pooling', effects='oneway',
                 unit=None, time=None, hasconst=None, **kwargs):

        if type(exog) in [pd.core.frame.DataFrame, pd.core.series.Series]:
            self.groupings = Grouping(index_pandas=exog.index)
        elif type(endog) in [pd.core.frame.DataFrame, pd.core.series.Series]:
            self.groupings = Grouping(index_pandas=endog.index)
        else:
            self.groupings = Grouping(index_list=[unit, time])
        self.exog, idx = self.groupings.sort(exog)
        self.endog, idx = self.groupings.sort(endog)
        self.groupings.index = idx # relevant index order may have changed

        self.method = method
        self.effects = effects

        self.panel_balanced = True # TODO: no hard code True

        if method == 'swar':
            self.var_u, self.var_e, self.theta = swar_ercomp(self.endog, self.exog)

        super(PanelLM, self).__init__(endog, exog, **kwargs)

    def initialize(self, unit=None, time=None):
        self.wexog = self.whiten(self.exog)
        self.wendog = self.whiten(self.endog)
        self.nobs = float(self.wexog.shape[0])
        self.rank = np.rank(self.exog)
        self.df_model = float(self.rank - self.k_constant)
        if self.method == 'within':
            self.df_resid = self.nobs - self.rank - self.groupings.index.levshape[0]
        else:
            self.df_resid = self.nobs - self.rank - 1
        self.df_model = float(self.rank - self.k_constant)

    def whiten(self, data):
        g = self.groupings
        if self.method == 'within':
            f = lambda x: x - x.mean()
            if (self.effects == 'oneway') or (self.effects == 'unit'):
                out = g.transform_array(data, f, 0)
                return out
            elif (self.effects == 'time'):
                out = g.transform_array(data, f, 1)
                return out
            elif (self.effects == 'twoways'):
                out = g.transform_array(data, f, 0)
                out = g.transform_array(out, f, 1)
                return out
            else:
                raise Exception('Method must be unit, time, oneway, or twoways')
        elif self.method == 'between':
            f = lambda x: x.mean()
            if (self.effects == 'oneway') or (self.effects == 'unit'):
                out = g.transform_array(data, f, 0)
            elif (self.effects == 'time'):
                out = g.transform_array(data, f, 1)
            else:
                raise Exception('effects must be unit, time, or oneway')
            return out
        elif self.method == 'pooling':
            return data
        elif self.method == 'swar':
            out = g.transform_slices(array=data, function=swar_transform,
                                     theta=self.theta) 
            return out

    def fit(self, method="pinv", **kwargs):
        wexog = self.wexog
        self.pinv_wexog = pinv_wexog = np.linalg.pinv(wexog)
        self.normalized_cov_params = np.dot(pinv_wexog, pinv_wexog.T)
        beta = np.dot(self.pinv_wexog, self.wendog)
        lfit = PanelLMResults(self, beta,
                   normalized_cov_params=self.normalized_cov_params)
        return lfit

def swar_ercomp(y, X):
    '''Swamy-Arora error decomposition'''
    b = PanelLM(y, X, 'between').fit()
    w = PanelLM(y, X, 'within').fit()
    w.model.groupings.count_categories(level=0)
    Ts = w.model.groupings.counts   
    Th = scipy.stats.mstats.hmean(Ts)
    var_e = w.ssr / (X.shape[0] - w.model.groupings.index.levshape[0] - X.shape[1] + 1)
    var_u = b.ssr / (b.model.groupings.index.levshape[0] - X.shape[1]) - var_e / Th
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

    @cache_readonly
    def bse(self):
        if self.model.method == 'within':
            scale = self.nobs - self.model.groupings.index.levshape[0] - self.df_model
            scale = np.sum(self.wresid**2) / scale
            bse = np.sqrt(np.diag(self.cov_params(scale=scale)))
        else:
            bse = np.sqrt(np.diag(self.cov_params()))
        return bse

    @cache_readonly
    def ssr(self):
        return np.sum(self.wresid**2)

    @cache_readonly
    def resid(self):
        if (self.model.method == 'within') and not (self.model.panel_balanced):
            Xb_bar = np.dot(self.model.exog.mean(axis=0), self.params)
            alph = np.mean(self.model.endog) - Xb_bar
            pred = alph + np.dot(self.model.exog, self.params)
            resid = self.model.endog - pred
        else:
            resid = self.model.wendog - self.model.predict(self.params,
                                                           self.model.wexog)
        return resid

    def predict(self):
        return self.model.predict(self.params, self.model.wexog)

    @cache_readonly
    def fvalue(self):
        f = self.mse_model/self.mse_resid
        if self.model.method != 'within':
            f = f / 2.
        return f

    @cache_readonly
    def scale(self):
        wresid = self.wresid
        return np.dot(wresid.T, wresid) / self.df_resid

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


'''
from statsmodels.iolib.summary import summary_col
from patsy import dmatrices
def test(y, X):
    mod1 = PanelLM(y, X, method='pooling').fit()
    mod2 = PanelLM(y, X, method='between').fit()
    mod3 = PanelLM(y, X.ix[:,1:], method='within').fit()
    mod4 = PanelLM(y, X.ix[:,1:], method='within', effects='time').fit()
    mod5 = PanelLM(y, X.ix[:,1:], method='within', effects='twoways').fit()
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
dat = read.csv('EmplUK.csv')
dat = pdata.frame(dat, c('firm', 'year'))
mod = plm(emp ~ wage + capital, data=dat, model='within')
'''
