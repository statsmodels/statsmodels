# -*- coding: utf-8 -*-
"""

Created on Sun Jan 29 11:16:09 2012

Author: Josef Perktold
License: BSD-3
"""

from collections import defaultdict
import numpy as np

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly


#influence measures

def reset_ramsey(res, degree=5):
    '''Ramsey's RESET specification test for linear models

    This is a general specification test, for additional non-linear effects
    in a model.


    Notes
    -----
    The test fits an auxilliary OLS regression where the design matrix, exog,
    is augmented by powers 2 to degree of the fitted values. Then it performs
    an F-test whether these additional terms are significant.

    If the p-value of the f-test is below a threshold, e.g. 0.1, then this
    indicates that there might be additional non-linear effects in the model
    and that the linear model is mis-specified.


    References
    ----------
    http://en.wikipedia.org/wiki/Ramsey_RESET_test

    '''
    order = degree + 1
    k_vars = res.model.exog.shape[1]
    #vander without constant and x:
    y_fitted_vander = np.vander(res.fittedvalues, order)[:, :-2] #drop constant
    exog = np.column_stack((res.model.exog, y_fitted_vander))
    res_aux = OLS(res.model.endog, exog).fit()
    #r_matrix = np.eye(degree, exog.shape[1], k_vars)
    r_matrix = np.eye(degree-1, exog.shape[1], k_vars)
    #df1 = degree - 1
    #df2 = exog.shape[0] - degree - res.df_model  (without constant)
    return res_aux.f_test(r_matrix) #, r_matrix, res_aux



def variance_inflation_factor(exog, exog_idx):
    '''variance inflation factor, VIF, for one exogenous variable

    The variance inflation factor is a measure for the increase of the
    variance of the parameter estimates if an additional variable, given by
    exog_idx is added to the linear regression. It is a measure for
    multicollinearity of the design matrix, exog.

    One recommendation is that if VIF is greater than 5, then the explanatory
    variable given by exog_idx is highly collinear with the other explanatory
    variables, and the parameter estimates will have large standard errors
    because of this.

    Parameters
    ----------
    exog : ndarray, (nobs, k_vars)
        design matrix with all explanatory variables, as for example used in
        regression
    exog_idx : int
        index of the exogenous variable in the columns of exog

    Returns
    -------
    vif : float
        variance inflation factor

    Notes
    -----
    This function does not save the auxilliary regression. If we are
    interested also in other diagnostic measures, then using xxx will avoid
    repeated calculations

    See Also
    --------
    xxx : class for regression diagnostics  TODO: doesn't exist yet

    References
    ----------
    http://en.wikipedia.org/wiki/Variance_inflation_factor

    '''
    k_vars = exog.shape[1]
    x_i = exog[:, exog_idx]
    mask = np.arange(k_vars) != exog_idx
    x_noti = exog[:, mask]
    r_squared_i = OLS(x_i, x_noti).fit().rsquared
    vif = 1. / (1. - r_squared_i)
    return vif


class Influence(object):
    '''class to calculate outlier and influence measures for OLS result
    '''

    def __init__(self, results):
        #check which model is allowed
        self.results = results
        self.nobs, self.k_vars = results.model.exog.shape
        self.endog = results.model.endog
        self.exog = results.model.exog
        self.model_class = results.model.__class__

        self.sigma_est = np.sqrt(results.mse_resid)

        self.aux_regression_exog = {}
        self.aux_regression_endog = {}

    @cache_readonly
    def hat_matrix_diag(self):
        '''OLS only currently
        #temporary, should go to model class
        '''
        return (self.exog * self.results.model.pinv_wexog.T).sum(1)

    @cache_readonly
    def resid_press(self):
        '''
        '''
        hii = self.hat_matrix_diag
        return self.results.resid / (1 - hii)

    @cache_readonly
    def influence(self):
        '''
        matches the influence measure that gretl repors
        u * h / (1 - h)
        '''
        hii = self.hat_matrix_diag
        return self.results.resid * hii / (1 - hii)

    @cache_readonly
    def hat_diag_factor(self):
        '''
        this might be useful for internal reus
        h / (1 - h)
        '''
        hii = self.hat_matrix_diag
        return hii / (1 - hii)

    @cache_readonly
    def ess_press(self):
        '''
        '''
        return np.dot(self.resid_press, self.resid_press)

    @cache_readonly
    def resid_studentized_internal(self):
        '''this uses sigma from original estimate

        does not require leave one out loop
        '''
        return self.get_resid_studentized_external(sigma=None)
        #return self.results.resid / self.sigma_est

    @cache_readonly
    def resid_studentized_external(self):
        '''this uses sigma from leave-one-out estimates

        requires leave one out loop for observations
        '''
        #call self.summary to get all loo obs attributes
        self.get_all_obs()
        sigma_looo = np.sqrt(self.sigma2_not_obsi)
        return self.get_resid_studentized_external(sigma=sigma_looo)

    def get_resid_studentized_external(self, sigma=None):
        '''method
        '''
        hii = self.hat_matrix_diag
        if sigma is None:
            sigma2_est = self.results.mse_resid
            #can be replace by different estimators of sigma
            sigma = np.sqrt(sigma2_est)

        return  self.results.resid / sigma / np.sqrt(1 - hii)

    @cache_readonly
    def dffits_internal(self):
        #TODO: do I want to use different sigma estimate in
        #      resid_studentized_external
        # -> move definition of sigma_error to the __init__
        hii = self.hat_matrix_diag
        dffits_ = self.resid_studentized_internal * np.sqrt(hii / (1 - hii))
        dffits_threshold = 2 * np.sqrt(self.k_vars * 1. / self.nobs)
        return dffits_, dffits_threshold

    @cache_readonly
    def dffits(self):
        #TODO: do I want to use different sigma estimate in
        #      resid_studentized_external
        # -> move definition of sigma_error to the __init__
        hii = self.hat_matrix_diag
        dffits_ = self.resid_studentized_external * np.sqrt(hii / (1 - hii))
        dffits_threshold = 2 * np.sqrt(self.k_vars * 1. / self.nobs)
        return dffits_, dffits_threshold

    def cooks_distance(self):
        hii = self.hat_matrix_diag
        #Eubank p.93, 94
        cooks_d2 = self.resid_studentized_internal**2 / self.k_vars
        cooks_d2 *= hii / (1 - hii)

        from scipy import stats
        #alpha = 0.1
        #print stats.f.isf(1-alpha, n_params, res.df_modelwc)
        pvals = stats.f.sf(cooks_d2, self.k_vars, self.results.df_resid)

        return cooks_d2, pvals

    @cache_readonly
    def resid_var(self):
        #TODO:check if correct outside of ols
        return self.results.mse_resid * (1 - self.hat_matrix_diag)

    @cache_readonly
    def resid_std(self):
        return np.sqrt(self.resid_var)


    def ols_xnoti(self, drop_idx, endog_idx='endog', store=True):
        '''

        there are too many combinations to store them all, except for small
        problems

        this needs more thought, memory versus speed

        reverse the structure, access store, if fail calculate ?
        '''
        #this creates keys in store even if store = false ! bug
        if endog_idx == 'endog':
            stored = self.aux_regression_endog
            if hasattr(stored, drop_idx):
                return stored[drop_idx]
            x_i = self.results.model.endog

        else:
            #nested dictionary
            try:
                self.aux_regression_exog[endog_idx][drop_idx]
            except KeyError:
                pass

            stored = self.aux_regression_exog[endog_idx]
            stored = {}

            x_i = self.exog[:, endog_idx]

        mask = np.arange(k_vars) != drop_idx
        x_noti = self.exog[:, mask]
        res = OLS(x_i, x_noti).fit()
        if store:
            stored[drop_idx] = res

        return res

    def get_drop_var(self, attributes):
        '''regress endog on exog without one of the variables

        this uses a k_vars loop

        '''
        from statsmodels.sandbox.tools.cross_val import LeaveOneOut

        endog = self.results.model.endog
        exog = self.exog

        cv_iter = LeaveOneOut(self.k_vars)
        res_loo = defaultdict(list)
        for inidx, outidx in cv_iter:
            for att in attributes:
                res_i = self.model_class(endog, exog[:,inidx]).fit()
            res_loo[att].append(getattr(res_i, att))

        return res_loo

    def get_drop_obs(self, attributes):
        '''regress endog on exog dropping one observation at a time

        this uses a nobs loop
        '''
        from statsmodels.sandbox.tools.cross_val import LeaveOneOut

        endog = self.results.model.endog
        exog = self.exog

        cv_iter = LeaveOneOut(self.nobs)
        res_loo = defaultdict(list)
        for inidx, outidx in cv_iter:
            for att in attributes:
                res_i = self.model_class(endog[inidx], exog[inidx,:]).fit()
                if isinstance(att, tuple): #hasattr(att, '__call__'): #callable:
                    res_loo[att[0]].append(att[1](res_i))
                else:
                    res_loo[att].append(getattr(res_i, att))

        return res_loo

    def get_all_obs(self):
        #this might not be the most efficient
        get_det_cov_params = lambda res: np.linalg.det(res.cov_params())
        attributes = ['params', 'mse_resid', ('det_cov_params', get_det_cov_params)]
        res = self.get_drop_obs(attributes)

        self.sigma2_not_obsi = np.asarray(res['mse_resid'])
        self.params_not_obsi = np.asarray(res['params'])
        self.det_cov_params_not_obsi = np.asarray(res['det_cov_params'])

        #this is not stored in linear model
        #self.xpx = np.dot(self.exog.T, self.exog)

        #don't use inplace division / because then we change original
        self.cov_ratio = (self.det_cov_params_not_obsi
                            / np.linalg.det(self.results.cov_params()))

        dfbetas = self.results.params - self.params_not_obsi#[None,:]
        dfbetas /= np.sqrt(self.sigma2_not_obsi[:,None])
        dfbetas /=  np.sqrt(np.diag(self.results.normalized_cov_params))
        self.dfbetas = dfbetas

    def summary_obs(self):
        #print self.dfbetas

#        table_raw = [ np.arange(self.nobs),
#                      self.endog,
#                      self.fittedvalues,
#                      self.cooks_distance(),
#                      self.resid_studentized_internal,
#                      self.hat_matrix_diag,
#                      self.dffits_internal,
#                      self.resid_studentized_external,
#                      self.dffits,
#                      self.dfbetas
#                      ]
        table_raw = [ ('obs', np.arange(self.nobs)),
                      ('endog', self.endog),
                      ('fitted\nvalue', self.results.fittedvalues),
                      ("Cook's\nd", self.cooks_distance()[0]),
                      ("student.\nresidual", self.resid_studentized_internal),
                      ('hat diag', self.hat_matrix_diag),
                      ('dffits \ninternal', self.dffits_internal[0]),
                      ("ext.stud.\nresidual", self.resid_studentized_external),
                      ('dffits', self.dffits[0]),
                      ('dfbeta\nslope', self.dfbetas[:,1]) #skip needs to partially unravel
                      ]
        colnames, data = zip(*table_raw) #unzip
        self.table_data = data
        data = np.column_stack(data)
        data = np.round(data,4)
        self.table = data
        from statsmodels.iolib.table import SimpleTable, default_html_fmt
        from statsmodels.iolib.tableformatting import fmt_base
        from copy import deepcopy
        fmt = deepcopy(fmt_base)
        fmt_html = deepcopy(default_html_fmt)
        fmt['data_fmts'] = ["%4d"] + ["%6.3f"] * (data.shape[1] - 1)
        #fmt_html['data_fmts'] = fmt['data_fmts']
        return SimpleTable(data, headers=colnames, txt_fmt=fmt,
                           html_fmt=fmt_html)


def summary_obs(res, alpha=0.05):

    from scipy import stats
    from statsmodels.sandbox.regression.predstd import wls_prediction_std

    infl = Influence(res)

    #standard error for predicted mean
    #Note: using hat_matrix only works for fitted values
    predict_mean_se = np.sqrt(infl.hat_matrix_diag*res.mse_resid)

    tppf = stats.t.isf(alpha/2., res.df_resid)
    predict_mean_ci = np.column_stack([
                        res.fittedvalues - tppf * predict_mean_se,
                        res.fittedvalues + tppf * predict_mean_se])


    #standard error for predicted observation
    predict_se, predict_ci_low, predict_ci_upp = wls_prediction_std(res)
    predict_ci = np.column_stack((predict_ci_low, predict_ci_upp))

    #standard deviation of residual
    resid_se = np.sqrt(res.mse_resid * (1 - infl.hat_matrix_diag))

    table_sm = np.column_stack([
                                  np.arange(res.nobs) + 1,
                                  res.model.endog,
                                  res.fittedvalues,
                                  predict_mean_se,
                                  predict_mean_ci[:,0],
                                  predict_mean_ci[:,1],
                                  predict_ci[:,0],
                                  predict_ci[:,1],
                                  res.resid,
                                  resid_se,
                                  infl.resid_studentized_internal,
                                  infl.cooks_distance()[0]
                                  ])


    #colnames, data = zip(*table_raw) #unzip
    data = table_sm
    ss2 = ['Obs', 'Dep Var\nPopulation', 'Predicted\nValue', 'Std Error\nMean Predict', 'Mean ci\n95% low', 'Mean ci\n95% upp', 'Predict ci\n95% low', 'Predict ci\n95% upp', 'Residual', 'Std Error\nResidual', 'Student\nResidual', "Cook's\nD"]
    colnames = ss2
    #self.table_data = data
    #data = np.column_stack(data)
    data = np.round(data,4)
    #self.table = data
    from statsmodels.iolib.table import SimpleTable, default_html_fmt
    from statsmodels.iolib.tableformatting import fmt_base
    from copy import deepcopy
    fmt = deepcopy(fmt_base)
    fmt_html = deepcopy(default_html_fmt)
    fmt['data_fmts'] = ["%4d"] + ["%6.3f"] * (data.shape[1] - 1)
    #fmt_html['data_fmts'] = fmt['data_fmts']
    st = SimpleTable(data, headers=colnames, txt_fmt=fmt,
                       html_fmt=fmt_html)

    return st, data, ss2



if __name__ == '__main__':

    import statsmodels.api as sm

    data = np.array('''\
    64 57  8
    71 59 10
    53 49  6
    67 62 11
    55 51  8
    58 50  7
    77 55 10
    57 48  9
    56 42 10
    51 42  6
    76 61 12
    68 57  9'''.split(), float).reshape(-1,3)
    varnames = 'weight height age'.split()

    endog = data[:,0]
    exog = sm.add_constant(data[:,2], prepend=True)


    res_ols = sm.OLS(endog, exog).fit()

    hh = (res_ols.model.exog * res_ols.model.pinv_wexog.T).sum(1)
    x = res_ols.model.exog
    hh_check = np.diag(np.dot(x, np.dot(res_ols.model.normalized_cov_params, x.T)))

    from numpy.testing import assert_almost_equal
    assert_almost_equal(hh, hh_check, decimal=13)

    res = res_ols #alias

    #http://en.wikipedia.org/wiki/PRESS_statistic
    #predicted residuals, leave one out predicted residuals
    resid_press = res.resid / (1-hh)
    ess_press = np.dot(resid_press, resid_press)

    sigma2_est = np.sqrt(res.mse_resid) #can be replace by different estimators of sigma
    sigma_est = np.sqrt(sigma2_est)
    resid_studentized = res.resid / sigma_est / np.sqrt(1 - hh)
    #http://en.wikipedia.org/wiki/DFFITS:
    dffits = resid_studentized * np.sqrt(hh / (1 - hh))

    nobs, k_vars = res.model.exog.shape
    #Belsley, Kuh and Welsch (1980) suggest a threshold for abs(DFFITS)
    dffits_threshold = 2 * np.sqrt(k_vars/nobs)

    res_ols.df_modelwc = res_ols.df_model + 1
    n_params = res.model.exog.shape[1]
    #http://en.wikipedia.org/wiki/Cook%27s_distance
    cooks_d = res.resid**2 / sigma2_est / res_ols.df_modelwc * hh / (1 - hh)**2
    #or
    #Eubank p.93, 94
    cooks_d2 = resid_studentized**2 / res_ols.df_modelwc * hh / (1 - hh)
    #threshold if normal, also Wikipedia
    from scipy import stats
    alpha = 0.1
    #df looks wrong
    print stats.f.isf(1-alpha, n_params, res.df_resid)
    print stats.f.sf(cooks_d, n_params, res.df_resid)


    print 'Cooks Distance'
    print cooks_d
    print cooks_d2

    doplot = 0
    if doplot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
#        ax = fig.add_subplot(3,1,1)
#        plt.plot(andrew_results.weights, 'o', label='rlm weights')
#        plt.legend(loc='lower left')
        ax = fig.add_subplot(3,1,2)
        plt.plot(cooks_d, 'o', label="Cook's distance")
        plt.legend(loc='upper left')
        ax2 = fig.add_subplot(3,1,3)
        plt.plot(resid_studentized, 'o', label='studentized_resid')
        plt.plot(dffits, 'o', label='DFFITS')
        leg = plt.legend(loc='lower left', fancybox=True)
        leg.get_frame().set_alpha(0.5) #, fontsize='small')
        ltext = leg.get_texts() # all the text.Text instance in the legend
        plt.setp(ltext, fontsize='small') # the legend text fontsize


    print reset_ramsey(res, degree=3)

    #note, constant in last column
    for i in range(1):
        print variance_inflation_factor(res.model.exog, i)

    infl = Influence(res_ols)
    print infl.resid_studentized_external
    print infl.resid_studentized_internal
    infl.summary_obs()

'''
>>> res.resid
array([  4.28571429,   4.        ,   0.57142857,  -3.64285714,
        -4.71428571,   1.92857143,  10.        ,  -6.35714286,
       -11.        ,  -1.42857143,   1.71428571,   4.64285714])
>>> infl.hat_matrix_diag
array([ 0.10084034,  0.11764706,  0.28571429,  0.20168067,  0.10084034,
        0.16806723,  0.11764706,  0.08403361,  0.11764706,  0.28571429,
        0.33613445,  0.08403361])
>>> infl.resid_press
array([  4.76635514,   4.53333333,   0.8       ,  -4.56315789,
        -5.24299065,   2.31818182,  11.33333333,  -6.94036697,
       -12.46666667,  -2.        ,   2.58227848,   5.06880734])
>>> infl.ess_press
465.98646628086374
'''