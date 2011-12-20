# -*- coding: utf-8 -*-
"""linear model with Theil prior probabilistic restrictions, generalized Ridge

Created on Tue Dec 20 00:10:10 2011

Author: Josef Perktold
License: BSD-3

open issues
* selection of smoothing factor, strength of prior, cross validation
* GLS, does this really work this way
* None of inherited results have been checked yet,
  I'm not sure if any need to be adjusted or if only interpretation changes

* helper functions to construct priors?
* increasing penalization for ordered regressors, e.g. polynomials

* compare with random/mixed effects/coefficient, like estimated priors

"""

import numpy as np
import scikits.statsmodels.base.model as base
from scikits.statsmodels.regression.linear_model import OLS, GLS, RegressionResults


def atleast_2dcols(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:,None]
    return x



class TheilGLS(GLS):
    '''GLS with probabilistic restrictions

    essentially Bayes with informative prior

    note: I'm making up the GLS part, might work only for OLS

    '''


    def __init__(self, endog, exog, r_matrix, q_matrix=None, sigma_prior=None, sigma=None):
        self.r_matrix = np.asarray(r_matrix)
        self.q_matrix = atleast_2dcols(q_matrix)
        if np.size(sigma_prior) == 1:
            sigma_prior = sigma_prior * np.eye(self.r_matrix.shape[0]) #no numerical shortcuts

        self.sigma_prior = sigma_prior
        self.sigma_prior_inv = np.linalg.pinv(sigma_prior) #or inv
        super(self.__class__, self).__init__(endog, exog, sigma=sigma)


    def fit(self, lambd=1.):
        #this does duplicate transformation, but I need resid not wresid
        res_gls = GLS(self.endog, self.exog, sigma=self.sigma).fit()
        self.res_gls = res_gls
        sigma2_e = res_gls.mse_resid

        r_matrix = self.r_matrix
        q_matrix = self.q_matrix
        sigma_prior_inv = self.sigma_prior_inv
        x = self.wexog
        y = self.wendog[:,None]
        #why are sigma2_e * lambd multiplied, not ratio?
        #larger lambd -> stronger prior  (it's not the variance)
        xpx = np.dot(x.T, x) + \
              sigma2_e * lambd * np.dot(r_matrix.T, np.dot(sigma_prior_inv, r_matrix))
        xpy = np.dot(x.T, y) + \
              sigma2_e * lambd * np.dot(r_matrix.T, np.dot(sigma_prior_inv, q_matrix))
        #xpy = xpy[:,None]

        xpxi = np.linalg.pinv(xpx)
        params = np.dot(xpxi, xpy)    #or solve
        params = np.squeeze(params)
        self.normalized_cov_params = xpxi

        lfit = RegressionResults(self, params,
                       normalized_cov_params=self.normalized_cov_params)
        return lfit

    def fit_minic(self):
        #this doesn't make sense, since number of parameters stays unchanged
        #need leave-one-out, gcv; or some penalization for weak priors
        #added extra penalization for lambd
        def get_bic(lambd):
            return self.fit(lambd).bic + 1./lambd  #added 1/lambd for checking

        from scipy import optimize
        lambd = optimize.fmin(get_bic, 1.)
        return lambd



if __name__ == '__main__':

    import numpy as np
    import scikits.statsmodels.api as sm

    np.random.seed(765367)
    nsample = 100
    x = np.linspace(0,10, nsample)
    X = sm.add_constant(np.column_stack((x, x**2, (x/5.)**3)), prepend=True)
    beta = np.array([10, 1, 0.1, 0.5])
    y = np.dot(X, beta) + np.random.normal(size=nsample)

    res_ols = sm.OLS(y, X).fit()

    R = [[0, 0, 0 , 1]]
    r = [0] #, 0, 0 , 0]
    lambd = 1 #1e-4
    mod = TheilGLS(y, X, r_matrix=R, q_matrix=r, sigma_prior=lambd)
    res = mod.fit()
    print res_ols.params
    print res.params

    #example 2
    #I need more flexible penalization in example, the penalization should
    #get stronger for higher order terms
    np.random.seed(1)
    nobs = 100
    k_vars = 10
    k_true = 6
    x = np.linspace(-2,2, nobs)
    #X = sm.add_constant(np.column_stack((x, x**2, (x/5.)**3)), prepend=True)
    X = (x/x.max())[:,None]**np.arange(k_vars)
    beta = np.zeros(k_vars)
    beta[:k_true] = np.array([1, -2, 0.5, 1.5, 1, 1, 1])[:k_true]
    y_true = np.dot(X, beta)
    y = y_true + 0.5 * np.random.normal(size=nobs)

    res_ols = sm.OLS(y, X).fit()

    R = np.c_[np.zeros((k_vars-4, 4)), np.eye(k_vars-4)]
    r = np.zeros(k_vars-4)
    lambd = 2 #1e-4
    mod = TheilGLS(y, X, r_matrix=R, q_matrix=r, sigma_prior=lambd)
    res = mod.fit()
    print res_ols.params
    print res.params

    res_bic = mod.fit_minic()   #this will just return zero
    res = mod.fit(res_bic)

    print res_bic
    for lambd in np.linspace(0, 20, 21):
        res_l = mod.fit(lambd)
        print lambd, res_l.params[-2:], res_l.bic, res_l.bic + 1./lambd

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(beta, 'k-o', label='true')
    plt.plot(res_ols.params, '-o', label='ols')
    plt.plot(res.params, '-o', label='theil')
    plt.legend()
    plt.title('Polynomial fitting: estimated coefficients')

    plt.figure()
    plt.plot(y, 'o')
    plt.plot(y_true, 'k-', label='true')
    plt.plot(res_ols.fittedvalues, '-', label='ols')
    plt.plot(res.fittedvalues, '-', label='theil')
    plt.legend()
    plt.title('Polynomial fitting: fitted values')
    #plt.show()

    #example 3
    nobs = 200
    nobs_i = 20
    n_groups = nobs // nobs_i
    k_vars = 3

    from scikits.statsmodels.sandbox.panel.random_panel import PanelSample
    dgp = PanelSample(nobs, k_vars, n_groups)
    dgp.group_means = 2 + np.random.randn(n_groups) #add random intercept
    print 'seed', dgp.seed
    y = dgp.generate_panel()
    X = np.column_stack((dgp.exog[:,1:],
                           dgp.groups[:,None] == np.arange(n_groups)))
    res_ols = sm.OLS(y, X).fit()
    R = np.c_[np.zeros((n_groups, k_vars-1)), np.eye(n_groups)]
    r = np.zeros(n_groups)
    R = np.c_[np.zeros((n_groups-1, k_vars)), np.eye(n_groups-1)]
    r = np.zeros(n_groups-1)
    R[:, k_vars-1] = -1

    lambd = 1 #1e-4
    mod = TheilGLS(y, X, r_matrix=R, q_matrix=r, sigma_prior=lambd)
    res = mod.fit()
    print res.params

    params_l = []
    for lambd in np.linspace(0, 20, 21):
        params_l.append(mod.fit(5.*lambd).params)

    params_l = np.array(params_l)

    plt.figure()
    plt.plot(params_l.T)
    plt.title('Panel Data with random intercept: shrinkage to being equal')
    plt.xlabel('parameter index')
    plt.figure()
    plt.plot(params_l[:,k_vars:])
    plt.title('Panel Data with random intercept: shrinkage to being equal')
    plt.xlabel('strength of prior')

    plt.show()
