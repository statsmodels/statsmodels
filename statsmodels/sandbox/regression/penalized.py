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
  One question is which results are based on likelihood (residuals) and which
  are based on "posterior" as for example bse and cov_params

* helper functions to construct priors?
* increasing penalization for ordered regressors, e.g. polynomials

* compare with random/mixed effects/coefficient, like estimated priors



there is something fishy with the result instance, some things, e.g.
normalized_cov_params, don't look like they update correctly as we
search over lambda -> some stale state again ?

I added df_model to result class using the hatmatrix, but df_model is defined
in model instance not in result instance. -> not clear where refactoring should
occur. df_resid doesn't get updated correctly.
problem with definition of df_model, it has 1 subtracted for constant



"""
from __future__ import print_function
from statsmodels.compat.python import lrange
import numpy as np
import statsmodels.base.model as base
from statsmodels.regression.linear_model import OLS, GLS, RegressionResults


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
        #print('lambd inside fit', lambd
        xpx = np.dot(x.T, x) + \
              sigma2_e * lambd * np.dot(r_matrix.T, np.dot(sigma_prior_inv, r_matrix))
        xpy = np.dot(x.T, y) + \
              sigma2_e * lambd * np.dot(r_matrix.T, np.dot(sigma_prior_inv, q_matrix))
        #xpy = xpy[:,None]

        xpxi = np.linalg.pinv(xpx)
        params = np.dot(xpxi, xpy)    #or solve
        params = np.squeeze(params)
        self.normalized_cov_params = xpxi    #why attach it to self, i.e. model?

        lfit = TheilRegressionResults(self, params,
                       normalized_cov_params=xpxi)

        lfit.penalization_factor = lambd
        return lfit

    def fit_minic(self):
        #this doesn't make sense, since number of parameters stays unchanged
        #need leave-one-out, gcv; or some penalization for weak priors
        #added extra penalization for lambd
        def get_bic(lambd):
            #return self.fit(lambd).bic #+lambd #+ 1./lambd  #added 1/lambd for checking
            #return self.fit(lambd).gcv()
            #return self.fit(lambd).cv()
            return self.fit(lambd).aicc()

        from scipy import optimize
        lambd = optimize.fmin(get_bic, 1.)
        return lambd

#TODO:
#I need the hatmatrix in the model if I want to do iterative fitting, e.g. GCV
#move to model or use it from a results instance inside the model,
#    each call to fit returns results instance

class TheilRegressionResults(RegressionResults):

    #cache
    def hatmatrix_diag(self):
        '''

        diag(X' xpxi X)

        where xpxi = (X'X + lambd * sigma_prior)^{-1}

        Notes
        -----

        uses wexog, so this includes weights or sigma - check this case

        not clear whether I need to multiply by sigmahalf, i.e.

        (W^{-0.5} X) (X' W X)^{-1} (W^{-0.5} X)'  or
        (W X) (X' W X)^{-1} (W X)'

        projection y_hat = H y    or in terms of transformed variables (W^{-0.5} y)

        might be wrong for WLS and GLS case
        '''
        xpxi = self.model.normalized_cov_params
        #something fishy with self.normalized_cov_params in result, doesn't update
        #print(self.model.wexog.shape, np.dot(xpxi, self.model.wexog.T).shape
        return (self.model.wexog * np.dot(xpxi, self.model.wexog.T).T).sum(1)

    def hatmatrix_trace(self):
        return self.hatmatrix_diag().sum()

    #this doesn't update df_resid
    @property   #needs to be property or attribute (no call)
    def df_model(self):
        return self.hatmatrix_trace()

    #Note: mse_resid uses df_resid not nobs-k_vars, which might differ if df_model, tr(H), is used
    #in paper for gcv ess/nobs is used instead of mse_resid
    def gcv(self):
        return self.mse_resid / (1. - self.hatmatrix_trace() / self.nobs)**2

    def cv(self):
        return ((self.resid / (1. - self.hatmatrix_diag()))**2).sum() / self.nobs

    def aicc(self):
        aic = np.log(self.mse_resid) + 1
        aic += 2 * (1. + self.hatmatrix_trace()) / (self.nobs - self.hatmatrix_trace() -2)
        return aic


#contrast/restriction matrices, temporary location

def coef_restriction_meandiff(n_coeffs, n_vars=None, position=0):

    reduced = np.eye(n_coeffs) - 1./n_coeffs
    if n_vars is None:
        return reduced
    else:
        full = np.zeros((n_coeffs, n_vars))
        full[:, position:position+n_coeffs] = reduced
        return full

def coef_restriction_diffbase(n_coeffs, n_vars=None, position=0, base_idx=0):

    reduced = -np.eye(n_coeffs)  #make all rows, drop one row later
    reduced[:, base_idx] = 1

    keep = lrange(n_coeffs)
    del keep[base_idx]
    reduced = np.take(reduced, keep, axis=0)

    if n_vars is None:
        return reduced
    else:
        full = np.zeros((n_coeffs-1, n_vars))
        full[:, position:position+n_coeffs] = reduced
        return full

def next_odd(d):
    return d + (1 - d % 2)

def coef_restriction_diffseq(n_coeffs, degree=1, n_vars=None, position=0, base_idx=0):
    #check boundaries, returns "valid" ?

    if degree == 1:
        diff_coeffs = [-1, 1]
        n_points = 2
    elif degree > 1:
        from scipy import misc
        n_points = next_odd(degree + 1)  #next odd integer after degree+1
        diff_coeffs = misc.central_diff_weights(n_points, ndiv=degree)

    dff = np.concatenate((diff_coeffs, np.zeros(n_coeffs - len(diff_coeffs))))
    from scipy import linalg
    reduced = linalg.toeplitz(dff, np.zeros(n_coeffs - len(diff_coeffs) + 1)).T
    #reduced = np.kron(np.eye(n_coeffs-n_points), diff_coeffs)

    if n_vars is None:
        return reduced
    else:
        full = np.zeros((n_coeffs-1, n_vars))
        full[:, position:position+n_coeffs] = reduced
        return full


##
##    R = np.c_[np.zeros((n_groups, k_vars-1)), np.eye(n_groups)]
##    r = np.zeros(n_groups)
##    R = np.c_[np.zeros((n_groups-1, k_vars)),
##              np.eye(n_groups-1)-1./n_groups * np.ones((n_groups-1, n_groups-1))]



if __name__ == '__main__':

    import numpy as np
    import statsmodels.api as sm

    examples = [2]

    np.random.seed(765367)
    np.random.seed(97653679)
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
    print(res_ols.params)
    print(res.params)

    #example 2
    #I need more flexible penalization in example, the penalization should
    #get stronger for higher order terms
    #np.random.seed(1)
    nobs = 200
    k_vars = 10
    k_true = 6
    sig_e = 0.25   #0.5
    x = np.linspace(-2,2, nobs)
    #X = sm.add_constant(np.column_stack((x, x**2, (x/5.)**3)), prepend=True)
    X = (x/x.max())[:,None]**np.arange(k_vars)
    beta = np.zeros(k_vars)
    beta[:k_true] = np.array([1, -2, 0.5, 1.5, -0.1, 0.1])[:k_true]
    y_true = np.dot(X, beta)
    y = y_true + sig_e * np.random.normal(size=nobs)

    res_ols = sm.OLS(y, X).fit()

    #R = np.c_[np.zeros((k_vars-4, 4)), np.eye(k_vars-4)]  # has two large true coefficients penalized
    not_penalized = 4
    R = np.c_[np.zeros((k_vars-not_penalized, not_penalized)), np.eye(k_vars-not_penalized)]
    #increasingly strong penalization
    R = np.c_[np.zeros((k_vars-not_penalized, not_penalized)), np.diag((1+2*np.arange(k_vars-not_penalized)))]
    r = np.zeros(k_vars-not_penalized)
##    R = -coef_restriction_diffseq(6, 1, n_vars=10, position=4)  #doesn't make sense for polynomial
##    R = np.vstack((R, np.zeros(R.shape[1])))
##    R[-1,-1] = 1
    r = np.zeros(R.shape[0])
    lambd = 2 #1e-4
    mod = TheilGLS(y, X, r_matrix=R, q_matrix=r, sigma_prior=lambd)
    res = mod.fit()
    print(res_ols.params)
    print(res.params)

    res_bic = mod.fit_minic()   #this will just return zero
    res = mod.fit(res_bic)

    print(res_bic)
    for lambd in np.linspace(0, 80, 21):
        res_l = mod.fit(lambd)
        #print(lambd, res_l.params[-2:], res_l.bic, res_l.bic + 1./lambd, res.df_model
        print((lambd, res_l.params[-2:], res_l.bic, res.df_model, np.trace(res.normalized_cov_params)))


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

    if 3 in examples:
        #example 3
        nobs = 600
        nobs_i = 20
        n_groups = nobs // nobs_i
        k_vars = 3

        from statsmodels.sandbox.panel.random_panel import PanelSample
        dgp = PanelSample(nobs, k_vars, n_groups)
        dgp.group_means = 2 + np.random.randn(n_groups) #add random intercept
        print('seed', dgp.seed)
        y = dgp.generate_panel()
        X = np.column_stack((dgp.exog[:,1:],
                               dgp.groups[:,None] == np.arange(n_groups)))
        res_ols = sm.OLS(y, X).fit()
        R = np.c_[np.zeros((n_groups, k_vars-1)), np.eye(n_groups)]
        r = np.zeros(n_groups)
        R = np.c_[np.zeros((n_groups-1, k_vars)),
                  np.eye(n_groups-1)-1./n_groups * np.ones((n_groups-1, n_groups-1))]
        r = np.zeros(n_groups-1)
        R[:, k_vars-1] = -1

        lambd = 1 #1e-4
        mod = TheilGLS(y, X, r_matrix=R, q_matrix=r, sigma_prior=lambd)
        res = mod.fit()
        print(res.params)

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

        #plt.show()
