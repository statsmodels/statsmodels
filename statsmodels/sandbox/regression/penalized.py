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
from statsmodels.tools.decorators import (cache_readonly)
import statsmodels.base.model as base
from statsmodels.regression.linear_model import OLS, GLS, RegressionResults


def atleast_2dcols(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:,None]
    return x



class TheilGLS(GLS):
    '''GLS with stochastic restrictions

    TheilGLS estimates the following linear model

    .. math:: y = X \beta + u

    using additional information given by a stochastic constraint

    .. math:: q = R \beta + v

    :math:`E(u) = 0`, :math:`cov(u) = \Sigma`
    :math:`cov(u, v) = \Sigma_p`, with full rank.

    u and v are assumed to be independent of each other.
    If :math:`E(v) = 0`, then the estimator is unbiased.

    Note: The explanatory variables are not rescaled, the parameter estimates
    not scale equivariant and fitted values are not scale invariant since
    scaling changes the relative penalization weights (for given \Sigma_p).

    Note: GLS is not tested yet, only Sigma is identity is tested

    Notes
    -----

    The parameter estimates solves the moment equation:

    .. math:: (X' \\Sigma X + \\lambda R' \\sigma^2 \\Simga_p^{-1} R) b = X' \\Sigma y + \\lambda R' \\Simga_p^{-1} q

    :math:`\\lambda` is the penalization parameter similar to Ridge regression.

    If lambda is zero, then the parameter estimate is the same as OLS. If
    lambda goes to infinity, then the restriction is imposed with equality.

    R does not have to be square. The number of rows of R can be smaller
    than the number of parameters. In this case not all linear combination
    of parameters are penalized.

    The stochastic constraint can be interpreted in several different ways:

     - The prior information represents parameter estimates from independent
       prior samples.
     - We can consider it just as linear restrictions that we do not want
       to impose without uncertainty.
     - With a full rank square restriction matrix R, the parameter estimate
       is the same as a Bayesian posterior mean for the case of an informative
       normal prior, normal likelihood and known error variance \sigma.

    References
    ----------
    Theil Goldberger

    Baum, Christopher slides for tgmixed in Stata

    (I don't remember what I used when I first wrote the code.)

    Parameters
    ----------
    endog : array_like, 1-D
        dependent or endogenous variable
    exog : array_like, 1D or 2D
        array of explanatory or exogenous variables
    r_matrix : None or array_like, 2D
        array of linear restrictions for stochastic constraint.
        default is identity matrix that does not penalize constant, if constant
        is detected to be in `exog`.
    q_matrix : None or array_like
        mean of the linear restrictions. If None, the it is set to zeros.
    sigma_prior : None or array_like
        A fully specified sigma_prior is a square matrix with the same number
        of rows and columns as there are constraints (number of rows of r_matrix).
        If sigma_prior is None, a scalar or one-dimensional, then a diagonal matrix
        is created.
    sigma : None or array_like
        Sigma is the covariance matrix of the error term that is used in the same
        way as in GLS.

    '''


    def __init__(self, endog, exog, r_matrix=None, q_matrix=None,
                 sigma_prior=None, sigma=None):
        super(TheilGLS, self).__init__(endog, exog, sigma=sigma)

        if r_matrix is not None:
            r_matrix = np.asarray(r_matrix)
        else:
            try:
                const_idx = self.data.const_idx
            except AttributeError:
                const_idx = None

            k_exog = exog.shape[1]
            r_matrix = np.eye(k_exog)
            if const_idx is not None:
                keep_idx = range(k_exog)
                del keep_idx[const_idx]
                r_matrix = r_matrix[keep_idx]  # delete row for constant

        k_constraints, k_exog = r_matrix.shape
        self.r_matrix = r_matrix
        if k_exog != self.exog.shape[1]:
            raise ValueError('r_matrix needs to have the same number of columns'
                             'as exog')

        if q_matrix is not None:
            self.q_matrix = atleast_2dcols(q_matrix)
            if self.q_matrix.shape != (k_constraints, 1):
                raise ValueError('q_matrix has wrong shape')
        else:
            self.q_matrix = np.zeros(k_constraints)

        if sigma_prior is not None:
            sigma_prior = np.asarray(sigma_prior)
            if np.size(sigma_prior) == 1:
                sigma_prior = np.diag(sigma_prior * np.ones(k_constraints))
                #no numerical shortcuts are used for this case
            elif sigma_prior.ndim == 1:
                sigma_prior = np.diag(sigma_prior)
        else:
            sigma_prior = np.eye(k_constraints)

        if sigma_prior.shape != (k_constraints, k_constraints):
            raise ValueError('sigma_prior has wrong shape')

        self.sigma_prior = sigma_prior
        self.sigma_prior_inv = np.linalg.pinv(sigma_prior) #or inv


    def fit(self, lambd=1., cov_type='sandwich'):
        """Estimate parameters and return results instance

        Paramters
        ---------
        lambd : float
            penalization factor for the restriction, default is 1.
        cov_type : string, 'data-prior' or 'sandwich'
            'data-prior' assumes that the stochastic restriction reflects a
            previous sample. The covariance matrix of the parameter estimate
            is in this case the same form as the one of GLS.
            The covariance matrix for cov_type='sandwich' treats the stochastic
            restriction (R and q) as fixed and has a sandwich form analogously
            to M-estimators.


        Notes
        -----
        cov_params for cov_type data-prior, is calculated as

        .. math:: \\sigma^2 A^{-1}

        cov_params for cov_type sandwich, is calculated as

        .. math:: \\sigma^2 A^{-1} (X'X) A^{-1}

        where :math:`A = X' \\Sigma X + \\lambda \\sigma^2 R' \\Simga_p^{-1} R`

        :math:`\\sigma^2` is an estimate of the error variance.
        :math:`\\sigma^2` inside A is replaced by the estimate from the initial
        GLS estimate. :math:`\\sigma^2` in cov_params is obtained from the
        residuals of the final estimate.

        The sandwich form of the covariance estimator is not robust to
        misspecified heteroscedasticity or autocorrelation.

        """
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
        # Bayesian: lambd is precision = 1/sigma2_prior
        #print('lambd inside fit', lambd
        xx = np.dot(x.T, x)
        xpx = xx + \
              sigma2_e * lambd * np.dot(r_matrix.T, np.dot(sigma_prior_inv, r_matrix))
        xpy = np.dot(x.T, y) + \
              sigma2_e * lambd * np.dot(r_matrix.T, np.dot(sigma_prior_inv, q_matrix))
        #xpy = xpy[:,None]

        xpxi = np.linalg.pinv(xpx, rcond=1e-15**2)  #to match pinv(x) in OLS case
        params = np.dot(xpxi, xpy)    #or solve
        params = np.squeeze(params)
        # normalized_cov_params should have sandwich form xpxi @ xx @ xpxi
        if cov_type == 'sandwich':
            self.normalized_cov_params = xpxi.dot(xx).dot(xpxi)
        elif cov_type == 'data-prior':
            self.normalized_cov_params = xpxi    #why attach it to self, i.e. model?
        else:
            raise ValueError("cov_type has to be 'sandwich' or 'data-prior'")

        lfit = TheilRegressionResults(self, params,
                       normalized_cov_params=xpxi)

        lfit.penalization_factor = lambd
        return lfit

    def fit_minic(self, method='aicc', start_params=1., optim_args=None):
        """find penalization factor that minimizes gcv or an information criterion

        Parameters
        ----------
        method : string
            the name of an attribute of the results class. Currently the following
            are available aic, aicc, bic, gc and gcv.
        start_params : float
            starting values for the minimization to find the penalization factor
            `lambd`. Not since there can be local minima, it is best to try
            different starting values.
        optim_args : None or dict
            optimization keyword arguments used with `scipy.optimize.fmin`

        Returns
        -------
        min_lambd : float
            The penalization factor at which the target criterion is (locally)
            minimized.

        Notes
        -----
        This uses `scipy.optimize.fmin` as optimizer.

        """
        if optim_args is None:
            optim_args = {}

        #this doesn't make sense, since number of parameters stays unchanged
        # information criteria changes if we use df_model based on trace(hat_matrix)
        #need leave-one-out, gcv; or some penalization for weak priors
        #added extra penalization for lambd
        def get_ic(lambd):
            # this can be optimized more
            # for pure Ridge we can keep the eigenvector decomposition
            return getattr(self.fit(lambd), method)

        from scipy import optimize
        lambd = optimize.fmin(get_ic, start_params, **optim_args)
        return lambd

#TODO:
#I need the hatmatrix in the model if I want to do iterative fitting, e.g. GCV
#move to model or use it from a results instance inside the model,
#    each call to fit returns results instance
# note: we need to recalculate hatmatrix for each lambda, so keep in results is fine

class TheilRegressionResults(RegressionResults):

    def __init__(self, *args, **kwds):
        super(TheilRegressionResults, self).__init__(*args, **kwds)

        # overwrite df_model and df_resid
        self.df_model = self.hatmatrix_trace() - 1 #assume constant
        self.df_resid = self.model.endog.shape[0] - self.df_model - 1

    #cache
    @cache_readonly
    def hatmatrix_diag(self):
        '''

        diag(X' xpxi X)

        where xpxi = (X'X + sigma2_e * lambd * sigma_prior)^{-1}

        Notes
        -----

        uses wexog, so this includes weights or sigma - check this case

        not clear whether I need to multiply by sigmahalf, i.e.

        (W^{-0.5} X) (X' W X)^{-1} (W^{-0.5} X)'  or
        (W X) (X' W X)^{-1} (W X)'

        projection y_hat = H y    or in terms of transformed variables (W^{-0.5} y)

        might be wrong for WLS and GLS case
        '''
        # TODO is this still correct with sandwich normalized_cov_params, I guess not
        xpxi = self.model.normalized_cov_params
        #something fishy with self.normalized_cov_params in result, doesn't update
        #print(self.model.wexog.shape, np.dot(xpxi, self.model.wexog.T).shape
        return (self.model.wexog * np.dot(xpxi, self.model.wexog.T).T).sum(1)

    #@cache_readonly
    def hatmatrix_trace(self):
        return self.hatmatrix_diag.sum()

##    #this doesn't update df_resid
##    @property   #needs to be property or attribute (no call)
##    def df_model(self):
##        return self.hatmatrix_trace()

    #Note: mse_resid uses df_resid not nobs-k_vars, which might differ if df_model, tr(H), is used
    #in paper for gcv ess/nobs is used instead of mse_resid
    @cache_readonly
    def gcv(self):
        return self.mse_resid / (1. - self.hatmatrix_trace() / self.nobs)**2

    @cache_readonly
    def cv(self):
        return ((self.resid / (1. - self.hatmatrix_diag))**2).sum() / self.nobs

    @cache_readonly
    def aicc(self):
        aic = np.log(self.mse_resid) + 1
        aic += 2 * (1. + self.hatmatrix_trace()) / (self.nobs - self.hatmatrix_trace() -2)
        return aic


    def test_compatibility(self):
        """Hypothesis test for the compatibility of prior mean with data

        """
        # TODO: should we store the OLS results ?  not needed so far, but maybe cache
        #params_ols = np.linalg.pinv(self.model.exog).dot(self.model.endog)
        #res = self.wald_test(self.model.r_matrix, q_matrix=self.model.q_matrix, use_f=False)
        #from scratch
        res_ols = OLS(self.model.endog, self.model.exog).fit()
        r_mat = self.model.r_matrix
        r_diff = self.model.q_matrix - r_mat.dot(res_ols.params)[:,None]
        ols_cov_r = res_ols.cov_params(r_matrix=r_mat)
        statistic = r_diff.T.dot(np.linalg.solve(ols_cov_r + self.model.sigma_prior, r_diff))
        from scipy import stats
        df = np.linalg.matrix_rank(self.model.sigma_prior)   # same as r_mat.shape[0]
        pvalue = stats.chi2.sf(statistic, df)
        # TODO: return results class
        return statistic, pvalue, df


    def share_data(self):
        """a measure for the fraction of the data in the estimation result

        The share of the prior information is `1 - share_data`.

        Returns
        -------
        share : float between 0 and 1
            share of data defined as the ration between effective degrees of
            freedom of the model and the number (TODO should be rank) of the
            explanatory variables.

        """

        # this is hatmatrix_trace / self.exog.shape[1]
        # This needs to use rank of exog and not shape[1],
        # since singular exog is allowed
        return (self.df_model + 1) / self.model.rank  # + 1 is for constant


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

    examples = [3]

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
    params_l = []
    gcv_l = []
    aicc_l = []
    for lambd in np.linspace(0, 200, 21):
        res_l = mod.fit(lambd)
        #print(lambd, res_l.params[-2:], res_l.bic, res_l.bic + 1./lambd, res.df_model
        #print((lambd, res_l.params[-2:], res_l.bic, res.df_model, np.trace(res.normalized_cov_params)))
        params_l.append(res_l.params)
        gcv_l.append(res_l.gcv)
        aicc_l.append(res_l.aicc)

    params_l = np.array(params_l)


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

    plt.figure()
    plt.plot(gcv_l - gcv_l[0], label='gcv')
    plt.plot(aicc_l - aicc_l[0], label='aicc')
    plt.title('')
    plt.xlabel('strength of prior')
    plt.legend()
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
        gcv_l = []
        aicc_l = []
        for lambd in np.linspace(0, 20, 21):
            resi = mod.fit(5.*lambd)
            params_l.append(resi.params)
            gcv_l.append(resi.gcv)
            aicc_l.append(resi.aicc)

        params_l = np.array(params_l)

        plt.figure()
        plt.plot(params_l.T)
        plt.title('Panel Data with random intercept: shrinkage to being equal')
        plt.xlabel('parameter index')
        plt.figure()
        plt.plot(params_l[:,k_vars:])
        plt.title('Panel Data with random intercept: shrinkage to being equal')
        plt.xlabel('strength of prior')
        plt.figure()
        plt.plot(gcv_l, label='gcv')
        plt.plot(aicc_l, label='aicc')
        plt.title('Panel Data with random intercept: shrinkage to being equal')
        plt.xlabel('strength of prior')
        plt.legend()

        #plt.show()
