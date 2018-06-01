# -*- coding: utf-8 -*-
"""Score, lagrange multiplier and conditional moment tests
robust to misspecification or without specification of higher moments

Created on Thu Oct 30 00:42:38 2014

Author: Josef Perktold
License: BSD-3

Notes
-----

This module is a mixture of very general and very specific functions for
hypothesis testing in general models, targeted mainly to non-normal models.

Some of the options or versions of these tests are mainly intented for
cross-checking and to replicate different examples in references.

We need clean versions with good defaults for those functions that are
intended for the user.


References
----------
general:

Wooldridge
Cameron and Trivedi
Wooldridge, and Cameron Trivedi also cover the special cases for GLM/LEF
White
Pagan and Vella
Newey and McFadden
Davidson and MacKinnon

GLM:

Boos
Breslow 1990


Poisson:
...



"""

import numpy as np
from scipy import stats


from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS


class ResultsGeneric(object):


    def __init__(self, **kwds):
        self.__dict__.update(kwds)


class TestResults(ResultsGeneric):


    def summary(self):
        txt = 'Specification Test (LM, score)\n'
        stat = [self.c1, self.c2, self.c3]
        pval = [self.pval1, self.pval2, self.pval3]
        description = ['nonrobust', 'dispersed', 'HC']

        for row in zip(description, stat, pval):
            txt += '%-12s  statistic = %6.4f  pvalue = %6.5f\n' % row

        txt += '\nAssumptions:\n'
        txt += 'nonrobust: variance is correctly specified\n'
        txt += 'dispersed: variance correctly specified up to scale factor\n'
        txt += 'HC       : robust to any heteroscedasticity\n'
        txt += 'test is not robust to correlation across observations'


        return txt

def lm_test_glm(result, exog_extra, mean_deriv=None):
    '''score/lagrange multiplier test for GLM

    Wooldridge procedure for test of mean function in GLM

    Parameters
    ----------
    results : GLMResults instance
        results instance with the constrained model
    exog_extra : ndarray or None
        additional exogenous variables for variable addition test
        This can be set to None if mean_deriv is provided.
    mean_deriv : None or ndarray
        Extra moment condition that correspond to the partial derivative of
        a mean function with respect to some parameters.

    Returns
    -------
    test_results : Results instance
        The results instance has the following attributes which are score
        statistic and p-value for 3 versions of the score test.

        c1, pval1 : nonrobust score_test results
        c2, pval2 : score test results robust to over or under dispersion
        c3, pval3 : score test results fully robust to any heteroscedasticity

        The test results instance also has a simple summary method.

    Notes
    -----
    TODO: add `df` to results and make df detection more robust

    This implements the auxiliary regression procedure of Wooldridge,
    implemented based on the presentation in chapter 8 in Handbook of ...


    References
    ----------
    Wooldridge
    Wooldridge
    and more Wooldridge

    '''


    if hasattr(result, '_result'):
        res = result._result
    else:
        res = result

    mod = result.model
    nobs = mod.endog.shape[0]

    #mean_func = mod.family.link.inverse
    dlinkinv = mod.family.link.inverse_deriv

    # derivative of mean function w.r.t. beta (linear params)
    dm = lambda x, linpred: dlinkinv(linpred)[:,None] * x

    var_func = mod.family.variance

    x = result.model.exog
    x2 = exog_extra

    # test omitted
    lin_pred = res.predict(linear=True)
    dm_incl = dm(x, lin_pred)
    if x2 is not None:
        dm_excl = dm(x2, lin_pred)
        if mean_deriv is not None:
            # allow both and stack
            dm_excl = np.column_stack((dm_excl, mean_deriv))
    elif mean_deriv is not None:
        dm_excl = mean_deriv
    else:
        raise ValueError('either exog_extra or mean_deriv have to be provided')

    # TODO check for rank or redundant, note OLS calculates the rank
    k_constraint = dm_excl.shape[1]
    fittedvalues = res.predict()  # discrete has linpred instead of mean
    v = var_func(fittedvalues)
    std = np.sqrt(v)
    res_ols1 = OLS(res.resid_response / std, np.column_stack((dm_incl, dm_excl)) / std[:, None]).fit()

    # case: nonrobust assumes variance implied by distribution is correct
    c1 = res_ols1.ess
    pval1 = stats.chi2.sf(c1, k_constraint)
    #print c1, stats.chi2.sf(c1, 2)

    # case: robust to dispersion
    c2 = nobs * res_ols1.rsquared
    pval2 = stats.chi2.sf(c2, k_constraint)
    #print c2, stats.chi2.sf(c2, 2)

    # case: robust to heteroscedasticity
    from statsmodels.stats.multivariate_tools import partial_project
    pp = partial_project(dm_excl / std[:,None], dm_incl / std[:,None])
    resid_p = res.resid_response / std
    res_ols3 = OLS(np.ones(nobs), pp.resid * resid_p[:,None]).fit()
    #c3 = nobs * res_ols3.rsquared   # this is Wooldridge
    c3b = res_ols3.ess  # simpler if endog is ones
    pval3 = stats.chi2.sf(c3b, k_constraint)

    tres = TestResults(c1=c1, pval1=pval1,
                       c2=c2, pval2=pval2,
                       c3=c3b, pval3=pval3)

    return tres



def cm_test_robust(resid, resid_deriv, instruments, weights=1):
    '''score/lagrange multiplier of Wooldridge

    generic version of Wooldridge procedure for test of conditional moments

    Limitation: This version allows only for one unconditional moment
    restriction, i.e. resid is scalar for each observation.
    Another limitation is that it assumes independent observations, no
    correlation in residuals and weights cannot be replaced by cross-observation
    whitening.

    Parameters
    ----------
    resid : ndarray, (nobs, )
        conditional moment restriction, E(r | x, params) = 0
    resid_deriv : ndarray, (nobs, k_params)
        derivative of conditional moment restriction with respect to parameters
    instruments : ndarray, (nobs, k_instruments)
        indicator variables of Wooldridge, multiplies the conditional momen
        restriction
    weights : ndarray
        This is a weights function as used in WLS. The moment
        restrictions are multiplied by weights. This corresponds to the
        inverse of the variance in a heteroscedastic model.

    Returns
    -------
    test_results : Results instance
        ???  TODO

    Notes
    -----

    This implements the auxiliary regression procedure of Wooldridge,
    implemented based on procedure 2.1 in Wooldridge 1990.

    Wooldridge allows for multivariate conditional moments (`resid`)
    TODO: check dimensions for multivariate case for extension


    References
    ----------
    Wooldridge
    Wooldridge
    and more Wooldridge

    '''
    # notation: Wooldridge uses too mamny Greek letters
    # instruments is capital lambda
    # resid is small phi
    # resid_deriv is capital phi
    # weights is C


    nobs = resid.shape[0]


    from statsmodels.stats.multivariate_tools import partial_project

    w_sqrt = np.sqrt(weights)
    if np.size(weights) > 1:
        w_sqrt = w_sqrt[:,None]
    pp = partial_project(instruments * w_sqrt, resid_deriv * w_sqrt)
    mom_resid = pp.resid

    moms_test = mom_resid * resid[:, None] * w_sqrt

    # we get this here in case we extend resid to be more than 1-D
    k_constraint = moms_test.shape[1]

    # use OPG variance as in Wooldridge 1990. This might generalize
    cov = moms_test.T.dot(moms_test)
    diff = moms_test.sum(0)

    # see Wooldridge last page in appendix
    stat = diff.dot(np.linalg.solve(cov, diff))

    # for checking, this corresponds to nobs * rsquared of auxiliary regression
    stat2 = OLS(np.ones(nobs), moms_test).fit().ess


    pval = stats.chi2.sf(stat, k_constraint)

    return stat, pval, stat2


def lm_robust(score, R, Ainv, B, V=None):
    '''general formula for score/LM test

    generalized score or lagrange multiplier test for implicit constraints

    `r(params) = 0`, with gradient `R = d r / d params`

    linear constraints are given by `R params - q = 0`

    It is assumed that all arrays are evaluated at the constrained estimates.


    Parameters
    ----------
    score : ndarray, 1-D
        derivative of objective function at estimated parameters
        of constrained model
    constraint_matrix R : ndarray
        Linear restriction matrix or Jacobian of nonlinear constraints
    hessian_inv, Ainv : ndarray, symmetric, square
        inverse of second derivative of objective function
        TODO: could be OPG or any other estimator if information matrix
        equality holds
    cov_score B :  ndarray, symmetric, square
        covariance matrix of the score. This is the inner part of a sandwich
        estimator.
    cov_params V :  ndarray, symmetric, square
        covariance of full parameter vector evaluated at constrained parameter
        estimate. This can be specified instead of cov_score B.

    Returns
    -------
    lm_stat : float
        score/lagrange multiplier statistic

    Notes
    -----

    '''

    tmp = R.dot(Ainv)
    wscore = tmp.dot(score)  # C Ainv score

    if B is None and V is None:
        # only Ainv is given, so we assume information matrix identity holds
        # computational short cut, should be same if Ainv == inv(B)
        lm_stat = score.dot(Ainv.dot(score))
    else:
        # information matrix identity does not hold
        if V is None:
            inner = tmp.dot(B).dot(tmp.T)
        else:
            inner = R.dot(V).dot(R.T)

        #lm_stat2 = wscore.dot(np.linalg.pinv(inner).dot(wscore))
        # Let's assume inner is invertible, TODO: check if usecase for pinv exists
        lm_stat = wscore.dot(np.linalg.solve(inner, wscore))

    return lm_stat#, lm_stat2


def lm_robust_subset(score, k_constraints, score_deriv, cov_score):
    '''general formula for score/LM test

    generalized score or lagrange multiplier test for constraints on a subset
    of parameters

    `params_1 = value`, where params_1 is a subset of the unconstrained
    parameter vector.

    It is assumed that all arrays are evaluated at the constrained estimates.


    Parameters
    ----------
    score : ndarray, 1-D
        derivative of objective function at estimated parameters
        of constrained model
    k_constraint: int
        number of constraints
    score_deriv : ndarray, symmetric, square
        inverse of second derivative of objective function
        TODO: could be OPG or any other estimator if information matrix
        equality holds
    cov_score B :  ndarray, symmetric, square
        covariance matrix of the score. This is the inner part of a sandwich
        estimator.
    not cov_params V :  ndarray, symmetric, square
        covariance of full parameter vector evaluated at constrained parameter
        estimate. This can be specified instead of cov_score B.

    Returns
    -------
    lm_stat : float
        score/lagrange multiplier statistic

    Notes
    -----

    The implementation is based on Boos 1992 section 4.1. The same derivation
    is also in other articles and in text books.

    '''

    # Notation in Boos
    # score `S = sum (s_i)
    # score_obs `s_i`
    # score_deriv `I` is derivative of score (hessian)
    # `D` is covariance matrix of score, OPG product given independent observations

    #k_params = len(score)

    # Note: I reverse order between constraint and unconstrained compared to Boos

    # submatrices of score_deriv/hessian
    # these are I22 and I12 in Boos
    #h_uu = score_deriv[-k_constraints:, -k_constraints:]
    h_uu = score_deriv[:-k_constraints, :-k_constraints]
    h_cu = score_deriv[-k_constraints:, :-k_constraints]

    # TODO: pinv or solve ?
    tmp_proj = h_cu.dot(np.linalg.inv(h_uu))
    tmp = np.column_stack((-tmp_proj, np.eye(k_constraints))) #, tmp_proj))

    cov_score_constraints = tmp.dot(cov_score.dot(tmp.T))

    #lm_stat2 = wscore.dot(np.linalg.pinv(inner).dot(wscore))
    # Let's assume inner is invertible, TODO: check if usecase for pinv exists
    lm_stat = score.dot(np.linalg.solve(cov_score_constraints, score))
    pval = stats.chi2.sf(lm_stat, k_constraints)

#     # check second calculation Boos referencing Kent 1982 and Engle 1984
#     # we can use this when robust_cov_params of full model is available
#     h_inv = np.linalg.inv(score_deriv)
#     v = h_inv.dot(cov_score.dot(h_inv)) # this is robust cov_params
#     v_cc = v[:k_constraints, :k_constraints]
#     h_cc = score_deriv[:k_constraints, :k_constraints]
#     # brute force calculation:
#     h_resid_cu = h_cc - h_cu.dot(np.linalg.solve(h_uu, h_cu))
#     cov_s_c = h_resid_cu.dot(v_cc.dot(h_resid_cu))
#     diff = np.max(np.abs(cov_s_c - cov_score_constraints))
    return lm_stat, pval  #, lm_stat2


def lm_robust_subset_parts(score, k_constraints,
                           score_deriv_uu, score_deriv_cu,
                           cov_score_cc, cov_score_cu, cov_score_uu):
    """robust generalized score tests on subset of parameters

    This is the same as lm_robust_subset with arguments in parts of
    partitioned matrices.
    This can be useful, when we have the parts based on different estimation
    procedures, i.e. when we don't have the full unconstrained model.

    Calculates mainly the covariance of the constraint part of the score.

    TODO: these function should just return the covariance of the score
    instead of calculating the score/lm test.

    Implementation similar to lm_robust_subset based on Boos 1992, section 4.1
    """

    tmp_proj = np.linalg.solve(score_deriv_uu, score_deriv_cu)
    tmp = tmp_proj.dot(cov_score_cu)

    cov = cov_score_cc
    cov -= tmp_proj
    cov -= tmp_proj.T
    cov += tmp.dot(tmp_proj.T)

    lm_stat = score.dot(np.linalg.solve(cov, score))
    pval = stats.chi2.sf(lm_stat, k_constraints)
    return lm_stat, pval


def lm_robust_reparameterized(score, params_deriv,
                           score_deriv, cov_score):
    """robust generalized score test for transformed parameters

    The parameters are given by a nonlinear transformation of the estimated
    reduced parameters

    `params = g(params_reduced)`  with jacobian `G = d g / d params_reduced`

    score and other arrays are for full parameter space `params`

    Boos 1992, section 4.4
    """
    # Boos notation
    # params_deriv G

    k_params, k_constraints = params_deriv.shape

    G = params_deriv  # shortcut alias

    tmp_c0 = np.linalg.solve(G.dot(score_deriv.dot(G)))
    tmp_c1 = score_deriv.dot(G.dot(tmp_c0.dot(G)))
    tmp_c = np.eye(k_params) - tmp_c1

    cov = tmp_c.dot(cov_score.dot(tmp_c))  # warning: reduced rank

    lm_stat = score.dot(np.linalg.pinv(cov).dot(score))
    pval = stats.chi2.sf(lm_stat, k_constraints)
    return lm_stat, pval


def dispersion_poisson(results):
    """Score/LM type tests for Poisson variance assumptions

    Null Hypothesis is

    H0: var(y) = E(y) and assuming E(y) is correctly specified
    H1: var(y) ~= E(y)

    The tests are based on the constrained model, i.e. the Poisson model.
    The tests differ in their assumed alternatives, and in their maintained
    assumptions.

    Parameters
    ----------
    results : Poisson results instance
        TODO: currently uses GLMResults properties
        this can be either a discrete Poisson or a GLM with family Poisson
        results instance

    """
    from scipy import stats

    if hasattr(results, '_results'):
        results = results._results

    endog = results.model.endog
    nobs = endog.shape[0]   #TODO: use attribute, may need to be added
    fitted = results.predict()
    #fitted = results.fittedvalues  # discrete has linear prediction
    #this assumes Poisson
    resid2 = results.resid_response**2
    var_resid_endog = (resid2 - endog)
    var_resid_fitted = (resid2 - fitted)
    std1 = np.sqrt(2 * (fitted**2).sum())

    var_resid_endog_sum = var_resid_endog.sum()
    dean_a = var_resid_fitted.sum() / std1
    dean_b = var_resid_endog_sum / std1
    dean_c = (var_resid_endog / fitted).sum() / np.sqrt(2 * nobs)

    pval_dean_a = stats.norm.sf(np.abs(dean_a))
    pval_dean_b = stats.norm.sf(np.abs(dean_b))
    pval_dean_c = stats.norm.sf(np.abs(dean_c))

    results_all = [[dean_a, pval_dean_a],
                   [dean_b, pval_dean_b],
                   [dean_c, pval_dean_c]]
    description = [['Dean A', 'mu (1 + a mu)'],
                   ['Dean B', 'mu (1 + a mu)'],
                   ['Dean C', 'mu (1 + a)']]

    # Cameron Trived auxiliary regression page 78 count book 1989
    endog_v = var_resid_endog / fitted
    res_ols_nb2 = OLS(endog_v, fitted).fit(use_t=False)
    stat_ols_nb2 = res_ols_nb2.tvalues[0]
    pval_ols_nb2 = res_ols_nb2.pvalues[0]
    results_all.append([stat_ols_nb2, pval_ols_nb2])
    description.append(['CT nb2', 'mu (1 + a mu)'])

    res_ols_nb1 = OLS(endog_v, fitted).fit(use_t=False)
    stat_ols_nb1 = res_ols_nb1.tvalues[0]
    pval_ols_nb1 = res_ols_nb1.pvalues[0]
    results_all.append([stat_ols_nb1, pval_ols_nb1])
    description.append(['CT nb1', 'mu (1 + a)'])

    endog_v = var_resid_endog / fitted
    res_ols_nb2 = OLS(endog_v, fitted).fit(cov_type='HC1', use_t=False)
    stat_ols_hc1_nb2 = res_ols_nb2.tvalues[0]
    pval_ols_hc1_nb2 = res_ols_nb2.pvalues[0]
    results_all.append([stat_ols_hc1_nb2, pval_ols_hc1_nb2])
    description.append(['CT nb2 HC1', 'mu (1 + a mu)'])

    res_ols_nb1 = OLS(endog_v, np.ones(len(endog_v))).fit(cov_type='HC1', use_t=False)
    stat_ols_hc1_nb1 = res_ols_nb1.tvalues[0]
    pval_ols_hc1_nb1 = res_ols_nb1.pvalues[0]
    results_all.append([stat_ols_hc1_nb1, pval_ols_hc1_nb1])
    description.append(['CT nb1 HC1', 'mu (1 + a)'])


    return np.array(results_all), description


def dispersion_poisson_generic(results, exog_new_test, exog_new_control=None,
                               include_score=False, use_endog=True,
                               cov_type='HC1', cov_kwds=None, use_t=False):
    """A variable addition test for the variance function

    This uses an artificial regression to calculate a variant of an LM or
    generalized score test for the specification of the variance assumption
    in a Poisson model. The performed test is a Wald test on the coefficients
    of the `exog_new_test`.



    """
    from scipy import stats

    if hasattr(results, '_results'):
        results = results._results

    endog = results.model.endog
    nobs = endog.shape[0]   #TODO: use attribute, may need to be added
    #fitted = results.predict(results.params) #for discrete.Poisson
    fitted = results.fittedvalues
    #this assumes Poisson
    resid2 = results.resid_response**2
    if use_endog:
        var_resid = (resid2 - endog)
    else:
        var_resid = (resid2 - fitted)

    endog_v = var_resid / fitted

    k_constraints = exog_new_test.shape[1]
    ex_list = [exog_new_test]
    if include_score:
        score_obs = results.model.score_obs(results.params)
        ex_list.append(score_obs)

    if exog_new_control is not None:
        ex_list.append(score_obs)

    if len(ex_list) > 1:
        ex = np.column_stack(ex_list)
        use_wald = True
    else:
        ex = ex_list[0]  # no control variables in exog
        use_wald = False

    res_ols = OLS(endog_v, ex).fit(cov_type=cov_type, cov_kwds=cov_kwds,
                  use_t=use_t)

    if use_wald:
        # we have controls and need to test coefficients
        k_vars = ex.shape[1]
        constraints = np.eye(k_constraints, k_vars)
        ht = res_ols.wald_test(constraints)
        stat_ols = ht.statistic
        pval_ols = ht.pvalue
    else:
        # we don't have controls and can use overall fit
        nobs = endog_v.shape[0]
        stat_ols = nobs * res_ols.rsquared
        pval_ols = stats.chi2.sf(stat_ols, k_constraints)

    return stat_ols, pval_ols



def conditional_moment_test_generic(mom_test, mom_test_deriv,
                                    mom_incl, mom_incl_deriv,
                                    var_mom_all=None,
                                    cov_type='OPG', cov_kwds=None):
    """generic conditional moment test

    This is mainly intended as internal function in support of diagnostic
    and specification tests. It has no conversion and checking of correct
    arguments.

    Parameters
    ----------
    mom_test : ndarray, 2-D (nobs, k_constraints)
        moment conditions that will be tested to be zero
    mom_test_deriv : ndarray, 2-D, square (k_constraints, k_constraints)
        derivative of moment conditions under test with respect to the
        parameters of the model summed over observations.
    mom_incl : ndarray, 2-D (nobs, k_params)
        moment conditions that where use in estimation, assumed to be zero
        This is score_obs in the case of (Q)MLE
    mom_incl_deriv : ndarray, 2-D, square (k_params, k_params)
        derivative of moment conditions of estimator summed over observations
        This is the information matrix or Hessian in the case of (Q)MLE.
    var_mom_all : None, or ndarray, 2-D, (k, k) with k = k_constraints + k_params
        Expected product or variance of the joint (column_stacked) moment
        conditions. The stacking should have the variance of the moment
        conditions under test in the first k_constraint rows and columns.
        If it is not None, then it will be estimated based on cov_type.
        I think: This is the Hessian of the extended or alternative model
        under full MLE and score test assuming information matrix identity holds.

    Returns
    -------
    results


    Notes
    -----
    TODO: cov_type other than OPG is missing
    initial implementation based on Cameron Trived countbook 1998 p.48, p.56

    also included: mom_incl can be None if expected mom_test_deriv is zero.

    References
    ----------
    Cameron and Trivedi 1998 count book
    Wooldridge ???
    Pagan and Vella 1989

    """
    if cov_type != 'OPG':
        raise NotImplementedError

    k_constraints = mom_test.shape[1]

    if mom_incl is None:
        # assume mom_test_deriv is zero, do not include effect of mom_incl
        if var_mom_all is None:
            var_cm = mom_test.T.dot(mom_test)
        else:
            var_cm = var_mom_all

    else:
        # take into account he effect of parameter estimates on mom_test
        if var_mom_all is None:
            mom_all = np.column_stack((mom_test, mom_incl))
            # TODO: replace with inner sandwich covariance estimator
            var_mom_all = mom_all.T.dot(mom_all)

        tmp = mom_test_deriv.dot(np.linalg.pinv(mom_incl_deriv))
        h = np.column_stack((np.eye(k_constraints), -tmp))

        var_cm = h.dot(var_mom_all.dot(h.T))

    # calculate test results with chisquare
    var_cm_inv = np.linalg.pinv(var_cm)
    mom_test_sum = mom_test.sum(0)
    statistic = mom_test_sum.dot(var_cm_inv.dot(mom_test_sum))
    pval = stats.chi2.sf(statistic, k_constraints)

    # normal test of individual components
    se = np.sqrt(np.diag(var_cm))
    tvalues = mom_test_sum / se
    pvalues = stats.norm.sf(np.abs(tvalues))

    res = ResultsGeneric(var_cm=var_cm,
                         stat_cmt=statistic,
                         pval_cmt=pval,
                         tvalues=tvalues,
                         pvalues=pvalues)

    return res


def conditional_moment_test_regression(mom_test, mom_test_deriv=None,
                                    mom_incl=None, mom_incl_deriv=None,
                                    var_mom_all=None, demean=False,
                                    cov_type='OPG', cov_kwds=None):
    """generic conditional moment test based artificial regression

    this is very experimental, no options implemented yet

    so far
    OPG regression, or
    artificial regression with Robust Wald test

    The latter is (as far as I can see) the same as an overidentifying test
    in GMM where the test statistic is the value of the GMM objective function
    and it is assumed that parameters were estimated with optimial GMM, i.e.
    the weight matrix equal to the expectation of the score variance.

    """
    # so far coded from memory
    nobs, k_constraints = mom_test.shape

    endog = np.ones(nobs)
    if mom_incl is not None:
        ex = np.column_stack((mom_test, mom_incl))
    else:
        ex = mom_test
    if demean:
        ex -= ex.mean(0)
    if cov_type == 'OPG':
        res = OLS(endog, ex).fit()

        statistic = nobs * res.rsquared
        pval = stats.chi2.sf(statistic, k_constraints)
    else:
        res = OLS(endog, ex).fit(cov_type=cov_type, cov_kwds=cov_kwds)
        tres = res.wald_test(np.eye(ex.shape[1]))
        statistic = tres.statistic
        pval = tres.pvalue

    return statistic, pval



class CMTNewey(object):
    """generic moment test for GMM

    This is a class to calculate and hold the various results

    This is based on Newey 1985 on GMM.
    Lemma 1:
    Theorem 1


    The main method is `chisquare` which returns the result of the
    conditional moment test.

    Warning: name of class and methods will likely be changed

    Parameters
    ----------
    moments : ndarray, 1-D
        moments that are tested to be zero. They don't need to be derived
        from a likelihood function.
    moments_deriv : ndarray
        derivative of the moment function with respect to the parameters that
        are estimated
    cov_moments : ndarray
        An estimate for the joint (expected) covariance of all moments. This
        can be a heteroscedasticity or correlation robust covariance estimate,
        i.e. the inner part of a sandwich covariance.
    weights : ndarray
        Weights used in the GMM estimation.
    transf_mt : ndarray
        This defines the test moments where `transf_mt` is the matrix that
        defines a Linear combination of moments that have expected value equal
        to zero under the Null hypothesis.


    Notes
    -----
    The one letter names in Newey 1985 are

    moments, g :
    cov_moments, V :
    moments_deriv, H :
    weights, W :
    transf_mt, L :
        linear transformation to get the test condition from the moments

    not used, add as argument to methods or __init__?
    K cov for misspecification
    or mispecification_deriv

    This follows the GMM version in Newey 1985a, not the MLE version in
    Newey 1985b. Newey uses the generalized information matrix equality in the
    MLE version Newey (1985b).

    Newey 1985b Lemma 1 does not impose correctly specified likelihood, but
    assumes it in the following. Lemma 1 in both articles are essentially the
    same assuming D = H' W.


    References
    ----------
    - Newey 1985a, Generalized Method of Moment specification testing,
      Journal of Econometrics
    - Newey 1985b, Maximum Likelihood Specification Testing and Conditional
      Moment Tests, Econometrica


    """

    def __init__(self, moments, cov_moments, moments_deriv,
                 weights, transf_mt):

        self.moments = moments
        self.cov_moments = cov_moments
        self.moments_deriv = moments_deriv
        self.weights = weights
        self.transf_mt = transf_mt

        # derived quantities
        self.moments_constraint = transf_mt.dot(moments)
        self.htw = moments_deriv.T.dot(weights)   # H'W

        # TODO check these
        self.k_moments = self.moments.shape[-1]  # in this case only 1-D
        # assuming full rank of L'
        self.k_constraints = self.transf_mt.shape[0]


    @cache_readonly
    def asy_transf_params(self):

        moments_deriv = self.moments_deriv  # H
        #weights = self.weights  # W

        htw = self.htw  # moments_deriv.T.dot(weights)   # H'W
        res = np.linalg.solve(htw.dot(moments_deriv), htw)
        #res = np.linalg.pinv(htw.dot(moments_deriv)).dot(htw)
        return -res


    @cache_readonly
    def project_w(self):
        # P_w = I - H (H' W H)^{-1} H' W
        moments_deriv = self.moments_deriv  # H

        res = moments_deriv.dot(self.asy_transf_params)
        res += np.eye(res.shape[0])
        return res


    @cache_readonly
    def asy_transform_mom_constraints(self):
        # L P_w
        res = self.transf_mt.dot(self.project_w)
        return res

    @cache_readonly
    def asy_cov_moments(self):
        """

        `sqrt(T) * g_T(b_0) asy N(K delta, V)`

        mean is not implemented,
        V is the same as cov_moments in __init__ argument
        """

        return self.cov_moments

    @cache_readonly
    def cov_mom_constraints(self):

        # linear transformation
        transf = self.asy_transform_mom_constraints

        return transf.dot(self.asy_cov_moments).dot(transf.T)


    @cache_readonly
    def rank_cov_mom_constraints(self):
        return np.linalg.matrix_rank(self.cov_mom_constraints)


    def ztest(self):
        """statistic, p-value and degrees of freedom of separate moment test

        currently two sided test only

        TODO: This can use generic ztest/ttest features and return
        ContrastResults
        """
        diff = self.moments_constraint
        bse = np.sqrt(np.diag(self.cov_mom_constraints))

        # Newey uses a generalized inverse
        stat = diff / bse
        pval = stats.norm.sf(np.abs(stat))*2
        return stat, pval


    @cache_readonly
    def chisquare(self):
        """statistic, p-value and degrees of freedom of joint moment test
        """
        diff = self.moments_constraint
        cov = self.cov_mom_constraints

        # Newey uses a generalized inverse
        stat = diff.T.dot(np.linalg.pinv(cov).dot(diff))
        df = self.rank_cov_mom_constraints
        from scipy import stats
        pval = stats.chi2.sf(stat, df)  # Theorem 1
        return stat, pval, df


class CMTTauchen(object):
    """generic moment tests or conditional moment tests for Quasi-MLE

    This is a generic class based on Tauchen 1985

    The main method is `chisquare` which returns the result of the
    conditional moment test.

    Warning: name of class and of methods will likely be changed


    Parameters
    ----------
    score : ndarray, 1-D
        moment condition used in estimation, score of log-likelihood function
    score_deriv : ndarray
        derivative of score function with respect to the parameters that are
        estimated. This is the Hessian in quasi-maximum likelihood
    moments : ndarray, 1-D
        moments that are tested to be zero. They don't need to be derived
        from a likelihood function.
    moments_deriv : ndarray
        derivative of the moment function with respect to the parameters that
        are estimated
    cov_moments : ndarray
        An estimate for the joint (expected) covariance of score and test
        moments. This can be a heteroscedasticity or correlation robust
        covariance estimate, i.e. the inner part of a sandwich covariance.

    """

    def __init__(self, score, score_deriv, moments, moments_deriv, cov_moments):
        self.score = score
        self.score_deriv = score_deriv
        self.moments = moments
        self.moments_deriv = moments_deriv
        self.cov_moments_all = cov_moments

        self.k_moments_test = moments.shape[-1]
        self.k_params = score.shape[-1]
        self.k_moments_all = self.k_params + self.k_moments_test


    @cache_readonly
    def cov_params_all(self):
        m_deriv = np.zeros((self.k_moments_all, self.k_moments_all))
        m_deriv[:self.k_params, :self.k_params] = self.score_deriv
        m_deriv[self.k_params:, :self.k_params] = self.moments_deriv
        m_deriv[self.k_params:, self.k_params:] = np.eye(self.k_moments_test)

        m_deriv_inv = np.linalg.inv(m_deriv)
        cov = m_deriv_inv.dot(self.cov_moments_all.dot(m_deriv_inv.T)) # K_inv J K_inv
        return cov

    @cache_readonly
    def cov_mom_constraints(self):
        return self.cov_params_all[self.k_params:, self.k_params:]


    @cache_readonly
    def rank_cov_mom_constraints(self):
        return np.linalg.matrix_rank(self.cov_mom_constraints)

    # TODO: not DRY, just copied from CMTNewey
    def ztest(self):
        """statistic, p-value and degrees of freedom of separate moment test

        currently two sided test only

        TODO: This can use generic ztest/ttest features and return
        ContrastResults
        """
        diff = self.moments_constraint
        bse = np.sqrt(np.diag(self.cov_mom_constraints))

        # Newey uses a generalized inverse
        stat = diff / bse
        pval = stats.norm.sf(np.abs(stat))*2
        return stat, pval


    @cache_readonly
    def chisquare(self):
        """statistic, p-value and degrees of freedom of joint moment test
        """
        diff = self.moments #_constraints
        cov = self.cov_mom_constraints

        # Newey uses a generalized inverse, we use it also here
        stat = diff.T.dot(np.linalg.pinv(cov).dot(diff))
        #df = self.k_moments_test
        # We allow for redundant mom_constraints:
        df = self.rank_cov_mom_constraints
        from scipy import stats
        pval = stats.chi2.sf(stat, df)
        return stat, pval, df
