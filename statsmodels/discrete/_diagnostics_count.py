# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:53:45 2017

Author: Josef Perktold
"""

import numpy as np
from scipy import stats

from statsmodels.discrete.discrete_model import Poisson
from statsmodels.regression.linear_model import OLS


def _combine_bins(edge_index, x):
    """group columns into bins using sum

    This is mainly a helper function for combining probabilities into cells.
    It similar to `np.add.reduceat(x, edge_index, axis=-1)` except for the
    treatment of the last index and last cell.

    Parameters
    ----------
    edge_index : array_like
         This defines the (zero-based) indices for the columns that are be
         combined. Each index in `edge_index` except the last is the starting
         index for a bin. The largest index in a bin is the next edge_index-1.
    x : 1d or 2d array
        array for which columns are combined. If x is 1-dimensional that it
        will be treated as a 2-d row vector.

    Returns
    -------
    x_new : ndarray


    Examples
    --------
    >>> dia.combine_bins([0,1,5], np.arange(4))
    (array([0, 6]), array([1, 4]))

    this aggregates to two bins with the sum of 1 and 4 elements
    >>> np.arange(4)[0].sum()
    0
    >>> np.arange(4)[1:5].sum()
    6

    If the rightmost index is smaller than len(x)+1, then the remaining
    columns will not be included.

    >>> dia.combine_bins([0,1,3], np.arange(4))
    (array([0, 3]), array([1, 2]))
    """
    x = np.asarray(x)
    if x.ndim == 1:
        is_1d = True
        x = x[None, :]
    else:
        is_1d = False
    xli = []
    kli = []
    for bin_idx in range(len(edge_index) - 1):
        i, j = edge_index[bin_idx : bin_idx + 2]
        xli.append(x[:, i:j].sum(1))
        kli.append(j - i)

    x_new = np.column_stack(xli)
    if is_1d:
        x_new = x_new.squeeze()
    return x_new, np.asarray(kli)


def plot_probs(freq, probs_predicted, label='predicted', upp_xlim=None,
               fig=None):
    """diagnostic plots for comparing two lists of discrete probabilities

    Parameters
    ----------
    freq, probs_predicted : nd_arrays
        two arrays of probabilities, this can be any probabilities for
        the same events, default is designed for comparing predicted
        and observed probabilities
    label : str or tuple
        If string, then it will be used as the label for probs_predicted and
        "freq" is used for the other probabilities.
        If label is a tuple of strings, then the first is they are used as
        label for both probabilities

    upp_xlim : None or int
        If it is not None, then the xlim of the first two plots are set to
        (0, upp_xlim), otherwise the matplotlib default is used
    fig : None or matplotlib figure instance
        If fig is provided, then the axes will be added to it in a (3,1)
        subplots, otherwise a matplotlib figure instance is created

    Returns
    -------
    Figure
        The figure contains 3 subplot with probabilities, cumulative
        probabilities and a PP-plot
    """

    if isinstance(label, list):
        label0, label1 = label
    else:
        label0, label1 = 'freq', label

    if fig is None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,12))
    ax1 = fig.add_subplot(311)
    ax1.plot(freq, '-o', label=label0)
    ax1.plot(probs_predicted, '-d', label=label1)
    if upp_xlim is not None:
        ax1.set_xlim(0, upp_xlim)
    ax1.legend()
    ax1.set_title('probabilities')

    ax2 = fig.add_subplot(312)
    ax2.plot(np.cumsum(freq), '-o', label=label0)
    ax2.plot(np.cumsum(probs_predicted), '-d', label=label1)
    if upp_xlim is not None:
        ax2.set_xlim(0, upp_xlim)
    ax2.legend()
    ax2.set_title('cumulative probabilities')

    ax3 = fig.add_subplot(313)
    ax3.plot(np.cumsum(probs_predicted), np.cumsum(freq), 'o')
    ax3.plot(np.arange(len(freq)) / len(freq), np.arange(len(freq)) / len(freq))
    ax3.set_title('PP-plot')
    ax3.set_xlabel(label1)
    ax3.set_ylabel(label0)
    return fig


def test_chisquare_prob(results, probs, bin_edges=None, method=None):
    """
    chisquare test for predicted probabilities using cmt-opg

    Parameters
    ----------
    results : results instance
        Instance of a count regression results
    probs : ndarray
        Array of predicted probabilities with observations
        in rows and event counts in columns
    bin_edges : None or array
        intervals to combine several counts into cells
        see combine_bins

    Returns
    -------
    (api not stable, replace by test-results class)
    statistic : float
        chisquare statistic for tes
    p-value : float
        p-value of test
    df : int
        degrees of freedom for chisquare distribution
    extras : ???
        currently returns a tuple with some intermediate results
        (diff, res_aux)

    Notes
    -----

    Status : experimental, no verified unit tests, needs to be generalized
    currently only OPG version with auxiliary regression is implemented


    Assumes counts are np.arange(probs.shape[1]), i.e. consecutive
    integers starting at zero.

    Auxiliary regression drops the last column of binned probs to avoid
    that probabilities sum to 1.
    """
    res = results
    score_obs = results.model.score_obs(results.params)
    d_ind = (res.model.endog[:, None] == np.arange(probs.shape[1])).astype(int)
    if bin_edges is not None:
        d_ind_bins, k_bins = _combine_bins(bin_edges, d_ind)
        probs_bins, k_bins = _combine_bins(bin_edges, probs)
    else:
        d_ind_bins, k_bins = d_ind, d_ind.shape[1]
        probs_bins = probs
    diff1 = d_ind_bins - probs_bins
    #diff2 = (1 - d_ind.sum(1)) - (1 - probs_bins.sum(1))
    x_aux = np.column_stack((score_obs, diff1[:, :-1])) #, diff2))
    nobs = x_aux.shape[0]
    res_aux = OLS(np.ones(nobs), x_aux).fit()

    chi2_stat = nobs * (1 - res_aux.ssr / res_aux.uncentered_tss)
    df = res_aux.model.rank - score_obs.shape[1]
    if df < k_bins - 1:
        # not a problem in general, but it can be for OPG version
        import warnings
        warnings.warn('auxiliary model is rank deficient')
    extras = (diff1, res_aux)
    return chi2_stat, stats.chi2.sf(chi2_stat, df), df, extras


def test_poisson_zeroinflation(results_poisson, exog_infl=None):
    """score test for zero inflation or deflation in Poisson

    This implements Jansakul and Hinde 2009 score test
    for excess zeros against a zero modified Poisson
    alternative. They use a linear link function for the
    inflation model to allow for zero deflation.

    Parameters
    ----------
    results_poisson: results instance
        The test is only valid if the results instance is a Poisson
        model.
    exog_infl : ndarray
        Explanatory variables for the zero inflated or zero modified
        alternative. I exog_infl is None, then the inflation
        probability is assumed to be constant.

    Returns
    -------
    score test results based on chisquare distribution

    Notes
    -----
    This is a score test based on the null hypothesis that
    the true model is Poisson. It will also reject for
    other deviations from a Poisson model if those affect
    the zero probabilities, e.g. in the direction of
    excess dispersion as in the Negative Binomial
    or Generalized Poisson model.
    Therefore, rejection in this test does not imply that
    zero-inflated Poisson is the appropriate model.

    Status: experimental, no verified unit tests,

    TODO: If the zero modification probability is assumed
    to be constant under the alternative, then we only have
    a scalar test score and we can use one-sided tests to
    distinguish zero inflation and deflation from the
    two-sided deviations. (The general one-sided case is
    difficult.)

    References
    ----------
    Jansakul and Hinde 2009
    """
    if not isinstance(results_poisson.model, Poisson):
        # GLM Poisson would be also valid, not tried
        import warnings
        warnings.warn('Test is only valid if model is Poisson')

    nobs = results_poisson.model.endog.shape[0]

    if exog_infl is None:
        exog_infl = np.ones((nobs, 1))


    endog = results_poisson.model.endog
    exog = results_poisson.model.exog

    mu = results_poisson.predict()
    prob_zero = np.exp(-mu)

    cov_poi = results_poisson.cov_params()
    cross_derivative = (exog_infl.T * (-mu)).dot(exog).T
    cov_infl = (exog_infl.T * ((1 - prob_zero) / prob_zero)).dot(exog_infl)
    score_obs_infl = exog_infl * (((endog == 0) - prob_zero) / prob_zero)[:,None]
    #score_obs_infl = exog_infl * ((endog == 0) * (1 - prob_zero) / prob_zero - (endog>0))[:,None] #same
    score_infl = score_obs_infl.sum(0)
    cov_score_infl = cov_infl - cross_derivative.T.dot(cov_poi).dot(cross_derivative)
    cov_score_infl_inv = np.linalg.pinv(cov_score_infl)
    statistic = score_infl.dot(cov_score_infl_inv).dot(score_infl)
    df2 = np.linalg.matrix_rank(cov_score_infl)  # more general, maybe not needed
    df = exog_infl.shape[1]
    pvalue = stats.chi2.sf(statistic, df)
    return statistic, pvalue, df, df2


def test_poisson_zeroinflation_brock(results_poisson):
    """score test for zero modification in Poisson, special case

    This assumes that the Poisson model has a constant and that
    the zero modification probability is constant.

    This is a special case of test_poisson_zeroinflation derived by
    van den Brock 1995.

    The test reports two sided and one sided alternatives based on
    the normal distribution of the test statistic.
    """

    mu = results_poisson.predict()
    prob_zero = np.exp(-mu)
    endog = results_poisson.model.endog
    nobs = len(endog)

    score =  ((endog == 0) / prob_zero).sum() - nobs
    var_score = (1 / prob_zero).sum() - nobs - endog.sum()
    statistic = score / np.sqrt(var_score)
    pvalue_two = 2 * stats.norm.sf(np.abs(statistic))
    pvalue_upp = stats.norm.sf(statistic)
    pvalue_low = stats.norm.cdf(statistic)
    return statistic, pvalue_two, pvalue_upp, pvalue_low, stats.chi2.sf(statistic**2, 1)
