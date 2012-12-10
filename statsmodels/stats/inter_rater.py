# -*- coding: utf-8 -*-
"""Inter Rater Agreement

contains
--------
fleiss_kappa
cohens_kappa

Created on Thu Dec 06 22:57:56 2012
Author: Josef Perktold
License: BSD-3

References
----------
Wikipedia: kappa's initially based on these two pages
    http://en.wikipedia.org/wiki/Fleiss%27_kappa
    http://en.wikipedia.org/wiki/Cohen's_kappa
SAS-Manual : formulas for cohens_kappa, especially variances
see also R package irr, not looked at it yet except index

TODO
----
other statistics and tests, in R package irr, SAS has more
inconsistent naming, changed variable names as I added more functionality
convenience functions to create required data format from raw data

"""


import numpy as np
from scipy import stats  #get rid of this? need only norm.sf


class ResultsBunch(dict):

    def __init__(self, **kwds):
        dict.__init__(self, kwds)
        self.__dict__  = self
        self._initialize()

    def __str__(self):
        return self.template % self

def int_ifclose(x, dec=1, width=4):
    '''helper function for creating result string for int or float

    only dec=1 and width=4 is implemented

    Parameters
    ----------
    x : int or float
        value to format
    dec : 1
        number of decimals to print if x is not an integer
    width : 4
        width of string

    Returns
    -------
    xint : int or float
        x is converted to int if it is within 1e-14 of an integer
    x_string : str
        x formatted as string, either '%4d' or '%4.1f'

    '''
    xint = int(round(x))
    if np.max(np.abs(xint - x)) < 1e-14:
        return xint, '%4d' % xint
    else:
        return x, '%4.1f' % x


def fleiss_kappa(table):
    '''Fleiss' kappa multi-rater agreement measure

    Parameters
    ----------
    table : array_like, 2-D
        assumes subjects in rows, and categories in columns

    Returns
    -------
    kappa : float
        Fleiss's kappa statistic for inter rater agreement

    Notes
    -----
    coded from Wikipedia page
    http://en.wikipedia.org/wiki/Fleiss%27_kappa

    no variance or tests yet

    '''
    n_sub, n_cat =  table.shape
    n_total = table.sum()
    n_rater = table.sum(1)
    n_rat = n_rater.max()
    #assume fully ranked
    assert n_total == n_sub * n_rat

    #marginal frequency  of categories
    p_cat = table.sum(0) / n_total

    table2 = table * table
    p_rat = (table2.sum(1) - n_rat) / (n_rat * (n_rat - 1.))
    p_mean = p_rat.mean()

    p_mean_exp = (p_cat*p_cat).sum()

    kappa = (p_mean - p_mean_exp) / (1- p_mean_exp)
    return kappa


def cohens_kappa(table, weights=None, return_results=True, wt=None):
    '''Compute Cohen's kappa with variance and equal-zero test

    Parameters
    ----------
    table : array_like, 2-Dim
        square array with results of two raters, one rater in rows, second
        rater in columns
    weights : array_like
        The interpretation of weights depends on the wt argument.
        If both are None, then the simple kappa is computed.
        see wt for the case when wt is not None
        If weights is two dimensional, then it is directly used as a weight
        matrix. For computing the variance of kappa, the maximum of the
        weights is assumed to be smaller or equal to one.
        TODO: fix conflicting definitions in the 2-Dim case for
    wt : None or string
        If wt and weights are None, then the simple kappa is computed.
        If wt is given, but weights is None, then the weights are set to
        be [0, 1, 2, ..., k].
        If weights is a one-dimensional array, then it is used to construct
        the weight matrix given the following options.

        wt in ['linear', 'ca' or None] : use linear weights, Cicchetti-Allison
            actual weights are linear in the score "weights" difference
        wt in ['quadratic', 'fc'] : use linear weights, Fleiss-Cohen
            actual weights are squared in the score "weights" difference
        wt = 'toeplitz' : weight matrix is constructed as a toeplitz matrix
            from the one dimensional weights.

    return_results : bool
        If True (default), then an instance of KappaResults is returned.
        If False, then only kappa is computed and returned.

    Returns
    -------
    results or kappa
        If return_results is True (default), then a results instance with all
        statistics is returned
        If return_results is False, then only kappa is calculated and returned.

    Notes
    -----
    There are two conflicting definitions of the weight matrix, Wikipedia
    versus SAS manual. However, the computation are invariant to rescaling
    of the weights matrix, so there is no difference in the results.

    Weights for 'linear' and 'quadratic' are interpreted as scores for the
    categories, the weights in the computation are based on the pairwise
    difference between the scores.
    Weights for 'toeplitz' are a interpreted as weighted distance. The distance
    only depends on how many levels apart two entries in the table are but
    not on the levels themselves.

    example:

    weights = '0, 1, 2, 3' and wt is either linear or toeplitz means that the
    weighting only depends on the simple distance of levels.

    weights = '0, 0, 1, 1' and wt = 'linear' means that the first two levels
    are zero distance apart and the same for the last two levels. This is
    the sampe as forming two aggregated levels by merging the first two and
    the last two levels, respectively.

    weights = [0, 1, 2, 3] and wt = 'quadratic' is the same as squaring these
    weights and using wt = 'toeplitz'.

    References
    ----------
    Wikipedia
    SAS Manual

    '''
    table = np.asarray(table, float) #avoid integer division
    agree = np.diag(table).sum()
    nobs = table.sum()
    probs = table / nobs
    freqs = probs  #TODO: rename to use freqs instead of probs for observed
    probs_diag = np.diag(probs)
    freq_row = table.sum(1) / nobs
    freq_col = table.sum(0) / nobs
    prob_exp = freq_col * freq_row[:,None]
    assert np.allclose(prob_exp.sum(), 1)
    #print prob_exp.sum()
    agree_exp = np.diag(prob_exp).sum() #need for kappa_max
    if weights is None and wt is None:
        kind = 'Simple'
        kappa = (agree / nobs - agree_exp) / (1 - agree_exp)

        if return_results:
            #variance
            term_a = probs_diag * (1 - (freq_row + freq_col) * (1 - kappa))**2
            term_a = term_a.sum()
            term_b = probs * (freq_col[:,None] + freq_row)**2
            d_idx = np.arange(table.shape[0])
            term_b[d_idx, d_idx] = 0   #set diagonal to zero
            term_b = (1 - kappa)**2 * term_b.sum()
            term_c = (kappa - agree_exp * (1-kappa))**2
            var_kappa = (term_a + term_b - term_c) / (1 - agree_exp)**2 / nobs
            #term_c = freq_col * freq_row[:,None] * (freq_col + freq_row[:,None])
            term_c = freq_col * freq_row * (freq_col + freq_row)
            var_kappa0 =( agree_exp + agree_exp**2 - term_c.sum()) / (1 - agree_exp)**2 / nobs

    else:
        #weights follows the Wikipedia definition, not the SAS, which is 1 -
        kind = 'Weighted'
        weights = np.asarray(weights, float)
        if weights.ndim == 1:
            if wt in ['ca', 'linear', None]:
                weights = np.abs(weights[:, None] - weights) /  \
                           (weights[-1] - weights[0])
            elif wt in ['fc', 'quadratic']:
                weights = (weights[:, None] - weights)**2 /  \
                           (weights[-1] - weights[0])**2
            elif wt == 'toeplitz':
                #assume toeplitz structure
                from scipy.linalg import toeplitz
                #weights = toeplitz(np.arange(table.shape[0]))
                weights = toeplitz(weights)
            else:
                raise ValueError('wt option is not known')
        else:
            rows, cols = table.shape
            if (table.shape != weights.shape):
                raise ValueError('weights are not square')
        #this is formula from Wikipedia
        kappa = 1 - (weights * table).sum() / nobs / (weights * prob_exp).sum()
        #TODO: add var_kappa for weighted version
        if return_results:
            var_kappa = np.nan
            var_kappa0 = np.nan
            #switch to SAS manual weights, problem if user specifies weights
            #w is negative in some examples,
            #but weights is scale invariant in examples and rough check of source
            w = 1. - weights
            w_row = (freq_col * w).sum(1)
            w_col = (freq_row[:, None] * w).sum(0)
            agree_wexp = (w * freq_col * freq_row[:,None]).sum()
            term_a = freqs * (w -  (w_col + w_row[:,None]) * (1 - kappa))**2
            fac = 1. / ((1 - agree_wexp)**2 * nobs)
            var_kappa = term_a.sum() - (kappa - agree_wexp * (1 - kappa))**2
            var_kappa *=  fac

            freqse = freq_col * freq_row[:,None]
            var_kappa0 = (freqse * (w -  (w_col + w_row[:,None]))**2).sum() - agree_wexp**2
            var_kappa0 *=  fac

    kappa_max = (np.minimum(freq_row, freq_col).sum() - agree_exp) / (1 - agree_exp)

    if return_results:
        res = KappaResults( kind=kind,
                    kappa=kappa,
                    kappa_max=kappa_max,
                    weights=weights,
                    var_kappa=var_kappa,
                    var_kappa0=var_kappa0
                    )
        return res
    else:
        return kappa


kappa_template = '''\
                  %(kind)s Kappa Coefficient
              --------------------------------
              Kappa                     %(kappa)6.4f
              ASE                       %(std_kappa)6.4f
            %(alpha_ci)s%% Lower Conf Limit      %(kappa_low)6.4f
            %(alpha_ci)s%% Upper Conf Limit      %(kappa_upp)6.4f

                 Test of H0: %(kind)s Kappa = 0

              ASE under H0              %(std_kappa0)6.4f
              Z                         %(z_value)6.4f
              One-sided Pr >  Z         %(pvalue_one_sided)6.4f
              Two-sided Pr > |Z|        %(pvalue_two_sided)6.4f
'''

'''
                   Weighted Kappa Coefficient
              --------------------------------
              Weighted Kappa            0.4701
              ASE                       0.1457
              95% Lower Conf Limit      0.1845
              95% Upper Conf Limit      0.7558

               Test of H0: Weighted Kappa = 0

              ASE under H0              0.1426
              Z                         3.2971
              One-sided Pr >  Z         0.0005
              Two-sided Pr > |Z|        0.0010
'''


class KappaResults(ResultsBunch):

    template = kappa_template

    def _initialize(self):
        if not 'alpha' in self:
            self['alpha'] = 0.025
            self['alpha_ci'] = int_ifclose(100 - 0.025 * 200)[1]

        self['std_kappa'] = np.sqrt(self['var_kappa'])
        self['std_kappa0'] = np.sqrt(self['var_kappa0'])

        self['z_value'] = self['kappa'] / self['std_kappa0']

        self['pvalue_one_sided'] = stats.norm.sf(self['z_value'])
        self['pvalue_two_sided'] = self['pvalue_one_sided'] * 2

        delta = stats.norm.isf(self['alpha']) * self['std_kappa']
        self['kappa_low'] = self['kappa'] - delta
        self['kappa_upp'] = self['kappa'] + delta

    def __str__(self):
        return self.template % self

