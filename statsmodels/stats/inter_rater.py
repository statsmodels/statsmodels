# -*- coding: utf-8 -*-
"""inter rater agreement

Created on Thu Dec 06 22:57:56 2012
Author: Josef Perktold
License: BSD-3

References
----------
Wikipedia
SAS-Manual

"""


import numpy as np
from numpy.testing import assert_almost_equal

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

table0 = np.asarray('''\
1 	0 	0 	0 	0 	14 	1.000
2 	0 	2 	6 	4 	2 	0.253
3 	0 	0 	3 	5 	6 	0.308
4 	0 	3 	9 	2 	0 	0.440
5 	2 	2 	8 	1 	1 	0.330
6 	7 	7 	0 	0 	0 	0.462
7 	3 	2 	6 	3 	0 	0.242
8 	2 	5 	3 	2 	2 	0.176
9 	6 	5 	2 	1 	0 	0.286
10 	0 	2 	2 	3 	7 	0.286'''.split(), float).reshape(10,-1)


Total = np.asarray("20 	28 	39 	21 	32".split('\t'), int)
Pj = np.asarray("0.143 	0.200 	0.279 	0.150 	0.229".split('\t'), float)
kappa_wp = 0.210
table1 = table0[:, 1:-1]


def fleiss_kappa(table):
    '''

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



def cohens_kappa(table, weights=None, return_var=True, wt=None):
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
            from the one dimensional weights. (maximum weight in this case
            should be less or equal to one)
            TODO: test variance estimate for this case

    Returns
    -------
    kappa or results
        TODO: change to this
        If return_kappa is False (default), then a results instance is returned with all
        statistics.
        If return_kappa is true, then only kappa is calculated and returned.

    Notes
    -----
    There are two conflicting definitions of the weight matrix.
    Wikipedia versus SAS manual.

    TODO: need more checks and consistency for weight options

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
        kappa = (agree / nobs - agree_exp) / (1 - agree_exp)

        if return_var:
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
        weights = np.asarray(weights, float)
        if weights.ndim == 1:
            if wt is 'ca':
                weights = np.abs(weights[:, None] - weights) /  \
                           (weights[-1] - weights[0])
            elif wt is 'fs':
                weights = (weights[:, None] - weights)**2 /  \
                           (weights[-1] - weights[0])**2
            else:
                #assume toeplitz structure
                from scipy.linalg import toeplitz
                #weights = toeplitz(np.arange(table.shape[0]))
                weights = toeplitz(weights)
        else:
            rows, cols = table.shape
            if (table.shape != weights.shape):
                raise ValueError('weights are not square')
        kappa = 1 - (weights * table).sum() / nobs / (weights * prob_exp).sum()
        #TODO: add var_kappa for weighted version
        if return_var:
            var_kappa = np.nan
            var_kappa0 = np.nan
            #switch to SAS manula weights, problem if user specifies weights
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

    if return_var:
        return kappa, kappa_max, weights, var_kappa, var_kappa0
    else:
        return kappa, kappa_max, weights

print fleiss_kappa(table1)
table4 = np.array([[20,5], [10, 15]])
print 'res', cohens_kappa(table4), 0.4 #wikipedia

table5 = np.array([[45, 15], [25, 15]])
print 'res', cohens_kappa(table5), 0.1304 #wikipedia

table6 = np.array([[25, 35], [5, 35]])
print 'res', cohens_kappa(table6), 0.2593  #wikipedia
print 'res', cohens_kappa(table6, weights=np.arange(2)), 0.2593  #wikipedia
t7 = np.array([[16, 18, 28],
               [10, 27, 13],
               [28, 20, 24]])
print cohens_kappa(t7, weights=[0, 1, 2])

table8 = np.array([[25, 35], [5, 35]])
print 'res', cohens_kappa(table8)

#SAS example from http://www.john-uebersax.com/stat/saskappa.htm
'''
   Statistic          Value       ASE     95% Confidence Limits
   ------------------------------------------------------------
   Simple Kappa      0.3333    0.0814       0.1738       0.4929
   Weighted Kappa    0.2895    0.0756       0.1414       0.4376
'''
t9 = [[0,  0,  0],
      [5, 16,  3],
      [8, 12, 28]]
res9 = cohens_kappa(t9)
print 'res', res9
print 'res', cohens_kappa(t9, weights=[0, 1, 2])


#check max kappa, constructed by hand, same marginals
table6a = np.array([[30, 30], [0, 40]])
res = cohens_kappa(table6a)
assert res[0] == res[1]
print np.divide(*cohens_kappa(table6)[:2])
print np.divide(*cohens_kappa(table6a)[:2])


from scipy import stats
delta = stats.norm.isf(0.025) * np.sqrt(res9[3])
print res9[0] - delta, res9[0] + delta

res9_sas = [0.3333, 0.0814, 0.1738, 0.4929]
res9_ = res9[0], np.sqrt(res9[3]), res9[0] - delta, res9[0] + delta
assert_almost_equal(res9_, res9_sas, decimal=4)

table10 = [[0, 4, 1],
           [0, 8, 0],
           [0, 1, 5]]
res10 = cohens_kappa(table10)
print 'res10', res10
print np.sqrt(res10[-1])

res10_sas = [0.4842, 0.1380, 0.2137, 0.7547]
delta = stats.norm.isf(0.025) * np.sqrt(res10[3])
print res10[0] - delta, res10[0] + delta
res10_ = res10[0], np.sqrt(res10[3]), res10[0] - delta, res10[0] + delta
assert_almost_equal(res10_, res10_sas, decimal=4)

res10w = cohens_kappa(table10, weights=[0, 1, 2])
res10w_sas = [0.4701, 0.1457, 0.1845, 0.7558]
res10w_sash0 = [0.1426, 3.2971, 0.0005]  #for test H0:kappa=0
delta = stats.norm.isf(0.025) * np.sqrt(res10w[3])
print res10w[0] - delta, res10w[0] + delta
res10w_ = res10w[0], np.sqrt(res10w[3]), res10w[0] - delta, res10w[0] + delta
assert_almost_equal(res10w_, res10w_sas, decimal=4)
#assert_almost_equal(???, res10w_sash0, decimal=4)
print np.sqrt(res10w[3:])

zval = res10w[0]/np.sqrt(res10w[-1])
pval = stats.norm.sf(zval)
pval_two_sided = pval * 2

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


class KappaResults(dict):

    def __init__(self, **kwds):
        self.update(kwds)
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
        return kappa_template % self


kappa, kappa_max, weights, var_kappa, var_kappa0 = res10
k10 = KappaResults( kind='Simple',
                    kappa=kappa,
                    kappa_max=kappa_max,
                    weights=weights,
                    var_kappa=var_kappa,
                    var_kappa0=var_kappa0
                    )



'''SAS result for table10

                  Simple Kappa Coefficient
              --------------------------------
              Kappa                     0.4842
              ASE                       0.1380
              95% Lower Conf Limit      0.2137
              95% Upper Conf Limit      0.7547

                  Test of H0: Kappa = 0

              ASE under H0              0.1484
              Z                         3.2626
              One-sided Pr >  Z         0.0006
              Two-sided Pr > |Z|        0.0011

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
