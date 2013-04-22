# -*- coding: utf-8 -*-
"""Base classes for statistical test results

Created on Mon Apr 22 14:03:21 2013

Author: Josef Perktold
"""

import numpy as np

class AllPairsResults(object):
    '''Results class for pairwise comparisons, based on p-values

    Parameter
    ---------
    pvals_raw : array_like, 1-D
        p-values from a pairwise comparison test
    all_pairs : list of tuples
        list of indices, one pair for each comparison
    multitest_method : string
        method that is used by default for p-value correction. This is used
        as default by the methods like if the multiple-testing method is not
        specified as argument.
    levels : None or list of strings
        optional names of the levels or groups
    n_levels : None or int
        If None, then the number of levels or groups is inferred from the
        other arguments. It can be explicitly specified, if it is not a
        standard all pairs comparison.

    Notes
    -----
    It should be possible to use this for other pairwise comparisons, for
    example all others compared to a control (Dunnet).


    '''


    def __init__(self, pvals_raw, all_pairs, multitest_method='hs',
                 levels=None, n_levels=None):
        self.pvals_raw = pvals_raw
        self.all_pairs = all_pairs
        if n_levels is None:
            # for all_pairs nobs*(nobs-1)/2
            #self.n_levels = (1. + np.sqrt(1 + 8 * len(all_pairs))) * 0.5
            self.n_levels = np.max(all_pairs) + 1
        else:
            self.n_levels = n_levels

        self.multitest_method = multitest_method
        self.levels = levels
        if levels is None:
            self.all_pairs_names = ['%r' % (pairs,) for pairs in all_pairs]
        else:
            self.all_pairs_names = ['%s-%s' % (levels[pairs[0]],
                                               levels[pairs[1]])
                                               for pairs in all_pairs]

    def pval_corrected(self, method=None):
        import statsmodels.stats.multitest as smt
        if method is None:
            method = self.multitest_method
        #TODO: breaks with method=None
        return smt.multipletests(self.pvals_raw, method=method)[1]

    def __str__(self):
        return self.summary()

    def pval_table(self):
        k = self.n_levels
        pvals_mat = np.zeros((k, k))
        # if we don't assume we have all pairs
        pvals_mat[zip(*self.all_pairs)] = self.pval_corrected()
        #pvals_mat[np.triu_indices(k, 1)] = self.pval_corrected()
        return pvals_mat

    def summary(self):
        import statsmodels.stats.multitest as smt
        maxlevel = max((len(ss) for ss in self.all_pairs_names))

        text = 'Corrected p-values using %s p-value correction\n\n' % \
                        smt.multitest_methods_names[self.multitest_method]
        text += 'Pairs' + (' ' * (maxlevel - 5 + 1)) + 'p-values\n'
        text += '\n'.join(('%s  %6.4g' % (pairs, pv) for (pairs, pv) in
                zip(self.all_pairs_names, self.pval_corrected())))
        return text

