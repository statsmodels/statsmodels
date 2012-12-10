# -*- coding: utf-8 -*-
"""

Created on Mon Dec 10 09:18:14 2012

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from statsmodels.stats.inter_rater import fleiss_kappa, cohens_kappa


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

table1 = table0[:, 1:-1]

table10 = [[0, 4, 1],
           [0, 8, 0],
           [0, 1, 5]]

def test_fleiss_kappa():
    #currently only example from Wikipedia page
    kappa_wp = 0.210
    assert_almost_equal(fleiss_kappa(table1), kappa_wp, decimal=3)


class CheckCohens(object):

    def test_results(self):
        res = self.res
        res2 = self.res2

        res_ = [res.kappa, res.std_kappa, res.kappa_low, res.kappa_upp, res.std_kappa0,
                res.z_value, res.pvalue_one_sided, res.pvalue_two_sided]

        assert_almost_equal(res_, res2, decimal=4)
        assert_equal(str(res), self.res_string)


class UnweightedCohens(CheckCohens):
    #comparison to printout of a SAS example
    def __init__(self):
        #temporary: res instance is at last position
        self.res = cohens_kappa(table10)
        res10_sas = [0.4842, 0.1380, 0.2137, 0.7547]
        res10_sash0 = [0.1484, 3.2626, 0.0006, 0.0011]  #for test H0:kappa=0
        self.res2 = res10_sas + res10_sash0 #concatenate

        self.res_string = '''\
                  Simple Kappa Coefficient
              --------------------------------
              Kappa                     0.4842
              ASE                       0.1380
              95% Lower Conf Limit      0.2137
              95% Upper Conf Limit      0.7547

                 Test of H0: Simple Kappa = 0

              ASE under H0              0.1484
              Z                         3.2626
              One-sided Pr >  Z         0.0006
              Two-sided Pr > |Z|        0.0011''' + '\n'

    def test_option(self):
        kappa = cohens_kappa(table10, return_results=False)
        assert_almost_equal(kappa, self.res2[0], decimal=4)

class TestWeightedCohens(CheckCohens):
    #comparison to printout of a SAS example
    def __init__(self):
        #temporary: res instance is at last position
        self.res = cohens_kappa(table10, weights=[0, 1, 2])
        res10w_sas = [0.4701, 0.1457, 0.1845, 0.7558]
        res10w_sash0 = [0.1426, 3.2971, 0.0005, 0.0010]  #for test H0:kappa=0
        self.res2 = res10w_sas + res10w_sash0 #concatenate

        self.res_string = '''\
                  Weighted Kappa Coefficient
              --------------------------------
              Kappa                     0.4701
              ASE                       0.1457
              95% Lower Conf Limit      0.1845
              95% Upper Conf Limit      0.7558

                 Test of H0: Weighted Kappa = 0

              ASE under H0              0.1426
              Z                         3.2971
              One-sided Pr >  Z         0.0005
              Two-sided Pr > |Z|        0.0010''' + '\n'

    def test_option(self):
        kappa = cohens_kappa(table10, weights=[0, 1, 2], return_results=False)
        assert_almost_equal(kappa, self.res2[0], decimal=4)


def test_cohenskappa_weights():
    #some tests for equivalent results with different options
    np.random.seed(9743678)
    table = np.random.randint(0, 10, size=(5,5)) + 5*np.eye(5)

    #example aggregation, 2 groups of levels
    mat = np.array([[1,1,1, 0,0],[0,0,0,1,1]])
    table_agg = np.dot(np.dot(mat, table), mat.T)
    res1 = cohens_kappa(table, weights=np.arange(5) > 2, wt='linear')
    res2 = cohens_kappa(table_agg, weights=np.arange(2), wt='linear')
    assert_almost_equal(res1.kappa, res2.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res2.var_kappa, decimal=14)

    #equivalence toeplitz with linear for special cases
    res1 = cohens_kappa(table, weights=2*np.arange(5), wt='linear')
    res2 = cohens_kappa(table, weights=2*np.arange(5), wt='toeplitz')
    res3 = cohens_kappa(table, weights=res1.weights[0], wt='toeplitz')
    #2-Dim weights
    res4 = cohens_kappa(table, weights=res1.weights)

    assert_almost_equal(res1.kappa, res2.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res2.var_kappa, decimal=14)

    assert_almost_equal(res1.kappa, res3.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res3.var_kappa, decimal=14)

    assert_almost_equal(res1.kappa, res4.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res4.var_kappa, decimal=14)

    #equivalence toeplitz with quadratic for special cases
    res1 = cohens_kappa(table, weights=5*np.arange(5)**2, wt='toeplitz')
    res2 = cohens_kappa(table, weights=5*np.arange(5), wt='quadratic')
    assert_almost_equal(res1.kappa, res2.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res2.var_kappa, decimal=14)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x'#, '--pdb-failures'
                        ], exit=False)
