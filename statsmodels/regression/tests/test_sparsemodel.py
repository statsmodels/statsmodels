# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 00:06:15 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
import pandas as pd
from scipy import sparse

from statsmodels.regression.linear_model import OLS
from statsmodels.regression.sparse_linear_model import SparseOLS

from numpy.testing import assert_allclose

DEBUG = False

def generate_sample1():
    xcat = np.repeat(np.arange(5), 10)

    df = pd.get_dummies(xcat)  #sparse=True) #sparse requires v 0.16

    exog = sparse.csc_matrix(df.values)
    beta = 1. / np.arange(1, 6)

    np.random.seed(999)
    y = exog.dot(beta) + np.random.randn(exog.shape[0])

    return y, exog


class TestSparseOLS(object):

    @classmethod
    def setup_class(cls):

        y, exog = generate_sample1()
        cls.mod_sparse = SparseOLS(y, exog)
        cls.mod_ols = OLS(y, exog.toarray())


    def check_basic(self, res1, res2):
        exog = res1.model.exog
        assert_allclose(res1.params, res2.params, rtol=1e-13)
        assert_allclose(res1.bse, res2.bse, rtol=1e-13)
        assert_allclose(res1.pvalues, res2.pvalues, rtol=1e-13)
        assert_allclose(res1.conf_int(), res2.conf_int(), rtol=1e-13)
        assert_allclose(res1.predict(exog[1:-4]),
                        res1.predict(exog[1:-4].toarray()), rtol=1e-13)
        assert_allclose(res1.predict(exog[1:-4]),
                        res2.predict(exog[1:-4].toarray()), rtol=1e-13)
        assert_allclose(res1.fittedvalues, res2.fittedvalues, rtol=1e-13)

        if DEBUG:
            # keep this temporarily for checking results
            res_sparse, res_ols = res1, res2
            print(res_sparse.params)
            print(res_sparse.bse)
            print(res_sparse.t_test(np.eye(5)))
            print()
            print('checking predict')
            print(res_ols.predict(exog[:4].toarray()))
            print(res_sparse.model.predict(res_sparse.params, exog[:4].toarray()))
            print(res_sparse.predict(exog[:4]))
            print(res_sparse.fittedvalues[:4])


    def test_HC0(self):

        res_sparse = self.mod_sparse.fit(cov_type='HC0')
        res_ols = self.mod_ols.fit(cov_type='HC0')
        self.check_basic(res_sparse, res_ols)

    def test_nonrobust(self):

        res_sparse = self.mod_sparse.fit()
        res_ols = self.mod_ols.fit()
        self.check_basic(res_sparse, res_ols)
