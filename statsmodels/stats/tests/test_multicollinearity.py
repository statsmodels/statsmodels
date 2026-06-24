# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 06:46:13 2015

Author: Josef Perktold
License: BSD-3
"""

import warnings

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_less,
    assert_equal,
)
import pandas as pd
import pytest

from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multicollinearity import (
    MultiCollinearity,
    MultiCollinearitySequential,
    collinear_index,
    vif,
    vif_ridge,
    vif_selection,
)
import statsmodels.stats.outliers_influence as smoi


def assert_allclose_large(x, y, rtol=1e-6, atol=0, ltol=1e12):
    """
    assert x and y are allclose or x and y are larger than ltol
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    mask = (y > ltol) & (x > ltol)
    assert_allclose(x[~mask], y[~mask], rtol=rtol, atol=atol)


def _get_data(nobs=100, k_vars=4):
    np.random.seed(987536)
    rho_coeff = np.linspace(0.5, 0.9, k_vars - 1)
    x = np.random.randn(nobs, k_vars - 1) * (1 - rho_coeff)
    x += rho_coeff * np.random.randn(nobs, 1)
    return x


class CheckMuLtiCollinear(object):
    @classmethod
    def get_data(cls):
        nobs, k_vars = 100, 4
        cls.x = x = _get_data(nobs=nobs, k_vars=k_vars)
        cls.xs = (x - x.mean(0)) / x.std(0)
        cls.xf = np.column_stack((np.ones(nobs), x))
        cls.check_pandas = False

    def test_sequential(self):
        xf = self.xf
        x = self.x
        ols_results = [
            OLS(xf[:, k], xf[:, :k]).fit() for k in range(1, xf.shape[1])
        ]
        rsquared0 = np.array([res.rsquared for res in ols_results])
        vif0 = 1.0 / np.maximum((1.0 - rsquared0), 1e-20)

        mcoll = MultiCollinearitySequential(self.x)

        assert_allclose(
            mcoll.rsquared_partial, rsquared0, rtol=1e-12, atol=1e-15
        )
        # infs could be just large values because of floating point imprecision
        # assert_allclose(mcoll.vif, vif0, rtol=1e-13)
        # mask_inf = np.isinf(vif0) & ~np.isinf(mcoll.vif)
        # assert_allclose(mcoll.vif[~mask_inf], vif0[~mask_inf], rtol=1e-13)
        # assert_array_less(1e30, mcoll.vif[mask_inf])
        assert_allclose_large(mcoll.vif, vif0, rtol=1e-13, ltol=1e-14)

        if not (1.0 - rsquared0 == 0).any():
            # The following requires nonsingular matrix because of Cholesky
            # check moment matrix as input
            x_dm = x - x.mean(0)  # standardize doesn't demean
            mcoll2 = MultiCollinearitySequential(
                None, moment_matrix=x_dm.T.dot(x_dm)
            )
            assert_allclose(
                mcoll2.rsquared_partial, mcoll.rsquared_partial, rtol=1e-13
            )
            assert_allclose(mcoll2.vif, mcoll.vif, rtol=1e-13)

            # check correlation matrix as input
            mcoll2 = MultiCollinearitySequential(
                None, np.corrcoef(x, rowvar=False), standardize=False
            )
            assert_allclose(
                mcoll2.rsquared_partial, mcoll.rsquared_partial, rtol=1e-13
            )
            assert_allclose(mcoll2.vif, mcoll.vif, rtol=1e-13)

        # Note we need a constant since x is not demeaned
        collinear_columns, keep_columns = collinear_index(xf)
        not_coll = [
            i for i in range(xf.shape[1]) if i not in collinear_columns
        ]
        assert_equal(not_coll, keep_columns)
        # I haven't checked what the equvalent threshold is exactly
        # subtracting 1 from index to ignore constant column
        assert_equal(collinear_columns - 1, np.nonzero(mcoll.vif > 1e14)[0])

        if self.check_pandas:
            collinear_columns, keep_columns = collinear_index(self.xf_pd)
            # keep = ['const', 'var0', 'var1', 'var2']
            assert list(self.xf_pd.columns[not_coll]) == keep_columns

    def test_multicoll(self):
        xf = np.asarray(self.xf)  # convert from pandas DataFrame, for OLS only
        k_vars = self.xf.shape[1]
        ols_results = []

        for k in range(1, k_vars):
            idx_k = list(range(k_vars))
            del idx_k[k]
            ols_results.append(OLS(xf[:, k], xf[:, idx_k]).fit())

        rsquared0 = np.array([res.rsquared for res in ols_results])
        vif0 = 1.0 / np.maximum((1.0 - rsquared0), 1e-20)
        # for checking singular cases with pdb in pytest
        # if ((1. - rsquared0)<1e-14).any():
        #     raise

        mcoll = MultiCollinearity(self.x)

        assert_allclose(mcoll.rsquared_partial, rsquared0, rtol=1e-13)
        assert_allclose_large(mcoll.vif, vif0, rtol=1e-12, ltol=1e12)
        assert_allclose(mcoll.tss, np.ones(self.x.shape[1]))
        rss = np.array([1 - res.rsquared for res in ols_results])
        assert_allclose(mcoll.rss, rss, atol=1e-10)
        alt_vif = mcoll.get_vif(ridge_factor=0)
        if mcoll.vif.max() < 1e10:
            # Only test nonsingular
            assert_allclose(mcoll.vif, alt_vif)

        if mcoll.eigenvalues.min() > 1e-14:
            # TODO: check what to do in singular case
            xf0 = self.xf[:, :-2]
            xf1 = self.xf[:, -2:]
            # TODO: use projection helper function
            resid = xf1 - xf0.dot(np.linalg.pinv(xf0).dot(xf1))
            corrp1 = np.corrcoef(resid, rowvar=0)[0, 1]
            assert_allclose(mcoll.corr_partial[-1, -2], corrp1, rtol=1e-13)

        vif1_ = vif(self.x)
        vif1 = np.asarray(vif1_)  # check values if pandas.Series
        # TODO: why does mcoll.vif have infs but vif1 doesn't?
        assert_allclose_large(vif1, mcoll.vif, rtol=1e-12, ltol=1e12)
        assert_allclose_large(vif1, vif0, rtol=1e-12, ltol=1e12)

        if self.check_pandas:
            assert_equal(vif1_.index.values, self.names)

        # check moment matrix as input
        x_dm = self.x - self.x.mean(0)  # standardize doesn't demean
        mcoll2 = MultiCollinearity(None, moment_matrix=x_dm.T.dot(x_dm))
        assert_allclose(
            mcoll2.rsquared_partial, mcoll.rsquared_partial, rtol=1e-13
        )
        # the following has floating point noise, mcoll.vif has inf
        assert_allclose_large(mcoll2.vif, mcoll.vif, rtol=1e-13, ltol=1e12)

        # compare with outlier influence function
        x_dm_arr = np.asarray(x_dm)  # smoi vif does not work with pandas
        vif_oi = np.array(
            [
                smoi.variance_inflation_factor(x_dm_arr, ii)
                for ii in range(x_dm_arr.shape[1])
            ]
        )
        assert_allclose_large(mcoll2.vif, vif_oi, rtol=1e-12, ltol=1e12)

        # check correlation matrix as input
        corr = np.corrcoef(self.x, rowvar=False)
        mcoll2 = MultiCollinearity(None, corr, standardize=False)
        assert_allclose(
            mcoll2.rsquared_partial, mcoll.rsquared_partial, rtol=1e-13
        )
        assert_allclose_large(mcoll2.vif, mcoll.vif, rtol=1e-13, ltol=1e12)

        corr = np.corrcoef(self.x, rowvar=False)
        vif1 = vif(None, moment_matrix=corr)
        assert_allclose_large(vif1, mcoll.vif, rtol=1e-12, ltol=1e12)

        evals = np.linalg.svd(corr, compute_uv=False)
        assert_allclose(mcoll2.eigenvalues, evals, rtol=1e-13, atol=1e-14)
        # we shouldn't have tiny negative eigenvalues, those are set to zero
        assert_equal(mcoll2.eigenvalues[mcoll2.eigenvalues < 0], np.array([]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            condn = mcoll2.condition_number
        assert_allclose_large(
            evals[0] / evals[-1], condn ** 2, rtol=1e-12, ltol=1e12
        )

        # test if standardize is false, equivalence with constant column
        mcoll_ns = MultiCollinearity(xf, standardize=False)
        vif_nc = mcoll_ns.vif[1:] * xf.var(0)[1:]
        # inf in vif_nc but not in mcoll_ns.vif
        assert_allclose_large(mcoll.vif, vif_nc, rtol=1e-12, ltol=1e12)
        vif_nc2 = vif(xf, standardize=False)[1:] * xf.var(0)[1:]
        assert_allclose_large(mcoll.vif, vif_nc2, rtol=1e-12, ltol=1e12)


class TestMultiCollinearNonSingular(CheckMuLtiCollinear):
    # Example: with non-singular continuous data
    @classmethod
    def setup_class(cls):
        cls.get_data()


class TestMultiCollinearSingular1(CheckMuLtiCollinear):
    # Example: with singular continuous data
    @classmethod
    def setup_class(cls):
        cls.get_data()
        x = np.column_stack((cls.x[:, -2:].sum(1), cls.x))

        cls.x = x
        cls.xs = (x - x.mean(0)) / x.std(0)
        cls.xf = np.column_stack((np.ones(x.shape[0]), x))


class TestMultiCollinearSingular2(CheckMuLtiCollinear):
    # Example: with singular dummy variables
    @classmethod
    def setup_class(cls):
        cls.get_data()
        nobs = cls.x.shape[0]
        xd = np.tile(np.arange(5), nobs // 5)[:, None] == np.arange(5)
        x = np.column_stack((cls.x, xd))

        cls.x = x
        cls.xs = (x - x.mean(0)) / x.std(0)
        cls.xf = np.column_stack((np.ones(x.shape[0]), x))


class TestMultiCollinearPandas(CheckMuLtiCollinear):
    # Example: with singular continuous data
    @classmethod
    def setup_class(cls):
        cls.get_data()
        import pandas

        cls.names = ["var%d" % i for i in range(cls.x.shape[1])]
        cls.x = pandas.DataFrame(cls.x, columns=cls.names)
        cls.xf_pd = pandas.DataFrame(cls.xf, columns=["const"] + cls.names)
        cls.check_pandas = True


@pytest.mark.parametrize("use_pandas", [False, True])
def test_vif_selection(use_pandas):
    x = _get_data(nobs=100, k_vars=15)
    if use_pandas:
        x = pd.DataFrame(x)
    idx, _ = vif_selection(x)
    assert_equal(idx, np.arange(10, dtype=int))
    x_sel = x.iloc[:, idx] if use_pandas else x[:, idx]
    assert_array_less(vif(x_sel), 10)

    x_rev = x[[c for c in x.columns[::-1]]] if use_pandas else x[:, ::-1]
    idx, _ = vif_selection(x_rev)
    expected = np.arange(4, 14, dtype=int)
    if use_pandas:
        expected = list(x_rev.columns[expected])
    assert_equal(idx, expected)
    idx = x.shape[1] - np.array(idx) - 1
    x_sel = x_rev.iloc[:, idx] if use_pandas else x[:, idx]
    assert_array_less(vif(x_sel), 10)

    threshold = 5
    idx, _ = vif_selection(x_rev, threshold=threshold)
    expected = np.arange(7, 14, dtype=int)
    if use_pandas:
        expected = list(x_rev.columns[expected])
    assert_equal(idx, expected)
    idx = x.shape[1] - np.array(idx) - 1
    x_sel = x_rev.iloc[:, idx] if use_pandas else x[:, idx]
    assert_array_less(vif(x_sel), threshold)


def test_vif_selection_equiv():
    x = _get_data(nobs=100, k_vars=15)
    idx, _ = vif_selection(x)
    moments = np.corrcoef(x, rowvar=False)
    idx_moment, _ = vif_selection(None, moment_matrix=moments)
    assert_equal(idx, idx_moment)
    xs = (x - x.mean(0)) / (x.std(0) * np.sqrt(x.shape[0]))
    idx_std, _ = vif_selection(xs, standardize=False)
    assert_equal(idx, idx_std)


@pytest.mark.parametrize("is_corr", [True, False])
def test_vif_ridge(is_corr):

    dta = np.array(
        """
    49 15.9 149.3 4.2 108.1
    50 16.4 161.2 4.1 114.8
    51 19 171.5 3.1 123.2
    52 19.1 175.5 3.1 126.9
    53 18.8 180.8 1.1 132.1
    54 20.4 190.7 2.2 137.7
    55 22.7 202.1 2.1 146
    56 26.5 212.4 5.6 154.1
    57 28.1 226.1 5 162.3
    58 27.6 231.9 5.1 164.3
    59 26.3 239 0.7 167.6
    60 31.1 258 5.6 176.8
    61 33.3 269.8 3.9 186.6
    62 37 288.4 3.1 199.7
    63 43.3 304.5 4.6 213.9
    64 49 323.4 7 223.8
    65 50.3 336.8 1.2 232
    66 56.6 353.9 4.5 242.9""".split(),
        float,
    ).reshape(-1, 5)

    # Obs    _RIDGE_     DOPROD     STOCK      CONSUM

    results_vif = np.array(
        """
      2     0.000     185.997    1.01891    186.110
      5     0.001      98.981    1.00845     99.041
      8     0.003      41.779    0.99890     41.804
     11     0.005      22.988    0.99311     23.001
     14     0.007      14.570    0.98836     14.579
     17     0.009      10.089    0.98401     10.095
     20     0.010       8.599    0.98192      8.604
     23     0.012       6.480    0.97783      6.483
     26     0.014       5.075    0.97384      5.078
     29     0.016       4.097    0.96991      4.099
     32     0.018       3.388    0.96603      3.389
     35     0.020       2.858    0.96219      2.859
     38     0.022       2.452    0.95838      2.452
     41     0.024       2.133    0.95461      2.134
     44     0.026       1.878    0.95086      1.879
     47     0.028       1.672    0.94714      1.672
     50     0.030       1.502    0.94345      1.502
     53     0.030       1.502    0.94345      1.502
     56     0.040       0.979    0.92532      0.979
     59     0.050       0.723    0.90773      0.723
     62     0.060       0.579    0.89065      0.578
     65     0.070       0.489    0.87405      0.488
     68     0.080       0.429    0.85792      0.428
     71     0.090       0.386    0.84222      0.386
     74     0.100       0.355    0.82696      0.355
     77     0.200       0.240    0.69474      0.240
     80     0.300       0.204    0.59187      0.204
     83     0.400       0.182    0.51027      0.182
     86     0.500       0.166    0.44446      0.165
     89     0.600       0.152    0.39061      0.152
     92     0.700       0.140    0.34598      0.140
     95     0.800       0.130    0.30859      0.130
     98     0.900       0.121    0.27695      0.121
    101     1.000       0.113    0.24994      0.112""".split(),
        float,
    ).reshape(
        -1, 5
    )  # noqa

    import pandas as pd

    columns = "YEAR IMPORT DOPROD STOCK CONSUM".lower().split()
    example = pd.DataFrame(dta, columns=columns)

    x = example["doprod stock consum".split()].values
    y = example["import"].values

    x = x[:11]
    y = y[:11]
    if is_corr:
        xxs = np.corrcoef(x, rowvar=0)
    else:
        xxs = x

    pen_factors = results_vif[:, 1]
    res_vif = vif_ridge(xxs, pen_factors, is_corr=is_corr)

    assert_almost_equal(res_vif, results_vif[:, 2:], decimal=3)

    from statsmodels.tools.tools import add_constant

    exog = add_constant(x)

    xxi = np.linalg.inv(exog.T.dot(exog))
    vif_ols = (np.diag(xxi) * exog.var(0) * len(x))[1:]
    assert_allclose(res_vif[0], vif_ols, rtol=2e-12)


def test_constant_exception():
    with pytest.raises(ValueError):
        MultiCollinearity(np.ones((100, 1)))
    with pytest.raises(ValueError):
        vif_selection(np.ones((100, 1)))
