"""
Tests for statsmodels.stats.multicollinearity

Created on Thu Apr 30 06:46:13 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_less, assert_equal
import pandas as pd
import pytest

from statsmodels.datasets import longley
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.api import collinear_index, vif, vif_ridge
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from .results import results_multicollinearity as res_r


def get_data(nobs=100, k_vars=4, seed=987536):
    rng = np.random.default_rng(seed)
    rho = np.linspace(0.5, 0.9, k_vars)
    x = rng.standard_normal((nobs, k_vars)) * (1 - rho)
    return x + rho * rng.standard_normal((nobs, 1))


def vif_by_ols(x, constant=True):
    """VIF from the sum of squares of the auxiliary regressions

    The VIF is the ratio of the total sum of squares to the residual sum of
    squares, which is accurate also for large VIF. The total sum of squares
    is centered if there is a constant in the regression.
    """
    result = []
    for k in range(x.shape[1]):
        y = x[:, k]
        exog = np.delete(x, k, axis=1)
        if constant:
            exog = add_constant(exog)
            tss = ((y - y.mean()) ** 2).sum()
        else:
            tss = (y**2).sum()
        result.append(tss / OLS(y, exog).fit().ssr)
    return np.array(result)


@pytest.fixture(scope="module")
def longley_exog():
    return longley.load_pandas().exog


@pytest.fixture(scope="module")
def mtcars_exog():
    return pd.DataFrame(res_r.mtcars)


class TestVifReference:
    def test_longley(self, longley_exog):
        res = vif(longley_exog)
        assert isinstance(res, pd.Series)
        assert_equal(res.index.to_numpy(), longley_exog.columns.to_numpy())
        assert_allclose(res.to_numpy(), res_r.longley_vif, rtol=1e-9)
        assert_allclose(vif(longley_exog.to_numpy()), res_r.longley_vif, rtol=1e-9)

    def test_longley_subset(self, longley_exog):
        res = vif(longley_exog[["GNP", "UNEMP", "ARMED", "POP"]])
        assert_allclose(res.to_numpy(), res_r.longley4_vif, rtol=1e-9)

    def test_mtcars(self, mtcars_exog):
        res = vif(mtcars_exog)
        assert_equal(res.index.to_numpy(), ["disp", "hp", "wt", "drat", "qsec"])
        assert_allclose(res.to_numpy(), res_r.mtcars_vif, rtol=1e-9)

    def test_variance_inflation_factor(self, mtcars_exog):
        # The existing function for one variable agrees with R, with and
        # without a constant in the design matrix.
        x = mtcars_exog.to_numpy()
        for exog, first in [(x, 0), (add_constant(x), 1)]:
            res = [variance_inflation_factor(exog, k + first) for k in range(5)]
            assert_allclose(res, res_r.mtcars_vif, rtol=1e-9)

    def test_formula_design(self, longley_exog):
        # The design matrix of a formula model has a constant, which is
        # skipped. The result is the same for both formula engines.
        import statsmodels.formula.api as smf

        data = longley.load_pandas().data
        mod = smf.ols("TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR", data)
        res = vif(mod.data.orig_exog, has_constant="skip")
        assert np.isnan(res.iloc[0])
        assert_equal(res.index.to_numpy()[1:], longley_exog.columns.to_numpy())
        assert_allclose(res.to_numpy()[1:], res_r.longley_vif, rtol=1e-9)


class TestVifOLS:
    @classmethod
    def setup_class(cls):
        cls.x = get_data(nobs=100, k_vars=5)

    def test_ols(self):
        res = vif(self.x)
        assert res.shape == (5,)
        assert_array_less(1, res)
        assert_allclose(res, vif_by_ols(self.x), rtol=1e-10)

    def test_outliers_influence(self):
        # The existing function needs the constant for the uncentered case
        res = vif(self.x)
        xc = add_constant(self.x)
        res_oi = [variance_inflation_factor(xc, k) for k in range(1, 6)]
        assert_allclose(res, res_oi, rtol=1e-10)
        res_oi = [variance_inflation_factor(self.x, k) for k in range(5)]
        assert_allclose(res, res_oi, rtol=1e-10)

    def test_constant_column(self):
        xc = add_constant(self.x)
        with pytest.raises(ValueError, match="constant columns at positions"):
            vif(xc)
        res = vif(xc, has_constant="skip")
        assert np.isnan(res[0])
        assert_allclose(res[1:], vif(self.x), rtol=1e-13)

    def test_constant_pandas(self):
        df = pd.DataFrame(self.x, columns=list("abcde"))
        df.insert(2, "one", 1.0)
        df["two"] = 2.0
        with pytest.raises(ValueError, match="constant columns"):
            vif(df)
        res = vif(df, has_constant="skip")
        assert_equal(res.index.to_numpy(), list(df.columns))
        assert_equal(np.isnan(res.to_numpy()), df.columns.isin(["one", "two"]))
        assert_allclose(res.dropna().to_numpy(), vif(self.x), rtol=1e-13)

    def test_all_constant(self):
        res = vif(np.ones((10, 3)), has_constant="skip")
        assert np.isnan(res).all()
        assert vif(np.column_stack((np.ones(10), self.x[:10, 0])), has_constant="skip")[
            1
        ] == pytest.approx(1.0)

    def test_invalid_has_constant(self):
        with pytest.raises(ValueError, match="has_constant"):
            vif(self.x, has_constant="drop")

    def test_single_variable(self):
        assert_allclose(vif(self.x[:, 0]), [1.0])
        assert_allclose(vif(self.x[:, :1]), [1.0])

    def test_orthogonal(self):
        x = np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        assert_allclose(vif(x), np.ones(3), rtol=1e-14)

    def test_integer_and_list(self):
        x = np.array([[1, 2, 0], [2, 1, 1], [3, 5, 0], [4, 3, 1], [5, 9, 1]])
        res = vif(x)
        assert_allclose(res, vif(x.astype(float)), rtol=1e-14)
        assert_allclose(res, vif(x.tolist()), rtol=1e-14)
        assert_allclose(res, vif(pd.DataFrame(x).astype(float)), rtol=1e-14)

    def test_bool_dummies(self):
        dummies = pd.DataFrame(
            np.arange(100)[:, None] % 4 == np.arange(1, 4), columns=["d1", "d2", "d3"]
        )
        df = pd.concat(
            [pd.DataFrame(self.x[:, :2], columns=["a", "b"]), dummies], axis=1
        )
        res = vif(df)
        assert_allclose(
            res.to_numpy(), vif_by_ols(df.to_numpy(dtype=float)), rtol=1e-10
        )

    def test_float32(self):
        x32 = self.x.astype(np.float32)
        res = vif(x32)
        assert res.dtype == np.float64
        # data are converted to float64
        assert_allclose(res, vif(x32.astype(np.float64)), rtol=1e-14)
        # the precision of the data is limited
        assert_allclose(res, vif(self.x), rtol=1e-5)
        res_pd = vif(pd.DataFrame(x32))
        assert_allclose(res_pd.to_numpy(), res, rtol=1e-14)

    def test_float32_collinear(self):
        x = self.x.astype(np.float32)
        x = np.column_stack((x, x[:, 0] + x[:, 1]))
        assert x.dtype == np.float32
        # float32 rounding error in the sum, the columns are not exactly collinear
        assert_array_less(1e10, vif(x)[[0, 1, 5]])
        assert_array_less(vif(x)[2:5], 1e3)

    def test_invalid_data(self):
        x = self.x.copy()
        x[3, 1] = np.nan
        with pytest.raises(ValueError, match="missing or infinite"):
            vif(x)
        x[3, 1] = np.inf
        with pytest.raises(ValueError, match="missing or infinite"):
            vif(x)
        with pytest.raises(ValueError, match="ndim"):
            vif(np.ones((3, 3, 3)))


class TestVifSingular:
    @classmethod
    def setup_class(cls):
        cls.x = get_data(nobs=100, k_vars=4)

    def test_exact_collinear(self):
        x = self.x
        xs = np.column_stack((x, x[:, 1] + 2 * x[:, 3]))
        res = vif(xs)
        assert_equal(np.isinf(res), [False, True, False, True, True])
        # the other variables are not affected by the additional column
        assert_allclose(res[[0, 2]], vif_by_ols(x)[[0, 2]], rtol=1e-10)
        idx_collinear, idx_keep = collinear_index(xs)
        assert_equal(idx_collinear, [4])
        assert_equal(idx_keep, [0, 1, 2, 3])
        assert_array_less(vif(xs[:, idx_keep]), 1e3)

    def test_duplicate_column(self):
        res = vif(np.column_stack((self.x, self.x[:, 0])))
        assert_equal(np.isinf(res), [True, False, False, False, True])

    def test_multiple_relations(self):
        x = self.x
        xs = np.column_stack((x, x[:, 0] - x[:, 1], x[:, 2] + x[:, 3]))
        res = vif(xs)
        assert np.isinf(res).all()

    def test_small_weight(self):
        # The tiny component of a total is still determined by the total
        # and the large component.
        rng = np.random.default_rng(87245)
        big = 1e6 * rng.standard_normal(200)
        small = rng.standard_normal(200)
        other = rng.standard_normal(200)
        res = vif(np.column_stack((big, small, big + small, other)))
        assert_equal(np.isinf(res), [True, True, True, False])
        assert_allclose(res[3], 1.0, rtol=0.1)

    def test_nearly_collinear(self):
        rng = np.random.default_rng(87245)
        x = self.x
        noise = 1e-4 * rng.standard_normal(x.shape[0])
        xs = np.column_stack((x, x[:, 1] + 2 * x[:, 3] + noise))
        res = vif(xs)
        assert_array_less(1e6, res[[1, 3, 4]])
        assert np.isfinite(res).all()
        assert_allclose(res, vif_by_ols(xs), rtol=1e-6)
        # not detected as exactly collinear
        assert_equal(collinear_index(xs)[0], [])

    @pytest.mark.parametrize("scale", [1e-3, 1e-5])
    def test_nearly_collinear_separate(self, scale):
        # An almost collinear pair does not affect the other variables if
        # there is also an exact relation
        rng = np.random.default_rng(87245)
        x = rng.standard_normal((200, 8))
        x = np.column_stack(
            (x, x[:, 0] + x[:, 1], x[:, 4] + scale * rng.standard_normal(200))
        )
        res = vif(x)
        assert_array_less(res[[2, 3, 5, 6, 7]], 2)
        assert np.isinf(res[[0, 1, 8]]).all()
        assert_array_less(0.1 / scale**2, res[[4, 9]])
        assert np.isfinite(res[[4, 9]]).all()

    def test_dummy_trap(self, longley_exog):
        # A full set of dummy variables is an implicit constant. It is
        # collinear with the constant that is added by the VIF regressions.
        group = longley_exog["YEAR"].to_numpy().astype(int) % 4
        dummies = pd.get_dummies(group, prefix="d", dtype=float)
        df = pd.concat([longley_exog[["GNP", "ARMED"]], dummies], axis=1)
        res = vif(df)
        assert_equal(res.index.to_numpy(), ["GNP", "ARMED", "d_0", "d_1", "d_2", "d_3"])
        assert_equal(np.isinf(res.to_numpy()), [False, False, True, True, True, True])
        # dropping one dummy variable removes the collinearity
        res2 = vif(df.drop(columns="d_3"))
        assert np.isfinite(res2).all()
        assert_allclose(
            res2.to_numpy(), vif_by_ols(df.drop(columns="d_3").to_numpy()), rtol=1e-10
        )

    def test_moment_matrix_singular(self):
        xs = np.column_stack((self.x, self.x[:, 1] + self.x[:, 2]))
        res = vif(np.cov(xs, rowvar=False), is_moment=True)
        assert_allclose(res, vif(xs), rtol=1e-12)
        assert_equal(np.isinf(res), [False, True, True, False, True])


class TestVifMoment:
    @classmethod
    def setup_class(cls):
        cls.x = get_data(nobs=100, k_vars=5)
        cls.expected = vif(cls.x)

    def test_cov_corr(self):
        x = self.x
        assert_allclose(vif(np.cov(x, rowvar=False), is_moment=True), self.expected)
        corr = np.corrcoef(x, rowvar=False)
        assert_allclose(vif(corr, is_moment=True), self.expected, rtol=1e-12)
        # covariance of rescaled data
        cov = np.cov(x * [1, 10, 1e3, 1e-3, 5], rowvar=False)
        assert_allclose(vif(cov, is_moment=True), self.expected, rtol=1e-12)
        # centered cross-products
        xd = x - x.mean(0)
        assert_allclose(vif(xd.T @ xd, is_moment=True), self.expected, rtol=1e-12)

    def test_pandas(self):
        df = pd.DataFrame(self.x, columns=list("abcde"))
        res = vif(df.cov(), is_moment=True)
        assert isinstance(res, pd.Series)
        assert_equal(res.index.to_numpy(), list("abcde"))
        assert_allclose(res.to_numpy(), self.expected, rtol=1e-12)
        res = vif(df.corr().to_numpy(), is_moment=True)
        assert isinstance(res, np.ndarray)

    def test_uncentered(self):
        # The cross-product matrix without centering corresponds to the
        # regressions without a constant
        x = self.x + 1
        res = vif(x.T @ x, is_moment=True)
        assert_allclose(res, vif_by_ols(x, constant=False), rtol=1e-10)
        assert_array_less(vif(x), res)

    def test_constant(self):
        cov = np.cov(np.column_stack((self.x, np.ones(100))), rowvar=False)
        assert_equal(cov[-1], np.zeros(6))
        with pytest.raises(ValueError, match="constant columns"):
            vif(cov, is_moment=True)
        res = vif(cov, is_moment=True, has_constant="skip")
        assert np.isnan(res[-1])
        assert_allclose(res[:-1], self.expected, rtol=1e-12)

    def test_invalid(self):
        cov = np.cov(self.x, rowvar=False)
        with pytest.raises(ValueError, match="square"):
            vif(self.x, is_moment=True)
        asym = cov.copy()
        asym[0, 1] += 0.1
        with pytest.raises(ValueError, match="symmetric"):
            vif(asym, is_moment=True)
        indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(ValueError, match="positive semi-definite"):
            vif(indefinite, is_moment=True)
        negative = np.diag([1.0, -1.0, 2.0])
        with pytest.raises(ValueError, match="negative diagonal"):
            vif(negative, is_moment=True)
        with pytest.raises(ValueError, match="missing or infinite"):
            vif(np.full((2, 2), np.nan), is_moment=True)


# fmt: off
# Output of the ridge regression in SAS PROC REG with the OUTVIF option, as
# included in the original pull request. The data are the French import
# data, YEAR IMPORT DOPROD STOCK CONSUM.
RIDGE_DATA = np.array(
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
RIDGE_VIF = np.array(
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
).reshape(-1, 5)
# fmt: on


class TestVifRidge:
    @classmethod
    def setup_class(cls):
        # the first 11 observations and the variables DOPROD STOCK CONSUM
        cls.x = RIDGE_DATA[:11, 2:]
        cls.pen_factors = RIDGE_VIF[:, 1]

    @pytest.mark.parametrize("is_moment", [True, False])
    def test_sas(self, is_moment):
        data = np.corrcoef(self.x, rowvar=False) if is_moment else self.x
        res = vif_ridge(data, self.pen_factors, is_moment=is_moment)
        assert res.shape == (RIDGE_VIF.shape[0], 3)
        assert_allclose(res, RIDGE_VIF[:, 2:], atol=1.01e-3)

    def test_unpenalized(self):
        res = vif_ridge(self.x, self.pen_factors)
        assert_allclose(res[0], vif(self.x), rtol=1e-10)
        # compare with the OLS estimate of the regression with a constant
        exog = add_constant(self.x)
        xxi = np.linalg.inv(exog.T @ exog)
        vif_ols = (np.diag(xxi) * exog.var(0) * len(self.x))[1:]
        assert_allclose(res[0], vif_ols, rtol=2e-12)

    def test_formula(self):
        corr = np.corrcoef(self.x, rowvar=False)
        pen_factors = [0.01, 0.5, 2.0]
        expected = []
        for k in pen_factors:
            inv = np.linalg.inv(corr + k * np.eye(3))
            expected.append(np.diag(inv @ corr @ inv))
        res = vif_ridge(self.x, pen_factors)
        assert_allclose(res, expected, rtol=1e-12)
        # a scalar is a single penalization factor
        assert_allclose(vif_ridge(self.x, 0.5), np.asarray(expected)[[1]], rtol=1e-12)

    def test_pandas(self):
        df = pd.DataFrame(self.x, columns=["doprod", "stock", "consum"])
        res = vif_ridge(df, self.pen_factors)
        assert isinstance(res, pd.DataFrame)
        assert_equal(res.columns.to_numpy(), ["doprod", "stock", "consum"])
        assert_allclose(res.index.to_numpy(), self.pen_factors)
        assert_allclose(res.to_numpy(), vif_ridge(self.x, self.pen_factors))

    def test_constant(self):
        xc = add_constant(self.x)
        with pytest.raises(ValueError, match="constant columns"):
            vif_ridge(xc, self.pen_factors)
        res = vif_ridge(xc, self.pen_factors, has_constant="skip")
        assert np.isnan(res[:, 0]).all()
        assert_allclose(res[:, 1:], vif_ridge(self.x, self.pen_factors))

    def test_singular(self):
        x = np.column_stack((self.x, self.x[:, 0] + self.x[:, 1]))
        pen_factors = [0, 1e-3, 0.1]
        res = vif_ridge(x, pen_factors)
        assert_equal(np.isinf(res[0]), [True, True, False, True])
        assert np.isfinite(res[1:]).all()
        # the penalization makes the variance finite and smaller
        assert_array_less(res[2], res[1])

    def test_invalid(self):
        with pytest.raises(ValueError, match="non-negative"):
            vif_ridge(self.x, [0.1, -0.1])
        with pytest.raises(ValueError, match="non-negative"):
            vif_ridge(self.x, [0.1, np.nan])
        indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(ValueError, match="positive semi-definite"):
            vif_ridge(indefinite, [0.1], is_moment=True)

    def test_all_constant(self):
        res = vif_ridge(np.ones((5, 2)), [0, 0.1], has_constant="skip")
        assert res.shape == (2, 2)
        assert np.isnan(res).all()


class TestCollinearIndex:
    @classmethod
    def setup_class(cls):
        cls.x = get_data(nobs=100, k_vars=4)

    def test_not_collinear(self):
        idx_collinear, idx_keep = collinear_index(self.x)
        assert idx_collinear.shape == (0,)
        assert_equal(idx_keep, np.arange(4))
        assert idx_keep.dtype.kind == "i"

    def test_sequential(self):
        x = self.x
        # x2 depends on x0 and x1, x3 depends on x0, x1 and x2.
        xs = np.column_stack(
            (x[:, 0], x[:, 1], x[:, 0] - x[:, 1], x[:, 2], x[:, 0] + x[:, 3])
        )
        idx_collinear, idx_keep = collinear_index(xs)
        assert_equal(idx_collinear, [2])
        assert_equal(idx_keep, [0, 1, 3, 4])
        # A collinear column does not affect the columns after it
        xs = np.column_stack((x, x[:, 2] + x[:, 3], x[:, 1] - x[:, 2] + 3 * x[:, 3]))
        idx_collinear, idx_keep = collinear_index(xs)
        assert_equal(idx_collinear, [4, 5])
        # reversed order: x2 = (x2 + x3) - x3 and x1 are in the span of
        # the preceding columns
        idx_collinear, idx_keep = collinear_index(xs[:, ::-1])
        assert_equal(idx_collinear, [3, 4])
        assert_equal(idx_keep, [0, 1, 2, 5])

    def test_constant_first(self):
        xc = add_constant(self.x)
        assert_equal(collinear_index(xc)[0], [])
        # a full set of dummy variables is a constant
        dummies = np.tile(np.eye(3), (34, 1))[:100]
        xd = np.column_stack((np.ones(100), dummies, self.x))
        idx_collinear, idx_keep = collinear_index(xd)
        assert_equal(idx_collinear, [3])
        assert_equal(idx_keep, [0, 1, 2, 4, 5, 6, 7])
        # the columns are not demeaned, so there is no collinearity without
        # the constant column
        assert_equal(collinear_index(xd[:, 1:])[0], [])

    def test_zero_column(self):
        xs = np.column_stack((self.x[:, 0], np.zeros(100), self.x[:, 1]))
        assert_equal(collinear_index(xs)[0], [1])
        xs = np.column_stack((np.zeros(100), self.x[:, :2]))
        assert_equal(collinear_index(xs)[0], [0])

    def test_pandas(self):
        df = pd.DataFrame(self.x, columns=list("abcd"))
        df["e"] = df["a"] + df["b"]
        idx_collinear, idx_keep = collinear_index(df)
        assert isinstance(idx_collinear, pd.Index)
        assert_equal(idx_collinear.to_numpy(), ["e"])
        assert_equal(idx_keep.to_numpy(), list("abcd"))
        # in a different order
        df = df[["e", "a", "b", "c", "d"]]
        idx_collinear, idx_keep = collinear_index(df)
        assert_equal(idx_collinear.to_numpy(), ["b"])
        assert_equal(idx_keep.to_numpy(), ["e", "a", "c", "d"])

    def test_tol(self):
        rng = np.random.default_rng(87245)
        x = self.x
        xs = np.column_stack((x, x[:, 1] + x[:, 2] + 1e-6 * rng.standard_normal(100)))
        assert_equal(collinear_index(xs)[0], [])
        assert_equal(collinear_index(xs, tol=1e-4)[0], [4])
        assert_equal(collinear_index(xs, tol=1e-4)[1], [0, 1, 2, 3])

    def test_scale(self):
        # the relative tolerance does not depend on the scale of the columns
        xs = np.column_stack((self.x[:, 0], 1e8 * self.x[:, 1], 1e-8 * self.x[:, 2]))
        assert_equal(collinear_index(xs)[0], [])
        xs = np.column_stack((xs, xs[:, 0] + 1e-8 * xs[:, 1]))
        assert_equal(collinear_index(xs)[0], [3])

    def test_more_columns_than_observations(self):
        rng = np.random.default_rng(87245)
        x = rng.standard_normal((5, 8))
        idx_collinear, idx_keep = collinear_index(x)
        assert_equal(idx_collinear, [5, 6, 7])
        assert_equal(idx_keep, np.arange(5))

    def test_float32_and_int(self):
        xs = np.column_stack((self.x, self.x[:, 0] + self.x[:, 1]))
        # integer valued data are exact in float32
        xf = np.round(10 * xs[:, :2]).astype(np.float32)
        xf = np.column_stack((xf, xf[:, 0] - 2 * xf[:, 1]))
        assert_equal(collinear_index(xf)[0], [2])
        # the precision of float32 limits the collinearity of other data
        assert_equal(collinear_index(xs.astype(np.float32))[0], [])
        xi = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 2], [2, 1, 3]])
        assert_equal(collinear_index(xi)[0], [2])

    def test_invalid(self):
        x = self.x.copy()
        x[0, 0] = np.nan
        with pytest.raises(ValueError, match="missing or infinite"):
            collinear_index(x)

    @pytest.mark.parametrize(
        "design, expected",
        [("alias_sum", res_r.alias_sum), ("alias_dummies", res_r.alias_dummies)],
    )
    def test_r_alias(self, longley_exog, design, expected):
        # The aliased columns of the lm in R
        const = np.ones(len(longley_exog))
        gnp = longley_exog["GNP"].to_numpy()
        unemp = longley_exog["UNEMP"].to_numpy()
        armed = longley_exog["ARMED"].to_numpy()
        if design == "alias_sum":
            pop = longley_exog["POP"].to_numpy()
            x = np.column_stack((const, gnp, unemp, gnp + unemp, armed, pop))
        else:
            group = longley_exog["YEAR"].to_numpy() % 4
            dummies = group[:, None] == np.arange(4)
            x = np.column_stack((const, gnp, dummies, armed))
        idx_collinear, _ = collinear_index(x)
        assert_equal(idx_collinear, expected)
