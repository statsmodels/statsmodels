
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy import linalg

from statsmodels import robust
import statsmodels.robust.covariance as robcov
import statsmodels.robust.norms as robnorms
import statsmodels.robust.scale as robscale

from .results import results_cov as res_cov

cur_dir = Path(__file__).parent.resolve()

file_name = "hbk.csv"
file_path = Path(cur_dir).joinpath("results", file_name)

dta_hbk = pd.read_csv(file_path)


def test_mahalanobis():
    rs = np.random.RandomState(987676453)
    x = rs.randn(10, 3)

    d1 = (x**2).sum(1)
    d0 = robcov.mahalanobis(x, np.eye(3))
    assert_allclose(d0, d1, rtol=1e-10)
    d2 = robcov.mahalanobis(x, cov_inv=np.eye(3))
    assert_allclose(d2, d1, rtol=1e-10)

    d3 = robcov.mahalanobis(x, 2 * np.eye(3))
    assert_allclose(d3, 0.5 * d1, rtol=1e-10)
    d4 = robcov.mahalanobis(x, cov_inv=2 * np.eye(3))
    assert_allclose(d4, 2 * d1, rtol=1e-10)


def test_outliers_gy():
    # regression test and basic properties
    # no test for tie warnings
    seed = 567812  # 123
    rs = np.random.RandomState(seed)

    nobs = 1000
    x = rs.randn(nobs)
    d = x**2
    d2 = d.copy()
    n_outl = 10
    d2[:n_outl] += 10
    res = robcov._outlier_gy(d2, distr=None, k_endog=1, trim_prob=0.975)
    # next is regression test
    res1 = [0.017865444296085831, 8.4163674239050081, 17.0, 42.0, 5.0238861873148881]
    assert_allclose(res, res1, rtol=1e-13)
    reject_thr = (d2 > res[1]).sum()
    reject_float = nobs * res[0]
    assert_equal(reject_thr, res[2])
    assert_equal(int(reject_float), res[2])
    # tests for fixed cutoff at 0.975
    assert_equal((d2 > res[4]).sum(), res[3])
    assert_allclose(res[3], nobs * 0.025 + n_outl, rtol=0.5)
    # + n_outl because not under Null

    x3 = x[:-1].reshape(-1, 3)
    # standardize, otherwise the sample wouldn't be close enough to distr
    x3 = (x3 - x3.mean(0)) / x3.std(0)
    d3 = (x3**2).sum(1)
    nobs = len(d3)
    n_outl = 0

    res = robcov._outlier_gy(d3, distr=None, k_endog=3, trim_prob=0.975)
    # next is regression test
    res1 = [0.0085980695527445583, 12.605802816238732, 2.0, 9.0, 9.3484036044961485]
    assert_allclose(res, res1, rtol=1e-13)
    reject_thr = (d3 > res[1]).sum()
    reject_float = nobs * res[0]
    assert_equal(reject_thr, res[2])
    assert_equal(int(reject_float), res[2])
    # tests for fixed cutoff at 0.975
    assert_equal((d3 > res[4]).sum(), res[3])
    assert_allclose(res[3], nobs * 0.025 + n_outl, rtol=0.5)
    # fixed cutoff at 0.975, + n_outl because not under Null


class TestOGKMad:

    @classmethod
    def setup_class(cls):

        cls.res1 = robcov.cov_ogk(dta_hbk, rescale=False, ddof=0, reweight=0.9)
        cls.res2 = res_cov.results_ogk_mad

    def test(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.cov, res2.cov, rtol=1e-10)
        assert_allclose(res1.mean, res2.center, rtol=1e-10)
        assert_allclose(res1.cov_raw, res2.cov_raw, rtol=1e-8)
        assert_allclose(res1.loc_raw, res2.center_raw, rtol=1e-8)


class TestOGKTau(TestOGKMad):

    @classmethod
    def setup_class(cls):

        def sfunc(x):
            return robscale.scale_tau(x, normalize=False, ddof=0)[1]

        cls.res1 = robcov.cov_ogk(
            dta_hbk, scale_func=sfunc, rescale=False, ddof=0, reweight=0.9
        )
        cls.res2 = res_cov.results_ogk_tau

    def test(self):
        # not inherited because of weak agreement with R
        # I did not find options to improve agreement
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.cov, res2.cov, atol=0.05, rtol=1e-10)
        assert_allclose(res1.mean, res2.center, atol=0.03, rtol=1e-10)
        # cov raw differs in scaling, no idea why
        # note rrcov uses C code for this case with hardoced tau scale
        # our results are "better", i.e., correct outliers same as dgp
        # rrcov has one extra outlier
        fact = 1.1356801031633883
        assert_allclose(res1.cov_raw, res2.cov_raw * fact, rtol=1e-8)
        assert_allclose(res1.loc_raw, res2.center_raw, rtol=0.2, atol=0.1)


def test_tyler():

    # > library(ICSNP)
    # > resty = tyler.shape(hbk, location = ccogk$center, eps=1e-13,
    # print.it=TRUE)
    # [1] "convergence was reached after 55 iterations"

    res2 = np.array(
        [
            [
                1.277856643343122,
                0.298374848328023,
                0.732491311584908,
                0.232045093295329,
            ],
            [
                0.298374848328023,
                1.743589223324287,
                1.220675037619406,
                0.212549156887607,
            ],
            [
                0.732491311584907,
                1.220675037619407,
                2.417486791841682,
                0.295767635758891,
            ],
            [
                0.232045093295329,
                0.212549156887607,
                0.295767635758891,
                0.409157014373402,
            ],
        ]
    )

    # center is from an OGK version
    center = np.array(
        [
            1.5583333333333333,
            1.8033333333333335,
            1.6599999999999999,
            -0.0866666666666667,
        ]
    )
    k_vars = len(center)

    res1 = robcov.cov_tyler(dta_hbk.to_numpy() - center, normalize="trace")
    assert_allclose(np.trace(res1.cov), k_vars, rtol=1e-13)
    cov_det = res1.cov / np.linalg.det(res1.cov) ** (1.0 / k_vars)
    assert_allclose(cov_det, res2, rtol=1e-11)
    assert res1.n_iter == 56

    res1 = robcov.cov_tyler(dta_hbk.to_numpy() - center, normalize="det")
    assert_allclose(np.linalg.det(res1.cov), 1, rtol=1e-13)
    assert_allclose(res1.cov, res2, rtol=1e-11)
    assert res1.n_iter == 56

    res1 = robcov.cov_tyler(dta_hbk.to_numpy() - center, normalize="normal")
    cov_det = res1.cov / np.linalg.det(res1.cov) ** (1.0 / k_vars)
    assert_allclose(cov_det, res2, rtol=1e-11)
    assert res1.n_iter == 56

    res1 = robcov.cov_tyler(dta_hbk.to_numpy() - center)
    cov_det = res1.cov / np.linalg.det(res1.cov) ** (1.0 / k_vars)
    assert_allclose(cov_det, res2, rtol=1e-11)
    assert res1.n_iter == 56


def test_cov_ms():
    # use CovM as local CovS

    # > CovSest(x)  using package rrcov
    # same result with > CovSest(x, method="sdet")
    # but scale difers with method="biweight"
    mean_r = np.array([1.53420879676, 1.82865741024, 1.65565146981])
    cov_r = np.array(
        [
            [1.8090846049573, 0.0479283121828, 0.2446369025717],
            [0.0479283121828, 1.8189886310494, 0.2513025527579],
            [0.2446369025717, 0.2513025527579, 1.7287983150484],
        ]
    )

    scale2_r = np.linalg.det(cov_r) ** (1 / 3)
    shape_r = cov_r / scale2_r
    scale_r = np.sqrt(scale2_r)

    exog_df = dta_hbk[["X1", "X2", "X3"]]
    mod = robcov.CovM(exog_df)
    # with default start, default start could get wrong local optimum
    res = mod.fit()
    assert_allclose(res.mean, mean_r, rtol=1e-5)
    assert_allclose(res.shape, shape_r, rtol=1e-5)
    assert_allclose(res.cov, cov_r, rtol=1e-5)
    assert_allclose(res.scale, scale_r, rtol=1e-5)

    # with results start
    res = mod.fit(start_mean=mean_r, start_shape=shape_r, start_scale=scale_r)
    assert_allclose(res.mean, mean_r, rtol=1e-5)
    assert_allclose(res.shape, shape_r, rtol=1e-5)
    assert_allclose(res.cov, cov_r, rtol=1e-5)
    assert_allclose(res.scale, scale_r, rtol=1e-5)

    mod_s = robcov.CovDetS(exog_df)
    res = mod_s.fit()
    assert_allclose(res.mean, mean_r, rtol=1e-5)
    assert_allclose(res.shape, shape_r, rtol=1e-5)
    assert_allclose(res.cov, cov_r, rtol=1e-5)
    assert_allclose(res.scale, scale_r, rtol=1e-5)


def test_covdetmcd():

    # results from rrcov
    # > cdet = CovMcd(x = hbk, raw.only = TRUE, nsamp = "deterministic",
    #                 use.correction=FALSE)
    cov_dmcd_r = np.array(
        """
    2.2059619213639   0.0223939863695   0.7898958050933   0.4060613360808
    0.0223939863695   1.1384166802155   0.4315534571891  -0.2344041030201
    0.7898958050933   0.4315534571891   1.8930117467493  -0.3292893001459
    0.4060613360808  -0.2344041030201  -0.3292893001459   0.6179686100845
    """.split(),
        float,
    ).reshape(4, 4)

    mean_dmcd_r = np.array([1.7725, 2.2050, 1.5375, -0.0575])

    mod = robcov.CovDetMCD(dta_hbk)
    res = mod.fit(40, maxiter_step=100, reweight=False)
    assert_allclose(res.mean, mean_dmcd_r, rtol=1e-5)
    assert_allclose(res.cov, cov_dmcd_r, rtol=1e-5)

    # with reweighting
    # covMcd(x = hbk, nsamp = "deterministic", use.correction = FALSE)
    # iBest: 5; C-step iterations: 7, 7, 7, 4, 6, 6
    # Log(Det.):  -2.42931967153

    mean_dmcdw_r = np.array(
        [1.5338983050847, 1.8322033898305, 1.6745762711864, -0.0728813559322]
    )
    cov_dmcdw_r = np.array(
        """
    1.5677744869295   0.09285770205078   0.252076010128   0.13873444408300
    0.0928577020508   1.56769177397171   0.224929617385  -0.00516128856542
    0.2520760101278   0.22492961738467   1.483829106079  -0.20275013775619
    0.1387344440830  -0.00516128856542  -0.202750137756   0.43326701543885
    """.split(),
        float,
    ).reshape(4, 4)

    mod = robcov.CovDetMCD(dta_hbk)
    res = mod.fit(40, maxiter_step=100)  # default is reweight=True
    assert_allclose(res.mean, mean_dmcdw_r, rtol=1e-5)
    # R uses different trimming correction
    # compare only shape (using trace for simplicity)
    shape = res.cov / np.trace(res.cov)
    shape_r = cov_dmcdw_r / np.trace(cov_dmcdw_r)
    assert_allclose(shape, shape_r, rtol=1e-5)


def test_naive_ledoit_wolf_shrinkage():
    # GH-10368: the shrinkage target mu * I was missing, so that the result
    # was s * S instead of (1 - s) * S + s * mu * I.
    # Reference values from scikit-learn 1.9.1
    # >>> from sklearn.covariance import ledoit_wolf
    # >>> x = np.sin(np.arange(1.0, 25.0)).reshape(6, 4) * [1.0, 2.0, 3.0, 4.0]
    # >>> cov_sk, s = ledoit_wolf(x, assume_centered=True)  # s = 0.1837
    # >>> cov_sk_wide, s = ledoit_wolf(x[:3], assume_centered=True)  # s = 0.3020
    # >>> cov_sk_mean, s = ledoit_wolf(x)  # centered at the mean, s = 0.2150
    x = np.sin(np.arange(1.0, 25.0)).reshape(6, 4) * [1.0, 2.0, 3.0, 4.0]
    cov_sk = np.array(
        """
        1.2191580449982553  0.5277798166688882 -0.6131189647120341 -1.9389452077521723
        0.5277798166688882  2.2240825250612843  0.8390380551559709 -1.7800250513761013
       -0.6131189647120341  0.8390380551559709  3.9289838866958187  2.9315807039718345
       -1.9389452077521723 -1.7800250513761013  2.9315807039718345  8.513514012094985
        """.split(),
        float,
    ).reshape(4, 4)
    cov_sk_wide = np.array(
        """
        1.6537393783557195  0.37641044115167954 -0.6445260481866113 -1.6814579757095078
        0.37641044115167954 2.3531447364553637  0.6823312099602171 -1.2521386039850808
       -0.6445260481866113  0.6823312099602171  4.275095455868162   3.014773802129567
       -1.6814579757095078 -1.2521386039850808  3.014773802129567   8.083503772853819
        """.split(),
        float,
    ).reshape(4, 4)

    cov_sk_mean = np.array(
        """
        1.3069045522207179  0.49941085732934765 -0.5800578609689707 -1.8345726475079638
        0.49941085732934765 2.2721762731213047  0.8171635509777809 -1.6795400890559036
       -0.5800578609689707  0.8171635509777809  3.9084387296080854  2.7813397937792352
       -1.8345726475079638 -1.6795400890559036  2.7813397937792352  8.210170749503536
        """.split(),
        float,
    ).reshape(4, 4)

    res = robcov._naive_ledoit_wolf_shrinkage(x, 0)
    assert_allclose(res.cov, cov_sk, rtol=1e-13)

    # the shrinkage intensity uses the centered data
    res = robcov._naive_ledoit_wolf_shrinkage(x, x.mean(0))
    assert_allclose(res.cov, cov_sk_mean, rtol=1e-13)

    # fewer observations than variables, the second moment matrix is singular
    # and the shrinkage estimate is positive definite
    res = robcov._naive_ledoit_wolf_shrinkage(x[:3], 0)
    assert_allclose(res.cov, cov_sk_wide, rtol=1e-13)
    assert np.linalg.eigvalsh(res.cov).min() > 0.1


@pytest.mark.parametrize(
    "x",
    [
        # x'x / 4 is the identity matrix
        np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]),
        np.array([[1.0], [2.0], [3.0]]),
        np.zeros((5, 3)),
    ],
    ids=["identity", "one variable", "zero"],
)
def test_naive_ledoit_wolf_shrinkage_spherical(x):
    # The empirical covariance is mu * I, so that delta is 0 and the
    # shrinkage was 0 / 0 = nan. The estimate is the empirical covariance.
    res = robcov._naive_ledoit_wolf_shrinkage(x, 0)
    assert_allclose(res.cov, x.T.dot(x) / x.shape[0], rtol=1e-13)


def test_cov_starting_small_nobs():
    # the first deterministic starting percentile is
    # 200 * (k_vars + 2) / nobs, which is above 100 when
    # nobs < 2 * k_vars + 4, so np.percentile raised a ValueError.
    # The first starting subset should then fall back to the full sample.
    rng = np.random.default_rng(10241)
    x = rng.standard_normal((60, 30))
    k_vars = x.shape[1]

    starts = robcov._cov_starting(x)
    assert_allclose(starts[0].mean, x.mean(axis=0))
    assert_allclose(starts[0].cov, np.cov(x.T))
    assert np.linalg.matrix_rank(starts[0].cov) == k_vars

    # with standardization, the full-sample start is rotated back to the
    # covariance of the original data
    starts_std = robcov._cov_starting(x, standardize=True, retransform=True)
    assert_allclose(starts_std[0], np.cov(x.T))

    # the estimators that use the starting sets should not raise
    for res in (
        robcov.CovDetMCD(x).fit(45),
        robcov.CovDetS(x).fit(),
        robcov.CovDetMM(x).fit(),
    ):
        assert res.cov.shape == (30, 30)
        assert np.isfinite(res.cov).all()
        assert np.linalg.eigvalsh(res.cov).min() > 0


def test_cov_starting_keeps_trimmed_starts():
    # starts must not be dropped when a trim retains at most k_vars
    # observations; that happens at sizes where the estimator already
    # worked on main, e.g. n=100, k=30 has 25 observations in the
    # 25% trim. All four percentile trims contribute four starts each,
    # plus six global starts.
    rng = np.random.default_rng(10242)
    starts = robcov._cov_starting(rng.standard_normal((100, 30)))
    assert len(starts) == 22
    # rank-deficient trimmed starts are retained and used for ranking
    ranks = [np.linalg.matrix_rank(start.cov) for start in starts]
    assert min(ranks) < 30


def test_mahalanobis_singular_cov():
    # a rank-deficient starting covariance must not abort the candidate, the
    # pseudo-inverse of the covariance is used
    rng = np.random.default_rng(10243)
    x = rng.standard_normal((40, 5))

    # the pseudo-inverse of the zero matrix is zero
    d = robcov.mahalanobis(x, cov=np.zeros((5, 5)))
    assert_allclose(d, 0, atol=1e-15)

    # diagonal rank 3 covariance, only the first three coordinates count
    d = robcov.mahalanobis(x, cov=np.diag([1.0, 1.0, 1.0, 0.0, 0.0]))
    assert_allclose(d, (x[:, :3] ** 2).sum(1), rtol=1e-12)


def test_mahalanobis_singular_cov_additional():
    rng = np.random.default_rng(10243)
    x = rng.standard_normal((40, 5))
    a = rng.standard_normal((5, 3))
    # Linear algebra for the pseudo-inverse of a rank-deficient covariance
    # is unreliable on WASM
    pinv = a @ np.linalg.inv(a.T @ a) @ np.linalg.inv(a.T @ a) @ a.T
    d = robcov.mahalanobis(x, cov=a @ a.T)
    assert_allclose(d, np.einsum("ij,jk,ik->i", x, pinv, x), rtol=1e-8)

    d_sqrt = robcov.mahalanobis(x, cov=a @ a.T, sqrt=True)
    assert_allclose(d_sqrt, np.sqrt(d), rtol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_mahalanobis_numerically_singular_cov(seed):
    # GH-10247: A rank deficient covariance with rounding errors does not have
    # an exactly zero pivot in the LU factorization on some platforms (macOS,
    # WASM), where numpy.linalg.solve returns huge values without an error. The
    # rotation of the rank 3 covariance a a' by an orthogonal matrix has these
    # rounding errors on all platforms. The pseudo-inverse of b b' is
    # b (b'b)^(-2) b' if b has full column rank.
    rng = np.random.default_rng(10243 + seed)
    x = rng.standard_normal((40, 5))
    q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    b = q @ rng.standard_normal((5, 3))
    cov = b @ b.T
    cov = (cov + cov.T) / 2
    pinv = b @ np.linalg.inv(b.T @ b) @ np.linalg.inv(b.T @ b) @ b.T
    d = robcov.mahalanobis(x, cov=cov)
    assert_allclose(d, np.einsum("ij,jk,ik->i", x, pinv, x), rtol=1e-8)
    assert np.all(d >= 0)


def test_mahalanobis_ill_conditioned_cov():
    # a covariance that is ill-conditioned but not singular to working precision
    # does not use the pseudo-inverse, the distance is the one of the inverse
    rng = np.random.default_rng(10243)
    x = rng.standard_normal((40, 5))
    q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    evals = np.array([1.0, 1e-3, 1e-5, 1e-8, 1e-10])
    cov = (q * evals) @ q.T
    d = robcov.mahalanobis(x, cov=(cov + cov.T) / 2)
    assert_allclose(d, ((x @ q) ** 2 / evals).sum(1), rtol=1e-4)


@pytest.mark.parametrize("span", [1e8, 1e16, 1e24])
def test_mahalanobis_badly_scaled_cov(span):
    # The covariance of variables with a ratio of variances above 1e15 has a
    # numerical rank below its dimension but it is not singular, the
    # pseudo-inverse would drop the variables with a small variance.
    # The distance does not depend on the scales, z' corr^(-1) z.
    rng = np.random.default_rng(10243)
    corr = np.array(
        [
            [1.0, 0.5, 0.2, 0.1],
            [0.5, 1.0, 0.3, 0.2],
            [0.2, 0.3, 1.0, 0.4],
            [0.1, 0.2, 0.4, 1.0],
        ]
    )
    z = rng.standard_normal((50, 4)) @ np.linalg.cholesky(corr).T
    expected = np.einsum("ij,jk,ik->i", z, np.linalg.inv(corr), z)

    sd = np.geomspace(span**0.25, span**-0.25, 4)
    cov = sd[:, None] * corr * sd
    assert_allclose(robcov.mahalanobis(z * sd, cov=cov), expected, rtol=1e-8)


def test_mahalanobis_cov_not_square():
    # the error for a cov that is not square is not replaced by the fallback
    # for a singular matrix
    x = np.ones((10, 5))
    with pytest.raises(np.linalg.LinAlgError, match="square"):
        robcov.mahalanobis(x, cov=np.ones((5, 4)))


def test_cov_weighted_det_non_finite_determinant():
    # a determinant that overflows plain det is normalized via slogdet
    # (GH-10287): det(wcov) == 1 instead of a LinAlgError
    x = np.eye(6) * 1e30
    cov, _ = robcov.cov_weighted(
        x, np.ones(6), center=np.zeros(6), weights_cov_denom="det"
    )
    assert np.isfinite(cov).all()
    assert_allclose(np.linalg.det(cov), 1.0, rtol=1e-6)

    # a singular cross product has det == 0 and must raise as well
    x = np.diag([1.0, 0.0])
    with pytest.raises(np.linalg.LinAlgError, match="must be positive and finite"):
        robcov.cov_weighted(
            x, np.ones(2), center=np.zeros(2), weights_cov_denom="det"
        )

    # a finite determinant is normalized as before: for diag(2, 3) the
    # cross product is diag(4, 9) with det 36, so det ** (1 / 2) is 6
    x = np.diag([2.0, 3.0])
    cov, _ = robcov.cov_weighted(
        x, np.ones(2), center=np.zeros(2), weights_cov_denom="det"
    )
    assert_allclose(cov, np.diag([4 / 6, 9 / 6]), rtol=1e-15)


def test_covdetmcd_rank_by_logdet():
    # GH-10295: det(cov) overflows to inf for every starting set, so
    # np.argmin(det_all) silently picked the first start and the selected
    # start changed with the scale of the data.
    # The seed is chosen so that the best start is not the first one.
    rng = np.random.default_rng(11)
    x = rng.standard_normal((100, 30)) * 5e5
    h = 65

    res = robcov.CovDetMCD(x).fit(h, maxiter_step=2, reweight=False)
    res_scaled = robcov.CovDetMCD(x / 4.0).fit(h, maxiter_step=2, reweight=False)

    # the starts of the unscaled data overflow to inf for every determinant
    assert np.all(np.isinf(res.det_all))
    # determinants of the scaled data are finite
    assert np.all(np.isfinite(res_scaled.det_all))
    # the overflowing fit selects the same start as the old determinant
    # criterion evaluated on the finite, scaled determinants
    assert res.idx_best == np.argmin(res_scaled.det_all)
    # ranking by log-determinant is scale-equivariant ...
    assert res.idx_best == res_scaled.idx_best
    assert res.idx_best != 0
    # ... and exactly preserves a power-of-two rescaling
    assert_allclose(res.mean, 4.0 * res_scaled.mean)
    assert_allclose(res.cov, 16.0 * res_scaled.cov)


def test_covdetmm():

    # results from rrcov
    # CovMMest(x = hbk, eff.shape=FALSE,
    #          control=CovControlMMest(sest=CovControlSest(method="sdet")))
    cov_dmm_r = np.array(
        """
        1.72174266670826 0.06925842715939 0.20781848922667 0.10749343153015
        0.06925842715939 1.74566218886362 0.22161135221404 -0.00517023660647
        0.20781848922667 0.22161135221404 1.63937749762534 -0.17217102475913
        0.10749343153015 -0.00517023660647 -0.17217102475913 0.48174480967136
        """.split(),
        float,
    ).reshape(4, 4)

    mean_dmm_r = np.array(
        [1.5388643420460, 1.8027582110408, 1.6811517253521, -0.0755069488908]
    )

    # using same c as rrcov
    c = 5.81031555752526
    mod = robcov.CovDetMM(dta_hbk, norm=robnorms.TukeyBiweight(c=c))
    res = mod.fit()

    assert_allclose(res.mean, mean_dmm_r, rtol=1e-5)
    assert_allclose(res.cov, cov_dmm_r, rtol=1e-5, atol=1e-5)

    # using c from table,
    mod = robcov.CovDetMM(dta_hbk)
    res = mod.fit()

    assert_allclose(res.mean, mean_dmm_r, rtol=1e-3)
    assert_allclose(res.cov, cov_dmm_r, rtol=1e-3, atol=1e-3)


def test_covdet_one_column():
    # GH-10300: np.cov returns a 0-d array for data with a single column,
    # so the covariance shape was wrong in CovDetMCD, CovDetS, CovDetMM
    # and in the no-start branch of CovM.
    rng = np.random.default_rng(3)
    x = rng.standard_normal((50, 1))
    h = 26

    for res in (
        robcov.CovDetMCD(x).fit(h),
        robcov.CovDetS(x).fit(),
        robcov.CovDetMM(x).fit(),
        robcov.CovM(x).fit(),
    ):
        cov = np.asarray(res.cov)
        assert cov.shape == (1, 1)
        assert np.all(np.isfinite(cov))
        assert cov[0, 0] > 0

    # Accuracy on a constructed dataset with an unambiguous optimum: a tight
    # core of h points plus 24 far outliers. The C-steps converge to the core
    # for any correct implementation, i.e. to the minimum-variance sliding
    # window of the sorted data, which is the exact univariate MCD solution.
    core = 10 + 0.01 * rng.standard_normal((h, 1))
    outliers = np.concatenate([np.full((12, 1), -100.0), np.full((12, 1), 100.0)])
    x2 = np.concatenate([core, outliers])
    raw = robcov.CovDetMCD(x2).fit(h).results_raw

    x2_sorted = np.sort(x2[:, 0])
    windows = np.lib.stride_tricks.sliding_window_view(x2_sorted, h)
    window_vars = windows.var(axis=1, ddof=1)
    idx_min = np.argmin(window_vars)
    assert_allclose(raw.det_subset, window_vars[idx_min], rtol=1e-12)
    fac = robcov.coef_normalize_cov_truncated(h / x2.shape[0], 1)
    assert_allclose(raw.cov, fac * window_vars[idx_min], rtol=1e-12)
    assert_allclose(raw.mean[0], x2_sorted[idx_min : idx_min + h].mean(), rtol=1e-12)


def test_cov_ogk_one_column():
    # np.cov returns a 0-d array for a single column, so that the reweighted
    # covariance of cov_ogk had shape ()
    rng = np.random.default_rng(3)
    x = 1 + 2 * rng.standard_normal((20000, 1))
    res = robcov.cov_ogk(x)
    assert res.cov.shape == (1, 1)
    assert res.cov_raw.shape == (1, 1)
    assert res.mean.shape == (1,)
    # both are consistent for the variance at the normal distribution
    assert_allclose(res.cov, [[4.0]], rtol=0.05)
    assert_allclose(res.cov_raw, [[4.0]], rtol=0.05)
    assert_allclose(res.mean, [1.0], atol=0.1)
    # the reweighted covariance is the variance of the kept observations,
    # rescaled for the truncation
    kept = x[res.mask]
    assert_allclose(res.cov, np.var(kept, ddof=1) * res.scale_factor, rtol=1e-12)
    # without reweighting the result is the raw estimate
    res0 = robcov.cov_ogk(x, reweight=None)
    assert res0.cov.shape == (1, 1)
    assert_allclose(res0.cov, res.cov_raw)


def test_cov_tyler_regularized_one_column():
    # The plugin for the shrinkage factor is 0 / 0 for a single variable, and
    # the normalized scatter of a single variable is 1.
    rng = np.random.default_rng(3)
    x = rng.standard_normal((50, 1))
    res = robcov.cov_tyler_regularized(x)
    assert res.cov.shape == (1, 1)
    assert_allclose(res.cov, [[1.0]])
    assert res.shrinkage_factor == 0
    res = robcov.cov_tyler_regularized(x, shrinkage_factor=0.1)
    assert_allclose(res.cov, [[1.0]])
    assert res.shrinkage_factor == 0.1


def test_cov_iter_one_column():
    rng = np.random.default_rng(3)
    x = 1 + rng.standard_normal((50, 1))
    res = robcov._cov_iter(x, robcov.weights_quantile, weights_args=(0.5,))
    assert res.cov.shape == (1, 1)
    # the default starting value is the sample variance
    cov_init = np.var(x, ddof=1, axis=0)[None, :]
    res2 = robcov._cov_iter(
        x, robcov.weights_quantile, weights_args=(0.5,), cov_init=cov_init
    )
    assert_allclose(res.cov, res2.cov, rtol=1e-12)
    assert_allclose(res.mean, res2.mean, rtol=1e-12)


def test_naive_ledoit_wolf_one_column():
    # the shrinkage target is the empirical covariance for a single variable,
    # the shrinkage intensity was 0 / 0
    rng = np.random.default_rng(3)
    x = rng.standard_normal((50, 1))
    res = robcov._naive_ledoit_wolf_shrinkage(x, 0)
    assert_allclose(res.cov, [[np.mean(x**2)]], rtol=1e-12)


@pytest.mark.parametrize("nobs", [50, 51])
def test_cov_starting_one_column(nobs):
    rng = np.random.default_rng(3)
    x = 1 + 2 * rng.standard_normal((nobs, 1))
    res = robcov._cov_starting(x)
    for r in res:
        cov = np.asarray(r.cov)
        assert cov.shape == (1, 1)
        assert np.all(np.isfinite(cov))
        assert cov[0, 0] > 0

    xs = x - np.median(x, axis=0)
    # the first start is the covariance of the observations below the first
    # percentile of the squared distances, rescaled for the truncation
    p = min(3 / nobs * 200, 100)
    d = xs[:, 0] ** 2
    xsp = xs[d < np.percentile(d, p)]
    expected = np.var(xsp, ddof=1) * robcov.coef_normalize_cov_truncated(p / 100, 1)
    assert res[0].method == "pearson truncated"
    assert_allclose(res[0].cov, [[expected]], rtol=1e-12)

    # correlation matrices of a single variable are 1, the spatial sign of a
    # single variable is the sign, which is 0 for the median if nobs is odd
    tanh, spatial, spearman, normal_scores = res[-4:]
    assert [r.method for r in res[-4:]] == [
        "tanh",
        "spatial",
        "spearman",
        "normal-scores",
    ]
    for r in (tanh, spearman, normal_scores):
        assert_allclose(r.cov, [[1.0]])
    assert_allclose(spatial.cov, [[np.var(np.sign(xs), ddof=1)]], rtol=1e-12)

    # standardized and retransformed results have the same shape
    for kwds in ({"standardize": True}, {"standardize": True, "retransform": True}):
        for r in robcov._cov_starting(x, **kwds):
            assert np.shape(getattr(r, "cov", r)) == (1, 1)


def test_det_root_does_not_overflow():
    # det(10 * I_400) = 10**400 overflows to inf; its 400th root is 10.
    assert_allclose(robcov._det_root(10 * np.eye(400)), 10.0, rtol=1e-12)
    a = np.random.default_rng(0).standard_normal((20, 5))
    cov = a.T @ a
    assert_allclose(robcov._det_root(cov), np.linalg.det(cov) ** (1 / 5), rtol=1e-12)
    # singular matrices keep det(cov) ** (1 / k), which is 0
    assert_equal(robcov._det_root(np.zeros((3, 3))), 0.0)


@pytest.mark.parametrize("cov_class", [robcov.CovDetS, robcov.CovDetMM])
def test_covdet_large_k_det_normalization(cov_class):
    # GH-10257: normalizing to det(cov) = 1 overflowed for large k_vars and the
    # fit raised LinAlgError: Singular matrix.
    x = np.random.default_rng(0).standard_normal((400, 149))
    res = cov_class(x).fit()
    assert np.all(np.isfinite(res.cov))


def test_cov_tyler_regularized_n_iter():
    # GH: n_iter was accumulated as `n_iter += i` inside `for i in
    # range(maxiter)`, giving a triangular-number count instead of the
    # actual number of completed iterations.
    rs = np.random.RandomState(0)
    x = rs.standard_normal((50, 3))
    res = robcov.cov_tyler_regularized(x, shrinkage_factor=0.1, maxiter=4, eps=0)
    assert res.n_iter == 4


def test_cov_tyler_regularized_corr_uses_scale():
    # GH: `corr * np.outer(scale_mad, scale_mad)` discarded its result
    # instead of `corr *= ...`, so the plugin-shrinkage `corr` matrix never
    # reflected the per-column MAD scale, only the (scale-free) correlation
    # structure from the std-standardized data.
    rs = np.random.RandomState(0)
    nobs = 1000
    x = rs.standard_normal((nobs, 2))
    x[:, 0] *= 100  # column 0 has a much larger scale than column 1

    res = robcov.cov_tyler_regularized(x, shrinkage_factor=None)
    ratio = res.corr[0, 0] / res.corr[1, 1]
    assert ratio > 100


def test_robcov_SMOKE():
    # currently only smoke test or very loose comparisons to dgp
    nobs, k_vars = 100, 3

    mean = np.zeros(k_vars)
    cov = linalg.toeplitz(1.0 / np.arange(1, k_vars + 1))

    rs = np.random.RandomState(187649)
    x = rs.multivariate_normal(mean, cov, size=nobs)
    n_outliers = 1
    x[0, :2] = 50

    # xtx = x.T.dot(x)
    # cov_emp = np.cov(x.T)
    cov_clean = np.cov(x[n_outliers:].T)

    # GK, OGK
    robcov.cov_gk1(x[:, 0], x[:, 2])
    robcov.cov_gk(x)
    robcov.cov_ogk(x)

    # Tyler
    robcov.cov_tyler(x)
    robcov.cov_tyler_regularized(x, shrinkage_factor=0.1)

    x2 = np.array([x, x])
    x2_ = np.rollaxis(x2, 1)
    robcov.cov_tyler_pairs_regularized(
        x2_,
        start_cov=np.diag(robust.mad(x) ** 2),
        shrinkage_factor=0.1,
        nobs=x.shape[0],
        k_vars=x.shape[1],
    )

    # others, M-, ...

    # estimation for multivariate t
    r = robcov._cov_iter(x, robcov.weights_mvt, weights_args=(3, k_vars))
    # rough comparison with DGP cov
    assert_allclose(r.cov, cov, rtol=0.5)

    # trimmed sample covariance
    r = robcov._cov_iter(
        x, robcov.weights_quantile, weights_args=(0.50,), rescale="med"
    )
    # rough comparison with DGP cov
    assert_allclose(r.cov, cov, rtol=0.5)

    # We use 0.75 quantile for truncation to get better efficiency
    # at q=0.5, cov is pretty noisy at nobs=100 and passes at rtol=1
    res_li = robcov._cov_starting(x, standardize=True, quantile=0.75)
    for _, res in enumerate(res_li):
        # note: basic cov are not properly scaled
        # check only those with _cov_iter rescaling, `n_iter`
        # include also ogk
        # need more generic detection of appropriate cov
        if hasattr(res, "n_iter") or hasattr(res, "cov_ogk_raw"):
            # inconsistent returns, redundant for now b/c no arrays
            c = getattr(res, "cov", res)
            # rough comparison with DGP cov
            assert_allclose(c, cov, rtol=0.5)
            # check average scaling
            assert_allclose(np.diag(c).sum(), np.diag(cov).sum(), rtol=0.25)
            c1, m1 = robcov._reweight(x, res.mean, res.cov)
            assert_allclose(c1, cov, rtol=0.4)
            assert_allclose(c1, cov_clean, rtol=1e-8)  # oracle, w/o outliers
            assert_allclose(m1, mean, rtol=0.5, atol=0.2)


def test_cov_iter_invalid_rescale_raises():
    rs = np.random.RandomState(28345)
    x = rs.standard_normal((50, 3))
    with pytest.raises(ValueError, match="rescale"):
        robcov._cov_iter(
            x, robcov.weights_quantile, weights_args=(0.50,), rescale="not-a-rescale"
        )


def test_get_detcov_startidx_invalid_methods_cov_raises():
    rs = np.random.RandomState(28346)
    z = rs.standard_normal((50, 3))
    with pytest.raises(ValueError, match="methods_cov"):
        robcov._get_detcov_startidx(z, 30, methods_cov="not-all")


def _detmcd_data():
    # deterministic data without random numbers, 5 outliers in 40 observations
    n = 40
    i = np.arange(1, n + 1)
    x1 = np.sin(i) + 0.5 * np.cos(3 * i)
    x2 = np.cos(2 * i) + 0.3 * np.sin(5 * i)
    x3 = np.sin(3 * i + 1) + 0.2 * i / n
    x4 = np.cos(i / 2) + 0.4 * np.sin(7 * i)
    x2 = x2 + 0.6 * x1
    x3 = x3 + 0.4 * x1 - 0.3 * x2
    x = np.column_stack([x1, x2, x3, x4])
    x[[2, 10, 18, 26, 34]] += np.array([4, -3, 3.5, -4])
    return x


def test_get_detcov_startidx_r_deterministic_mcd():
    # The starting sets of the deterministic MCD of Hubert, Rousseeuw and
    # Verdonck (2012) for the tanh and the Spearman starting correlation, 1-based
    # indices from the function initset in r6pack of R robustbase 0.99.7 (the
    # h observations with the smallest distance after the orthogonalization,
    # standardization with median and Qn without the finite sample correction):
    #
    # z <- sweep(x, 2, apply(x, 2, median))
    # z <- sweep(z, 2, apply(z, 2, function(v) Qn(v, finite.corr = FALSE)), "/")
    # P <- eigen(cor(tanh(z)), symmetric = TRUE)$vectors   # or cor(z, method="spearman")
    # initset(z, scalefn, P, h)
    #
    # The distance used the wrong arguments of mahalanobis, mean was passed as
    # cov and cov as cov_inv.
    r_sets = {
        "tanh": [4, 6, 7, 8, 9, 10, 12, 15, 16, 20, 21, 22, 23, 25, 26, 28, 29, 31,
                 32, 33, 34, 40],
        "spearman": [4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 20, 21, 22, 23, 25, 26, 28,
                     29, 31, 32, 33, 34],
    }
    x = _detmcd_data()
    starts = robcov._get_detcov_startidx(
        x, 22, options_start={"loc_func": robcov.median, "scale_func": robscale.qn_scale}
    )
    methods = [method for _, method in starts]
    for method, expected in r_sets.items():
        idx = starts[methods.index(method)][0]
        assert_equal(np.sort(idx) + 1, expected)


def test_get_detcov_startidx_ranks_by_mahalanobis_distance():
    # every starting set contains the h observations with the smallest squared
    # distance (z - mean)' cov^{-1} (z - mean) of its orthogonalized estimate
    x = _detmcd_data()
    h = 22
    z = (x - robcov.median(x)) / robcov.mad(x)
    starts = robcov._get_detcov_startidx(x, h)
    covs = robcov._cov_starting(z, standardize=False, quantile=0.5)
    covs = [c for c in covs if hasattr(c, "method")]
    assert len(starts) == len(covs)
    for (idx, method), c in zip(starts, covs, strict=True):
        assert method == c.method
        mean, cov = robcov._orthogonalize_det(z, c.cov, robcov.median, robcov.mad)
        resid = z - mean
        d = np.einsum("ij,ij->i", resid, np.linalg.solve(cov, resid.T).T)
        assert_equal(np.sort(idx), np.sort(np.argsort(d)[:h]))
