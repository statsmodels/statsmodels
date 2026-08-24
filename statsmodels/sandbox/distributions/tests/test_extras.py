"""
Created on Sun Apr 17 22:13:36 2011

@author: josef
"""

from statsmodels.compat.scipy import SP_LT_19, SP_LT_116

import numpy as np
from numpy.testing import assert_, assert_allclose, assert_almost_equal
import pytest
from scipy import stats

from statsmodels.sandbox.distributions.extras import (
    ACSkewT_gen,
    ExpTransf_gen,
    LogTransf_gen,
    NormExpan_gen,
    get_u_argskwargs,
    invdnormalg,
    loggammaexpg,
    lognormalg,
    mvnormcdf,
    mvstdnormcdf,
    pdf_moments,
    pdf_moments_st,
    pdf_mvsk,
    skewnorm,
    skewnorm2,
)
from statsmodels.stats.moment_helpers import mvsk2mc


def test_skewnorm():
    # library("sn")
    # dsn(c(-2,-1,0,1,2), shape=10)
    # psn(c(-2,-1,0,1,2), shape=10)
    # noquote(sprintf("%.15e,", snp))
    pdf_r = np.array(
        [
            2.973416551551523e-90,
            3.687562713971017e-24,
            3.989422804014327e-01,
            4.839414490382867e-01,
            1.079819330263761e-01,
        ]
    )
    pdf_sn = skewnorm.pdf([-2, -1, 0, 1, 2], 10)

    # res = (snp-snp_r)/snp
    assert_(np.allclose(pdf_sn, pdf_r, rtol=1e-13, atol=0))

    pdf_sn2 = skewnorm2.pdf([-2, -1, 0, 1, 2], 10)
    assert_(np.allclose(pdf_sn2, pdf_r, rtol=1e-13, atol=0))

    cdf_r = np.array(
        [
            0.000000000000000e00,
            0.000000000000000e00,
            3.172551743055357e-02,
            6.826894921370859e-01,
            9.544997361036416e-01,
        ]
    )
    cdf_sn = skewnorm.cdf([-2, -1, 0, 1, 2], 10)
    # maxabs = np.max(np.abs(cdf_sn - cdf_r))
    # maxrel = np.max(np.abs(cdf_sn - cdf_r) / (cdf_r + 1e-50))
    # msg = f"maxabs={maxabs:15.13g}, maxrel={maxrel:15.13g}\n{cdf_sn!r}\n{cdf_r!r}"
    # assert_(np.allclose(cdf_sn, cdf_r, rtol=1e-13, atol=1e-25), msg=msg)
    assert_almost_equal(cdf_sn, cdf_r, decimal=10)

    cdf_sn2 = skewnorm2.cdf([-2, -1, 0, 1, 2], 10)
    maxabs = np.max(np.abs(cdf_sn2 - cdf_r))
    maxrel = np.max(np.abs(cdf_sn2 - cdf_r) / (cdf_r + 1e-50))
    msg = f"maxabs={maxabs:15.13g}, maxrel={maxrel:15.13g}"
    # assert_(np.allclose(cdf_sn2, cdf_r, rtol=1e-13, atol=1e-25), msg=msg)
    assert_almost_equal(cdf_sn2, cdf_r, decimal=10, err_msg=msg)


def test_skewt():
    skewt = ACSkewT_gen()
    x = [-2, -1, -0.5, 0, 1, 2]
    # noquote(sprintf("%.15e,", dst(c(-2,-1, -0.5,0,1,2), shape=10)))
    # default in R:sn is df=inf
    pdf_r = np.array(
        [
            2.973416551551523e-90,
            3.687562713971017e-24,
            2.018401586422970e-07,
            3.989422804014327e-01,
            4.839414490382867e-01,
            1.079819330263761e-01,
        ]
    )
    pdf_st = skewt.pdf(x, 1000000, 10)
    np.allclose(pdf_st, pdf_r, rtol=0, atol=1e-6)
    np.allclose(pdf_st, pdf_r, rtol=1e-1, atol=0)

    # noquote(sprintf("%.15e,", pst(c(-2,-1, -0.5,0,1,2), shape=10)))
    cdf_r = np.array(
        [
            0.000000000000000e00,
            0.000000000000000e00,
            3.729478836866917e-09,
            3.172551743055357e-02,
            6.826894921370859e-01,
            9.544997361036416e-01,
        ]
    )
    cdf_st = skewt.cdf(x, 1000000, 10)
    np.allclose(cdf_st, cdf_r, rtol=0, atol=1e-6)
    np.allclose(cdf_st, cdf_r, rtol=1e-1, atol=0)
    # assert_(np.allclose(cdf_st, cdf_r, rtol=1e-13, atol=1e-15))

    # noquote(sprintf("%.15e,", dst(c(-2,-1, -0.5,0,1,2), shape=10, df=5)))
    pdf_r = np.array(
        [
            2.185448836190663e-07,
            1.272381597868587e-05,
            5.746937644959992e-04,
            3.796066898224945e-01,
            4.393468708859825e-01,
            1.301804021075493e-01,
        ]
    )
    pdf_st = skewt.pdf(x, 5, 10)  # args = (df, alpha)
    assert_(np.allclose(pdf_st, pdf_r, rtol=1e-13, atol=1e-25))

    # noquote(sprintf("%.15e,", pst(c(-2,-1, -0.5,0,1,2), shape=10, df=5)))
    cdf_r = np.array(
        [
            8.822783669199699e-08,
            2.638467463775795e-06,
            6.573106017198583e-05,
            3.172551743055352e-02,
            6.367851708183412e-01,
            8.980606093979784e-01,
        ]
    )
    cdf_st = skewt.cdf(x, 5, 10)  # args = (df, alpha)
    assert_(np.allclose(cdf_st, cdf_r, rtol=1e-10, atol=0))

    # noquote(sprintf("%.15e,", dst(c(-2,-1, -0.5,0,1,2), shape=10, df=1)))
    pdf_r = np.array(
        [
            3.941955996757291e-04,
            1.568067236862745e-03,
            6.136996029432048e-03,
            3.183098861837907e-01,
            3.167418189469279e-01,
            1.269297588738406e-01,
        ]
    )
    pdf_st = skewt.pdf(x, 1, 10)  # args = (df, alpha) = (1, 10))
    assert_(np.allclose(pdf_st, pdf_r, rtol=1e-13, atol=1e-25))

    # noquote(sprintf("%.15e,", pst(c(-2,-1, -0.5,0,1,2), shape=10, df=1)))
    cdf_r = np.array(
        [
            7.893671370544414e-04,
            1.575817262600422e-03,
            3.128720749105560e-03,
            3.172551743055351e-02,
            5.015758172626005e-01,
            7.056221318361879e-01,
        ]
    )
    cdf_st = skewt.cdf(x, 1, 10)  # args = (df, alpha) = (1, 10)
    assert_(np.allclose(cdf_st, cdf_r, rtol=1e-13, atol=1e-25))


@pytest.mark.singleton_randomstate
@pytest.mark.xfail(
    condition=not SP_LT_19,
    reason=(
        "BUG: SkewNorm_gen._rvs(self, alpha) does not accept the `size` "
        "keyword argument that scipy.stats.rv_continuous.rvs() always "
        "passes to _rvs on modern scipy (it instead relies on the "
        "legacy self._size instance attribute set before calling _rvs). "
        "skewnorm.rvs(...) therefore always raises TypeError."
    ),
    raises=TypeError,
    strict=True,
)
def test_skewnorm_rvs():
    skewnorm.rvs(5, size=100, random_state=np.random.RandomState(0))


@pytest.mark.singleton_randomstate
@pytest.mark.xfail(
    condition=not SP_LT_19,
    reason=(
        "BUG: same root cause as test_skewnorm_rvs -- "
        "ACSkewT_gen._rvs(self, df, alpha) does not accept the `size` "
        "keyword argument scipy passes to _rvs, so ACSkewT_gen().rvs(...) "
        "always raises TypeError."
    ),
    raises=TypeError,
    strict=False,
)
def test_acskewt_rvs():
    ACSkewT_gen().rvs(5, 10, size=100, random_state=np.random.RandomState(0))


def test_pdf_mvsk_matches_pdf_moments():
    # pdf_mvsk takes (mean, central 2nd moment, skew, excess kurtosis);
    # pdf_moments takes raw central moments. Given equivalent inputs they
    # should produce the same expansion.
    mvsk = (0.0, 1.0, 0.5, 0.3)
    cnt = mvsk2mc(mvsk)
    f_mvsk = pdf_mvsk(mvsk)
    f_moments = pdf_moments(cnt)
    x = np.linspace(-3, 3, 13)
    assert_allclose(f_mvsk(x), f_moments(x), rtol=1e-12)


def test_pdf_moments_reduces_to_normal_for_zero_skew_kurt():
    mvsk = (0.0, 1.0, 0.0, 0.0)
    cnt = mvsk2mc(mvsk)
    f = pdf_moments(cnt)
    x = np.linspace(-3, 3, 13)
    assert_allclose(f(x), stats.norm.pdf(x), atol=1e-10)


def test_pdf_moments_requires_at_least_two_moments():
    with pytest.raises(ValueError, match="At least two moments"):
        pdf_moments([0.0])


def test_pdf_mvsk_requires_four_moments():
    with pytest.raises(ValueError, match="Four moments"):
        pdf_mvsk([0.0, 1.0])


@pytest.mark.xfail(
    reason=(
        "BUG: pdf_moments_st's inner loop does "
        "`for n in range((k - 3) / 2):`. In Python 3 `/` is true "
        "division, so this always passes a float to range(), which "
        "raises TypeError immediately for any call with >= 3 moments "
        "(the only case that reaches the loop). Even if that were fixed "
        "with `//`, the very next statement is an unconditional bare "
        "`raise SystemError`, so the function cannot currently produce a "
        "result either way; pdf_moments/pdf_mvsk are the intended "
        "non-broken replacements per this module's own docstring."
    ),
    raises=TypeError,
    strict=True,
)
def test_pdf_moments_st_is_broken():
    pdf_moments_st([0.0, 1.0, 0.0, 0.0])


class TestNormExpanGen:
    def test_centmom_mode_matches_mvsk_mode(self):
        mvsk = (1.0, 4.0, 0.3, 0.2)
        cnt = mvsk2mc(mvsk)
        dist_centmom = NormExpan_gen(cnt, mode="centmom")
        dist_mvsk = NormExpan_gen(mvsk, mode="mvsk")
        assert_allclose(dist_centmom.mvsk, dist_mvsk.mvsk, rtol=1e-12)
        x = np.linspace(-3, 5, 11)
        assert_allclose(dist_centmom.pdf(x), dist_mvsk.pdf(x), rtol=1e-12)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            NormExpan_gen((0.0, 1.0, 0.0, 0.0), mode="bogus")


def test_transf_gen_lognormalg_matches_underlying_distribution():
    # lognormalg = Transf_gen(stats.norm, exp, log, ...) is the
    # distribution of y=exp(x) for x standard normal
    x = np.array([0.5, 1.0, 2.0, 5.0])
    assert_allclose(lognormalg.cdf(x), stats.norm.cdf(np.log(x)), rtol=1e-10)
    # Transf_gen does not override _pdf, so scipy falls back to numerical
    # differentiation of the cdf -- compare against the analytical
    # change-of-variables formula with a looser tolerance
    assert_allclose(lognormalg.pdf(x), stats.norm.pdf(np.log(x)) / x, rtol=1e-6)


def test_transf_gen_invdnormalg_cdf_ppf_roundtrip():
    # invdnormalg is decr=True; a probability-level roundtrip through
    # ppf -> cdf should recover the same probability regardless of
    # direction
    for q in [0.1, 0.5, 0.9]:
        assert_allclose(invdnormalg.cdf(invdnormalg.ppf(q)), q, rtol=1e-6)


@pytest.mark.xfail(
    reason=(
        "BUG: loggammaexpg = Transf_gen(stats.gamma, log, exp, "
        "numargs=1) is meant to be called as loggammaexpg.cdf(x, a) "
        "for gamma shape parameter `a`, mirroring lognormalg's usage. "
        "But Transf_gen._cdf(self, x, *args, **kwargs) does not receive "
        "the shape argument by the time it calls "
        "self.kls._cdf(self.funcinv(x), *args, **kwargs) -- args is "
        "empty -- so stats.gamma._cdf() always raises TypeError for a "
        "missing required `a` argument."
    ),
    raises=TypeError,
    strict=True,
)
def test_transf_gen_loggammaexpg():
    loggammaexpg.cdf(1, 2)


def test_exptransf_gen_matches_underlying_distribution():
    # ExpTransf_gen wraps a distribution kls as the distribution of
    # y=exp(x) via cdf(y) = kls.cdf(log(y))
    et = ExpTransf_gen(stats.norm, numargs=0, name="exptest")
    x = np.array([0.5, 1.0, 2.0, 5.0])
    assert_allclose(et.cdf(x), stats.norm.cdf(np.log(x)), rtol=1e-10)


def test_logtransf_gen_matches_underlying_distribution():
    # LogTransf_gen wraps kls as the distribution of y=log(x) via
    # cdf(y) = kls.cdf(exp(y)). Its default lower bound is a=0, which is
    # only correct if the transformed variable's own range starts at 0;
    # for expon (support (0, inf)), log(x) ranges over all reals, so the
    # support bound must be passed explicitly as a=-np.inf.
    lt = LogTransf_gen(stats.expon, numargs=0, name="logtest", a=-np.inf)
    y = np.array([-1.0, 0.0, 1.0, 2.0])
    assert_allclose(lt.cdf(y), stats.expon.cdf(np.exp(y)), rtol=1e-10)


def test_logtransf_gen_default_lower_bound_clips_negative_support():
    # documents the a=0 default's effect: without overriding it, points
    # below the default lower bound are (silently) treated as outside the
    # support and report cdf=0, even though log(x) for x~expon can be
    # arbitrarily negative
    lt = LogTransf_gen(stats.expon, numargs=0, name="logtest")
    assert_allclose(lt.cdf([-1.0, 0.0]), [0.0, 0.0])


def test_get_u_argskwargs():
    u_args, u_kwargs = get_u_argskwargs(u_loc=1, u_scale=2, other=3)
    assert u_args is None
    assert u_kwargs == {"loc": 1, "scale": 2}


@pytest.mark.xfail(
    reason=(
        "BUG: get_u_argskwargs strips the 'u_' prefix from every kwarg "
        "key *before* trying to pop 'u_args' back out: "
        "`u_kwargs = {k.replace('u_', '', 1): v for k, v in kwargs.items() "
        "if k.startswith('u_')}` renames a passed `u_args=...` to key "
        "'args', then `u_kwargs.pop('u_args', None)` looks for a key that "
        "no longer exists (it's 'args' now), so it always returns the "
        "None default. A caller's u_args value is silently lost, and a "
        "stray 'args' key leaks into u_kwargs instead of being removed."
    ),
    raises=AssertionError,
    strict=True,
)
def test_get_u_argskwargs_u_args_is_silently_dropped():
    u_args, u_kwargs = get_u_argskwargs(u_args=(1, 2), u_loc=3, other=4)
    assert u_args == (1, 2)
    assert u_kwargs == {"loc": 3}


@pytest.mark.skipif(
    not SP_LT_116,
    reason=(
        "mvstdnormcdf/mvnormcdf delegate to mvndst, a compiled scipy.stats "
        "Fortran routine removed in SciPy >= 1.16.0"
    ),
)
class TestMvstdnormcdf:
    def test_mvstdnormcdf_matches_docstring_example(self):
        result = mvstdnormcdf([-np.inf, -np.inf], [0.0, np.inf], 0.5)
        assert_allclose(result, 0.5, atol=1e-6)

    def test_mvnormcdf_matches_mvstdnormcdf_when_standardized(self):
        corr = [[1.0, 0.5], [0.5, 1.0]]
        lower = [-np.inf, -np.inf]
        upper = [0.5, 1.0]
        result_std = mvstdnormcdf(lower, upper, corr, abseps=1e-6)
        result_norm = mvnormcdf(
            upper, mu=[0.0, 0.0], cov=corr, lower=lower, abseps=1e-6
        )
        assert_allclose(result_std, result_norm, atol=1e-6)

    def test_mvnormcdf_rescales_for_nonunit_variance(self):
        # scaling cov by a constant factor should not change the
        # standardized probability once mu/cov are used consistently
        std = np.array([2.0, 3.0])
        cov = np.diag(std) @ np.array([[1.0, 0.3], [0.3, 1.0]]) @ np.diag(std)
        mu = np.array([1.0, -1.0])
        upper = mu + std  # one std above the mean in each dimension
        result = mvnormcdf(upper, mu, cov, abseps=1e-6)
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        result_std = mvstdnormcdf([-np.inf, -np.inf], [1.0, 1.0], corr, abseps=1e-6)
        assert_allclose(result, result_std, atol=1e-6)
