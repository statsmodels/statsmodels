"""

Created on Thu Feb 28 13:24:59 2013

Author: Josef Perktold
"""
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pytest
from scipy import stats

from statsmodels.stats.gof import (
    chisquare,
    chisquare_effectsize,
    chisquare_power,
    gof_binning_discrete,
    gof_chisquare_discrete,
    powerdiscrepancy,
)
from statsmodels.tools.testing import Holder


def test_powerdiscrepancy_lambd_aliases():
    observed = np.array([2.0, 4.0, 2.0, 1.0, 1.0])
    expected = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    lambd_and_a = [
        ("loglikeratio", 0),
        ("freeman_tukey", -0.5),
        ("pearson", 1),
        ("modified_loglikeratio", -1),
        ("cressie_read", 2 / 3.0),
    ]
    for lambd, a in lambd_and_a:
        d_str, p_str = powerdiscrepancy(observed, expected, lambd=lambd)
        d_num, p_num = powerdiscrepancy(observed, expected, lambd=a)
        assert_almost_equal(d_str, d_num, decimal=13)
        assert_almost_equal(p_str, p_num, decimal=13)

    with pytest.raises(ValueError, match="lambd"):
        powerdiscrepancy(observed, expected, lambd="not-a-lambd")


def test_gof_binning_discrete_bernoulli():
    # With nsupp=20 and a 2-point support, each mass (0.7, 0.3) already
    # exceeds the minimum bin mass 1/20, so the algorithm's bins align
    # exactly with the distribution's support: bin 0 catches rvs==0 and
    # bin 1 catches rvs==1. This lets freq/expfreq be hand-verified
    # directly from the sample and from the Bernoulli pmf, independently
    # of the general binning algorithm.
    rng = np.random.default_rng(12345)
    p = 0.3
    n = 500
    rvs = (rng.random(n) < p).astype(int)
    distfn = stats.bernoulli
    arg = (p,)

    freq, expfreq, histsupp = gof_binning_discrete(rvs, distfn, arg, nsupp=20)

    assert_equal(freq, np.array([(rvs == 0).sum(), (rvs == 1).sum()]))
    assert_allclose(expfreq, n * np.array([1 - p, p]))
    assert freq.sum() == n
    expected_supp = [0.0, np.nextafter(0.0, 1.0), np.nextafter(1.0, 2.0)]
    assert_equal(histsupp, expected_supp)


def test_gof_chisquare_discrete_bernoulli():
    # Check the test against scipy.stats.chisquare with the
    # independently verified Bernoulli binning above.
    rng = np.random.default_rng(12345)
    p = 0.3
    n = 500
    rvs = (rng.random(n) < p).astype(int)
    distfn = stats.bernoulli
    arg = (p,)
    alpha = 0.05

    chis, pval, truefalse, outstr = gof_chisquare_discrete(
        distfn, arg, rvs, alpha, "bernoulli test"
    )

    freq, expfreq, _ = gof_binning_discrete(rvs, distfn, arg, nsupp=20)
    chis_expected, pval_expected = stats.chisquare(freq, expfreq)

    assert_almost_equal(chis, chis_expected, decimal=12)
    assert_almost_equal(pval, pval_expected, decimal=12)
    assert truefalse == (pval_expected > alpha)
    assert "bernoulli test" in outstr
    assert str(arg) in outstr
    assert str(pval) in outstr


@pytest.mark.parametrize("loc", [-4000, 0, 4000])
def test_gof_discrete_large_poisson(loc):
    # The support must not be truncated at -1000 or 1000 (GH-4359).
    rng = np.random.default_rng(24)
    sample = stats.poisson.rvs(2000, size=2000, random_state=rng) + loc
    arg = (2000, loc)
    freq, expfreq, edges = gof_binning_discrete(sample, stats.poisson, arg)

    assert len(freq) == 19
    assert freq.sum() == len(sample)
    assert_allclose(expfreq.sum(), len(sample))
    # Independent probabilities from SciPy, including the unbounded upper tail.
    expected = len(sample) * np.diff(stats.poisson.cdf(edges, *arg))
    assert_allclose(expfreq, expected)
    assert np.all(expfreq > 5)
    reference = stats.chisquare(freq, expected)
    chis, pval, passed, _ = gof_chisquare_discrete(
        stats.poisson, arg, sample, 0.05, "large Poisson"
    )
    assert np.isfinite(pval)
    assert_allclose([chis, pval], reference)
    assert passed == (reference.pvalue > 0.05)

    # The ppf based binning is checked against the cdf scan rule of the previous
    # implementation (an internal equivalence check, not an external standard),
    # reproduced in R 4.5.2 by scanning CDF values independently of the PPF search:
    # np.savetxt("sample.csv", sample - loc, fmt="%d")
    # x <- scan("sample.csv")
    # last <- 0; cuts <- c()
    # for (k in 0:5000) {
    #     current <- ppois(k, 2000)
    #     if (current - last >= 1/20 - 1e-14) {
    #         cuts <- c(cuts, k); last <- current
    #         if (current > 1 - 1/20) break
    #     }
    # }
    # breaks <- c(0, cuts + 0.5, Inf)
    # observed <- table(cut(x, breaks, right=FALSE, include.lowest=TRUE))
    # chisq.test(observed, p=diff(ppois(breaks, 2000)))
    assert_allclose([chis, pval], [20.3236798472953, 0.314894584643616], rtol=1e-11)
    reference_cuts = [
        1927, 1943, 1954, 1963, 1971, 1978, 1985, 1991, 1997,
        2003, 2009, 2015, 2022, 2029, 2037, 2046, 2058, 2076,
    ]
    assert_equal(np.floor(edges[1:-1]) - loc, reference_cuts)


@pytest.mark.parametrize("low", [-2020, 0, 2000])
@pytest.mark.parametrize("nsupp", [1, 4, 20])
def test_gof_binning_discrete_randint(low, nsupp):
    # Each of the 20 integers has probability 1/20. These expected bins
    # and counts follow directly, without using the binning implementation.
    counts = 5 * np.arange(1, 21)
    sample = np.repeat(np.arange(low, low + 20), counts)
    freq, expfreq, edges = gof_binning_discrete(
        sample, stats.randint, (low, low + 20), nsupp=nsupp
    )
    assert_equal(freq, counts.reshape(nsupp, 20 // nsupp).sum(axis=1))
    assert_allclose(expfreq, np.full(nsupp, len(sample) / nsupp))
    expected_edges = np.r_[low, low - 1 + np.arange(1, nsupp + 1) * (20 // nsupp)]
    expected_edges = expected_edges.astype(float)
    expected_edges[1:] = np.nextafter(expected_edges[1:], np.inf)
    assert_equal(edges, expected_edges)


@pytest.mark.parametrize("loc", [-2000, 0, 2000])
def test_gof_binning_discrete_unbounded_support(loc):
    sample = np.repeat([-10000, -2, -1, 0, 1, 2, 10000], [1, 10, 10, 10, 10, 10, 1])
    sample = sample + loc
    freq, expfreq, edges = gof_binning_discrete(sample, stats.dlaplace, (1, loc))
    # For a=1, CDF(-3) < .05 < CDF(-2), and CDF(2) > .95.
    # Interior atoms each exceed .05, so no other grouping is necessary.
    assert_equal(freq, [11, 10, 10, 10, 10, 1])
    expected_edges = np.r_[
        -np.inf, np.nextafter(np.arange(-2, 3) + loc, np.inf), np.inf
    ]
    assert_equal(edges, expected_edges)
    expected = len(sample) * np.diff(stats.dlaplace.cdf(expected_edges, 1, loc))
    assert_allclose(expfreq, expected)
    assert freq.sum() == len(sample)
    assert_allclose(expfreq.sum(), len(sample))


def _cdf_scan_cuts(distfn, arg, ks, nsupp=20):
    # Brute force version of the binning rule by scanning the cdf in order.
    cdf = distfn.cdf(ks, *arg)
    wsupp = 1.0 / nsupp
    last = 0.0
    cuts = []
    for k, current in zip(ks, cdf, strict=True):
        if current - last >= wsupp - 1e-14:
            cuts.append(k)
            last = current
            if last >= 1 - wsupp:
                break
    return np.array(cuts)


class NoPpf:
    """Wrap a distribution so that ``ppf`` fails like SciPy's generic ppf."""

    def __init__(self, dist):
        self.dist = dist

    def support(self, *arg):
        return self.dist.support(*arg)

    def cdf(self, x, *arg):
        return self.dist.cdf(x, *arg)

    def ppf(self, q, *arg):
        raise RuntimeError("updating stopped, endless loop")


@pytest.mark.parametrize("mu", [1e9, 1e10])
def test_gof_binning_discrete_large_mean_edges(mu):
    # Adding a fixed epsilon to an edge is lost in floating point for values
    # of this size, and an observation equal to an edge moved to the next bin.
    sd = np.sqrt(mu)
    ks = np.arange(np.floor(mu - 3 * sd), np.ceil(mu + 3 * sd))
    scan = _cdf_scan_cuts(stats.poisson, (mu,), ks)
    assert len(scan) > 10
    _, _, edges0 = gof_binning_discrete(np.array([mu]), stats.poisson, (mu,))
    cuts = np.floor(edges0[1:-1])
    # The accuracy of the ppf of SciPy for the Poisson distribution at means
    # this large depends on the SciPy version (it was off by up to 7 for a mean
    # of 1e9), so the edges agree with the scan of the cdf up to this accuracy.
    assert_allclose(cuts, scan, rtol=1e-7)
    # exactly on the edges and next to them
    sample = np.sort(np.r_[cuts, cuts + 1, cuts[:5] - 1, mu])

    freq, expfreq, edges = gof_binning_discrete(sample, stats.poisson, (mu,))

    assert_equal(np.floor(edges[1:-1]), cuts)
    # bin i holds cuts[i - 1] < x <= cuts[i]
    expected = np.bincount(np.searchsorted(cuts, sample), minlength=len(cuts) + 1)
    assert_equal(freq, expected)
    assert freq.sum() == len(sample)
    assert_allclose(expfreq.sum(), len(sample))


def test_gof_binning_discrete_outside_support():
    # Observations outside the support are dropped, and the largest value in
    # the support is counted.
    sample = np.array([-3, 0, 1, 2, 3, 19, 20, 21, 25])
    freq, expfreq, edges = gof_binning_discrete(sample, stats.randint, (0, 20))
    assert freq.sum() == 5
    assert freq[0] == 1
    assert freq[-1] == 1
    assert_allclose(expfreq.sum(), len(sample))


@pytest.mark.parametrize("nsupp", [0, -1, -20])
def test_gof_binning_discrete_nsupp_invalid(nsupp):
    sample = np.arange(10)
    with pytest.raises(ValueError, match="nsupp"):
        gof_binning_discrete(sample, stats.randint, (0, 10), nsupp=nsupp)
    with pytest.raises(TypeError, match="nsupp"):
        gof_binning_discrete(sample, stats.randint, (0, 10), nsupp=nsupp + 0.5)


@pytest.mark.parametrize(
    "dist, arg",
    [
        (stats.poisson, (50.5, 3)),
        (stats.poisson, (3, 0)),
        (stats.nbinom, (5, 0.2)),
        (stats.dlaplace, (0.3, -7)),
        (stats.randint, (-5, 40)),
        (stats.bernoulli, (0.3,)),
        (stats.skellam, (30.5, 20.25)),
    ],
)
def test_gof_binning_discrete_cdf_fallback(dist, arg):
    # Without a working ppf the bins use the cdf and are identical.
    rng = np.random.default_rng(7654321)
    sample = dist.rvs(*arg, size=400, random_state=rng)
    expected = gof_binning_discrete(sample, dist, arg)
    result = gof_binning_discrete(sample, NoPpf(dist), arg)
    for res, exp in zip(result, expected, strict=True):
        assert_allclose(res, exp, rtol=1e-13)
    assert_equal(result[0], expected[0])
    assert_equal(result[2], expected[2])


@pytest.mark.parametrize("nsupp", [4, 20])
@pytest.mark.parametrize("loc", [-300, 0, 150])
def test_gof_binning_discrete_cdf_fallback_cdf_scan(loc, nsupp):
    # The fallback agrees with a brute force scan of the cdf.
    mu = 80.5
    ks = np.arange(0, 400) + loc
    cuts = _cdf_scan_cuts(stats.poisson, (mu, loc), ks, nsupp=nsupp)
    sample = np.arange(loc, loc + 150)
    freq, expfreq, edges = gof_binning_discrete(
        sample, NoPpf(stats.poisson), (mu, loc), nsupp=nsupp
    )
    assert_equal(np.floor(edges[1:-1]), cuts)
    expected = np.bincount(np.searchsorted(cuts, sample), minlength=len(cuts) + 1)
    assert_equal(freq, expected)


@pytest.mark.parametrize(
    "arg",
    [(449.79, 396.15), (339.91, 435.17)],
)
def test_gof_binning_discrete_skellam_generic_ppf(arg):
    # SciPy's generic ppf of skellam can raise RuntimeError for large
    # parameters, which previously made the binning fail.
    rng = np.random.default_rng(6)
    sample = stats.skellam.rvs(*arg, size=500, random_state=rng)
    freq, expfreq, edges = gof_binning_discrete(sample, stats.skellam, arg)

    assert freq.sum() == len(sample)
    assert_allclose(expfreq.sum(), len(sample))
    cuts = np.floor(edges[1:-1])
    cdf = stats.skellam.cdf(cuts, *arg)
    previous = np.r_[0.0, cdf[:-1]]
    # each bin holds at least 1/20 of the mass, and not any fewer integers
    assert np.all(cdf - previous >= 0.05 - 1e-12)
    assert np.all(stats.skellam.cdf(cuts - 1, *arg) - previous < 0.05)
    expected = len(sample) * np.diff(stats.skellam.cdf(edges, *arg))
    assert_allclose(expfreq, expected, rtol=1e-8, atol=1e-8)
    chis, pval, _, _ = gof_chisquare_discrete(
        stats.skellam, arg, sample, 0.05, "skellam"
    )
    assert_allclose([chis, pval], stats.chisquare(freq, expfreq))


def test_chisquare_power():
    from .results.results_power import pwr_chisquare
    for case in pwr_chisquare.values():
        power = chisquare_power(case.w, case.N, case.df + 1,
                                alpha=case.sig_level)
        assert_almost_equal(power, case.power, decimal=6,
                            err_msg=repr(vars(case)))


def test_chisquare():
    # TODO: no tests for ``value`` yet
    res1 = Holder()
    res2 = Holder()
    # > freq = c(1048,  660,  510,  420,  362)
    # > pr1 = c(1020,  690,  510,  420,  360)
    # > pr2 = c(1050,  660,  510,  420,  360)
    # > c = chisq.test(freq, p=pr1, rescale.p = TRUE)
    # > cat_items(c, "res1.")
    res1.statistic = 2.084086388178453
    res1.parameter = 4
    res1.p_value = 0.72029651761105
    res1.method = "Chi-squared test for given probabilities"
    res1.data_name = "freq"
    res1.observed = np.array([
         1048, 660, 510, 420, 362
        ])
    res1.expected = np.array([
         1020, 690, 510, 420, 360
        ])
    res1.residuals = np.array([
         0.876714007519206, -1.142080481440321, -2.517068894406109e-15,
         -2.773674830645328e-15, 0.105409255338946
        ])

    # > c = chisq.test(freq, p=pr2, rescale.p = TRUE)
    # > cat_items(c, "res2.")
    res2.statistic = 0.01492063492063492
    res2.parameter = 4
    res2.p_value = 0.999972309849908
    res2.method = "Chi-squared test for given probabilities"
    res2.data_name = "freq"
    res2.observed = np.array([
         1048, 660, 510, 420, 362
        ])
    res2.expected = np.array([
         1050, 660, 510, 420, 360
        ])
    res2.residuals = np.array([
         -0.06172133998483677, 0, -2.517068894406109e-15,
         -2.773674830645328e-15, 0.105409255338946
        ])

    freq = np.array([1048,  660,  510,  420,  362])
    pr1 = np.array([1020,  690,  510,  420,  360])
    pr2 = np.array([1050,  660,  510,  420,  360])

    for pr, res in zip([pr1, pr2], [res1, res2], strict=True):
        stat, pval = chisquare(freq, pr)
        assert_almost_equal(stat, res.statistic, decimal=12)
        assert_almost_equal(pval, res.p_value, decimal=13)


def test_chisquare_effectsize():

    pr1 = np.array([1020,  690,  510,  420,  360])
    pr2 = np.array([1050,  660,  510,  420,  360])
    # > library(pwr)
    # > ES.w1(pr1/3000, pr2/3000)
    es_r = 0.02699815282115563
    es1 = chisquare_effectsize(pr1, pr2)
    es2 = chisquare_effectsize(pr1, pr2, cohen=False)
    assert_almost_equal(es1, es_r, decimal=14)
    assert_almost_equal(es2, es_r**2, decimal=14)

    # regression tests for correction
    res1 = chisquare_effectsize(pr1, pr2, cohen=False,
                                correction=(3000, len(pr1)-1))
    res0 = 0  # -0.00059994422693327625
    assert_equal(res1, res0)
    pr3 = pr2 + [0, 0, 0, 50, 50]
    res1 = chisquare_effectsize(pr1, pr3, cohen=False,
                                correction=(3000, len(pr1)-1))
    res0 = 0.0023106468846296755
    assert_almost_equal(res1, res0, decimal=14)
    # compare
    # res_nc = chisquare_effectsize(pr1, pr3, cohen=False)
    # 0.0036681143072077533


def test_chisquare_effectsize_negative_probs_raises():
    # negative probability entries previously returned nan silently
    with pytest.raises(ValueError, match="probs0 and probs1 must be non-negative"):
        chisquare_effectsize(np.array([-0.1, 0.6]), np.array([0.5, 0.5]))
    es = chisquare_effectsize(np.array([0.3, 0.7]), np.array([0.5, 0.5]))
    assert np.isfinite(es)
    assert es >= 0


def test_chisquare_power_invalid_inputs_raises():
    # impossible inputs previously returned nan power silently
    with pytest.raises(ValueError, match="nobs must be positive"):
        chisquare_power(0.3, -5, 5, alpha=0.1)
    with pytest.raises(ValueError, match="n_bins must be at least 1"):
        chisquare_power(0.3, 50, 0, alpha=0.1)
    with pytest.raises(ValueError, match="alpha must be in the range"):
        chisquare_power(0.3, 50, 5, alpha=2)
    # NaN is not in any of the valid ranges
    with pytest.raises(ValueError, match="alpha must be in the range"):
        chisquare_power(0.3, 50, 5, alpha=np.nan)
    with pytest.raises(ValueError, match="alpha must be in the range"):
        chisquare_power(0.3, 50, 5, alpha=[0.05, np.nan])
    with pytest.raises(ValueError, match="nobs must be positive"):
        chisquare_power(0.3, np.nan, 5, alpha=0.1)
    with pytest.raises(ValueError, match="n_bins must be at least 1"):
        chisquare_power(0.3, 50, np.nan, alpha=0.1)
    p = chisquare_power(0.3, 50, 5, alpha=0.1)
    assert np.isfinite(p)
    assert 0 < p < 1
