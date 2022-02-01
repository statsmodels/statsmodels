# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 23:32:57 2021

Author: Josef Perktold
License: BSD-3

"""
from statsmodels.compat.pytest import pytest_warns

from statsmodels.compat.scipy import SP_LT_15

import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import pytest
from scipy import stats

from statsmodels.distributions.copula.archimedean import (
    ArchimedeanCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
)
from statsmodels.distributions.copula.copulas import CopulaDistribution
import statsmodels.distributions.copula.depfunc_ev as trev
from statsmodels.distributions.copula.elliptical import (
    GaussianCopula,
    StudentTCopula,
)
from statsmodels.distributions.copula.extreme_value import (
    ExtremeValueCopula,
    copula_bv_ev,
)
from statsmodels.distributions.copula.other_copulas import IndependenceCopula
import statsmodels.distributions.copula.transforms as tra
from statsmodels.distributions.tools import (
    approx_copula_pdf,
    frequencies_fromdata,
)
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess

uniform = stats.uniform


ev_list = [
    [trev.transform_bilogistic, 0.5, 0.9, (0.25, 0.05), 0.5],
    [trev.transform_tawn, 0.5, 0.9, (0.5, 0.5, 0.5), 0.4724570876035117],
    # note evd has asymmetry reversed, interchange variables
    [trev.transform_tawn2, 0.5, 0.9, (0.25, 0.05), 0.464357480263932],
    [trev.transform_tawn2, 0.5, 0.9, (0.5, 0.25), 0.4916117128670654],
    [trev.transform_tawn2, 0.9, 0.5, (0.5, 0.25), 0.48340673415789],
    # note evd has parameter for hr 1/lmbda (inverse of our parameter)
    [trev.transform_hr, 0.5, 0.9, (2,), 0.4551235014298542],
    [trev.transform_joe, 0.5, 0.9, (0.5, 0.75, 1 / 0.25), 0.4543698299835434],
    [trev.transform_joe, 0.9, 0.5, (0.5, 0.75, 1 / 0.25), 0.4539773435983587],
    # tev is against R `copula` package
    # > cop = tevCopula(0.8, df = 4)
    # > pCopula(c(0.5, 0.75), cop)
    # [1] 0.456807960674953
    # > pCopula(c(0.5, 0.9), cop)
    # [1] 0.4911039761533587
    [trev.transform_tev, 0.5, 0.75, (0.8, 4), 0.456807960674953],
    [trev.transform_tev, 0.5, 0.9, (0.8, 4), 0.4911039761533587],
]

ev_dep_list = [
    # [trev.transform_bilogistic, 0.5, 0.9, (0.25, 0.05), 0.5],
    [trev.transform_tawn, 0.5, 0.9, (0.5, 0.5, 0.5), 0.4724570876035117,
     [0.8952847075210475, 0.8535533905932737, 0.8952847075210475]],
    # abvevd(c(0.25, 0.5, 0.75), dep=0.25, asy = c(0.5, 0.75), model = "alog")
    [trev.transform_tawn, 0.5, 0.9, (0.5, 0.75, 0.25), 0.4724570876035117,
     [0.8753426223607659, 0.7672861240893745, 0.8182268471629245]],

    [trev.transform_tawn2, 0.4, 0.9, (0.3, 0.2), 0,
     [0.8968750000000001, 0.8500000000000000, 0.8781249999999999]],
    # # note evd has asymmetry reversed, interchange variables - NOT anymore
    # [trev.transform_tawn2, 0.9, 0.5, (0.25, 0.05), 0.464357480263932],
    # [trev.transform_tawn2, 0.9, 0.5, (0.5, 0.25), 0.4916117128670654],
    # [trev.transform_tawn2, 0.5, 0.9, (0.5, 0.25), 0.48340673415789],
    # # note evd has parameter for hr 1/lmbda (inverse of our parameter)
    [trev.transform_hr, 0.5, 0.9, (1/2,), 0.4551235014298542,
     [0.7774638908611127, 0.6914624612740130, 0.7774638908611127]],
    # [trev.transform_joe, 0.5, 0.9, (0.5, 0.75, 1/0.25), 0.4543698299835434],
    # [trev.transform_joe, 0.9, 0.5, (0.5, 0.75, 1/0.25), 0.4539773435983587],
    # > abvevd(c(0.25, 0.5, 0.75), dep=0.75, asy=c(0.5, 0.75), model="aneglog")
    # [1] 0.9139915932031195 0.8803412942173715 0.8993537417026507
    [trev.transform_joe, 0.5, 0.9, (0.5, 0.75, 1/0.75), 0.,
     [0.9139915932031195, 0.8803412942173715, 0.8993537417026507]]
    ]


cop_list = [
    [tra.TransfFrank, 0.5, 0.9, (2,), 0.4710805107852225, 0.9257812360337806],
    [tra.TransfGumbel, 0.5, 0.9, (2,), 0.4960348880595387, 0.3973548776136501],
    [tra.TransfClayton, 0.5, 0.9, (2,), 0.485954322440435, 0.8921974147432954],
    [tra.TransfIndep, 0.5, 0.5, (), 0.25, 1],
]

gev_list = [
    # [cop.transform_tawn, 0.5, 0.9, (0.5, 0.5, 0.5), 0.4724570876035117],
    # > pbvevd(c(0.5,0.9), dep = 0.25, asy = c(0.5, 0.5), model = "alog")
    # [trev.transform_tawn, 0.5, 0.9, (0.5, 0.5, 0.25),
    #  0.4386367545837274, 0.12227570158361],
    [trev.transform_tawn, 0.5, 0.9, (0.5, 0.75, 0.25),
     0.4868879662205026, 0.4646154226541540, 0.1897142141905908],
    [trev.transform_tawn2, 0.4, 0.9, (0.3, 0.2),
     0.3838690483829361, 0.3989785485000293, 0.1084278364284748],
    # [trev.transform_tawn2, 0.5, 0.5, (0.5, 0.25), 0.387629940606913,
    # 0.1383277275273335],
    # [trev.transform_tawn2, 0.9, 0.5, (0.5, 0.25), 0.4519820720233402,
    # 0.1162545305128522],  # fails in pdf
    # note evd has parameter for hr 1/lmbda (inverse of our parameter)
    [trev.transform_hr, 0.4, 0.9, (2,),
     0.36459381872178737, 0.34879372499897571, 0.09305880295825367],
    # [trev.transform_joe, 0.5, 0.9, (0.5, 0.75, 1/0.25), 0.3700584213780548,
    # 0.08992436735088952],
    [trev.transform_joe, 0.4, 0.9, (0.5, 0.75, 1/0.25),
     0.36391125216656162, 0.34752631779552950, 0.09316705199822513],
    ]


def check_cop_rvs(cop, rvs=None, nobs=2000, k=10, use_pdf=True):
    if rvs is None:
        rvs = cop.rvs(nobs)
    freq = frequencies_fromdata(rvs, k, use_ranks=True)
    if use_pdf:
        pdfg = approx_copula_pdf(cop, k_bins=k, force_uniform=True)
        count_pdf = pdfg * nobs
    else:
        # use copula cdf if available
        raise NotImplementedError
    mask = count_pdf < 2
    if mask.sum() > 5:
        cp = count_pdf[mask]
        cp = np.concatenate([cp, [nobs - cp.sum()]])
        fr = freq[mask]
        cp = np.concatenate([fr, [nobs - fr.sum()]])
    else:
        fr = freq.ravel()
        cp = count_pdf.ravel()

    chi2_test = stats.chisquare(freq.ravel(), count_pdf.ravel())
    return chi2_test, rvs


extrali = [
    [trev.transform_tawn, 0.5, 0.9, (0.8, 0.5, 0.75), 0.4724570876035117],
    [trev.transform_tawn, 0.5, 0.9, (0.5, 0.75, 0.5), 0.4724570876035117],
    [trev.transform_tawn, 0.6, 0.4, (0.2, 0.7, 0.6), 0.4724570876035117],
]


@pytest.mark.parametrize("case", ev_list + extrali)
def test_ev_copula(case):
    # check ev copulas, cdf and transform against R `evd` package
    ev_tr, v1, v2, args, res1 = case
    res = copula_bv_ev([v1, v2], ev_tr, args=args)
    # assert_allclose(res, res1, rtol=1e-13)

    # check derivatives of dependence function
    if ev_tr in (trev.transform_bilogistic, trev.transform_tev):
        return
    d1_res = approx_fprime_cs(np.array([v1, v2]), ev_tr.evaluate, args=args)
    d1_res = np.diag(d1_res)
    d1 = ev_tr.deriv(np.array([v1, v2]), *args)
    assert_allclose(d1, d1_res, rtol=1e-8)

    d1_res = approx_hess(np.array([0.5]), ev_tr.evaluate, args=args)
    d1_res = np.diag(d1_res)
    d1 = ev_tr.deriv2(0.5, *args)
    assert_allclose(d1, d1_res, rtol=1e-7)


@pytest.mark.parametrize("case", ev_dep_list)
def test_ev_dep(case):
    ev_tr, v1, v2, args, res1, res2 = case  # noqa
    t = np.array([0.25, 0.5, 0.75])
    df = ev_tr(t, *args)
    assert_allclose(df, res2, rtol=1e-13)


@pytest.mark.parametrize("case", cop_list)
def test_copulas(case):
    # check ev copulas, cdf and transform against R `copula` package
    cop_tr, v1, v2, args, cdf2, pdf2 = case
    ca = ArchimedeanCopula(cop_tr())
    cdf1 = ca.cdf([v1, v2], args=args)
    pdf1 = ca.pdf([v1, v2], args=args)
    assert_allclose(cdf1, cdf2, rtol=1e-13)
    assert_allclose(pdf1, pdf2, rtol=1e-13)

    logpdf1 = ca.logpdf([v1, v2], args=args)
    assert_allclose(logpdf1, np.log(pdf2), rtol=1e-13)


@pytest.mark.parametrize("case", ev_list)
def test_ev_copula_distr(case):
    # check ev copulas, cdf and transform against R `evd` package
    ev_tr, v1, v2, args, res1 = case
    u = [v1, v2]
    res = copula_bv_ev(u, ev_tr, args=args)
    assert_allclose(res, res1, rtol=1e-13)

    ev = ExtremeValueCopula(ev_tr)
    cdf1 = ev.cdf(u, args)
    assert_allclose(cdf1, res1, rtol=1e-13)

    cev = CopulaDistribution(ev, [uniform, uniform], cop_args=args)
    cdfd = cev.cdf(np.array(u), cop_args=args)
    assert_allclose(cdfd, res1, rtol=1e-13)
    assert cdfd.shape == ()

    # using list u
    cdfd = cev.cdf(u, cop_args=args)
    assert_allclose(cdfd, res1, rtol=1e-13)
    assert cdfd.shape == ()

    # check vector values for u
    # bilogistic is not vectorized, uses integrate.quad
    if ev_tr != trev.transform_bilogistic:
        cdfd = cev.cdf(np.array(u) * np.ones((3, 1)), cop_args=args)
        assert_allclose(cdfd, res1, rtol=1e-13)
        assert cdfd.shape == (3,)


@pytest.mark.parametrize("case", cop_list)
def test_copulas_distr(case):
    # check ev copulas, cdf and transform against R `copula` package
    cop_tr, v1, v2, args, cdf2, pdf2 = case
    u = [v1, v2]
    ca = ArchimedeanCopula(cop_tr())
    cdf1 = ca.cdf(u, args=args)
    pdf1 = ca.pdf(u, args=args)

    cad = CopulaDistribution(ca, [uniform, uniform], cop_args=args)
    cdfd = cad.cdf(np.array(u), cop_args=args)
    assert_allclose(cdfd, cdf1, rtol=1e-13)
    assert cdfd.shape == ()

    # check pdf
    pdfd = cad.pdf(np.array(u), cop_args=args)
    assert_allclose(pdfd, pdf1, rtol=1e-13)
    assert cdfd.shape == ()

    # using list u
    cdfd = cad.cdf(u, cop_args=args)
    assert_allclose(cdfd, cdf1, rtol=1e-13)
    assert cdfd.shape == ()

    assert_allclose(cdf1, cdf2, rtol=1e-13)
    assert_allclose(pdf1, pdf2, rtol=1e-13)

    # check vector values for u
    cdfd = cad.cdf(np.array(u) * np.ones((3, 1)), cop_args=args)
    assert_allclose(cdfd, cdf2, rtol=1e-13)
    assert cdfd.shape == (3,)

    # check mv, check at marginal cdf
    cdfmv = ca.cdf([v1, v2, 1], args=args)
    assert_allclose(cdfmv, cdf1, rtol=1e-13)
    assert cdfd.shape == (3,)


@pytest.mark.parametrize("case", gev_list)
def test_gev_genextreme(case):
    gev = stats.genextreme(0)
    # check ev copulas, cdf and transform against R `evt` package
    ev_tr, v1, v2, args, res0, res1, res2 = case
    y = [v1, v2]
    u = gev.cdf(y)
    res = copula_bv_ev(u, ev_tr, args=args)
    assert_allclose(res, res1, rtol=1e-13)

    ev = ExtremeValueCopula(ev_tr)
    # evaluated at using u = y
    cdf1 = ev.cdf(y, args)
    assert_allclose(cdf1, res0, rtol=1e-13)

    # evaluated at transformed u = F(y)
    cdf1 = ev.cdf(u, args)
    assert_allclose(cdf1, res1, rtol=1e-13)

    cev = CopulaDistribution(ev, [gev, gev], cop_args=args)
    cdfd = cev.cdf(np.array(y), cop_args=args)
    assert_allclose(cdfd, res1, rtol=1e-13)
    pdfd = cev.pdf(np.array(y), cop_args=args)
    assert_allclose(pdfd, res2, rtol=1e-13)


class TestFrank:
    def test_basic(self):
        case = [tra.TransfFrank, 0.5, 0.9, (2,), 0.4710805107852225,
                0.9257812360337806]
        cop_tr, v1, v2, args, cdf2, pdf2 = case
        cop = FrankCopula()

        pdf1 = cop.pdf([v1, v2], args=args)
        assert_allclose(pdf1, pdf2, rtol=1e-13)
        logpdf1 = cop.logpdf([v1, v2], args=args)
        assert_allclose(logpdf1, np.log(pdf2), rtol=1e-13)

        cdf1 = cop.cdf([v1, v2], args=args)
        assert_allclose(cdf1, cdf2, rtol=1e-13)

        assert isinstance(cop.transform, cop_tr)

        # round trip conditional, no verification
        u = [0.6, 0.5]
        cdfc = cop.cdfcond_2g1(u, args=args)
        ppfc = cop.ppfcond_2g1(cdfc, [0.6], args=args)
        assert_allclose(ppfc, u[1], rtol=1e-13)


# The reference results are coming from the R package Copula.
# See ``copula_r_tests.rst`` for more details.


class CheckCopula:
    """Generic tests for copula."""

    copula = None
    dim = None
    u = np.array([[0.33706249, 0.6075078],
                  [0.62232507, 0.06241089],
                  [0.2001457, 0.54027684],
                  [0.77166391, 0.40610225],
                  [0.98534253, 0.99212789],
                  [0.72755898, 0.25913165],
                  [0.05943888, 0.61044613],
                  [0.0962475, 0.67585563],
                  [0.35496733, 0.79584436],
                  [0.44513594, 0.23050014]])
    pdf_u = None
    cdf_u = None

    def _est_visualization(self):
        sample = self.copula.rvs(10000)
        assert sample.shape == (10000, 2)
        # h = sns.jointplot(sample[:, 0], sample[:, 1], kind='hex')
        # h.set_axis_labels('X1', 'X2', fontsize=16)

    def test_pdf(self):
        pdf_u_test = self.copula.pdf(self.u)
        assert_array_almost_equal(self.pdf_u, pdf_u_test)

    def test_cdf(self):
        cdf_u_test = self.copula.cdf(self.u)
        assert_array_almost_equal(self.cdf_u, cdf_u_test)

    def test_validate_params(self):
        pass

    def test_rvs(self):
        nobs = 2000
        rng = np.random.RandomState(27658622)
        self.rvs = rvs = self.copula.rvs(nobs, random_state=rng)
        assert rvs.shape == (nobs, 2)
        assert_array_almost_equal(
            np.mean(rvs, axis=0), np.repeat(0.5, self.dim), decimal=2
        )

        # check empirical quantiles, uniform
        q0 = np.percentile(rvs, [25, 50, 75], axis=0)
        q1 = np.repeat(np.array([[0.25, 0.5, 0.75]]).T, 2, axis=1)
        assert_allclose(q0, q1, atol=0.025)

        tau = stats.kendalltau(*rvs.T)[0]
        tau_cop = self.copula.tau()
        assert_allclose(tau, tau_cop, rtol=0.08, atol=0.005)

        if isinstance(self.copula, IndependenceCopula):
            # skip rest, no `_arg_from_tau` in IndependenceCopula
            return
        theta = self.copula.fit_corr_param(rvs)
        theta_cop = getattr(self.copula, "theta", None)
        if theta_cop is None:
            # elliptical
            theta_cop = self.copula.corr[0, 1]
        assert_allclose(theta, theta_cop, rtol=0.1, atol=0.005)


class CheckModernCopula(CheckCopula):
    @pytest.mark.parametrize(
        "seed", ["random_state", "generator", "qmc", None, 0]
    )
    def test_seed(self, seed):
        if SP_LT_15 and seed in ("generator", 0):
            pytest.xfail(reason="Generator not supported for SciPy <= 1.3")
        if seed == "random_state":
            seed1 = np.random.RandomState(0)
            seed2 = np.random.RandomState(0)
        elif seed == "generator":
            seed1 = np.random.default_rng(0)
            seed2 = 0
        elif seed is None:
            seed1 = None
            singleton = np.random.mtrand._rand
            seed2 = np.random.RandomState()
            seed2.set_state(singleton.get_state())
        elif seed == "qmc":
            if not hasattr(stats, "qmc"):
                pytest.skip("QMC not available")
            else:
                pytest.xfail("QMC not working")
            seed1 = stats.qmc.Halton(2)
            seed2 = stats.qmc.Halton(2)
        else:
            seed1 = 0
            seed2 = np.random.default_rng(0)

        nobs = 2000
        expected_warn = None if seed1 is not None else FutureWarning
        with pytest_warns(expected_warn):
            rvs1 = self.copula.rvs(nobs, random_state=seed1)
        rvs2 = self.copula.rvs(nobs, random_state=seed2)
        assert_allclose(rvs1, rvs2)


class TestIndependenceCopula(CheckCopula):
    copula = IndependenceCopula()
    dim = 2
    pdf_u = np.ones(10)
    cdf_u = np.prod(CheckCopula.u, axis=1)


class TestGaussianCopula(CheckCopula):
    copula = GaussianCopula(corr=[[1.0, 0.8], [0.8, 1.0]])
    dim = 2
    pdf_u = [1.03308741, 0.06507279, 0.72896012, 0.65389439, 16.45012399,
             0.34813218, 0.06768115, 0.08168840, 0.40521741, 1.26723470]
    cdf_u = [0.31906854, 0.06230196, 0.19284669, 0.39952707, 0.98144792,
             0.25677003, 0.05932818, 0.09605404, 0.35211017, 0.20885480]

    def test_rvs(self):
        # copied from student t test,
        # currently inconsistent with non-elliptical copulas
        super().test_rvs()

        chi2t, rvs = check_cop_rvs(
            self.copula, rvs=self.rvs, nobs=2000, k=10, use_pdf=True
        )
        assert chi2t.pvalue > 0.1
        tau = stats.kendalltau(*rvs.T)[0]
        tau_cop = self.copula.tau()
        assert_allclose(tau, tau_cop, rtol=0.05)


class TestStudentTCopula(CheckCopula):
    copula = StudentTCopula(corr=[[1.0, 0.8], [0.8, 1.0]], df=2)
    dim = 2
    pdf_u = [0.8303065, 0.1359839, 0.5157746, 0.4776421, 26.2173959,
             0.3070661, 0.1349173, 0.1597064, 0.3303230, 1.0482301]
    cdf_u = [0.31140349, 0.05942746, 0.18548601, 0.39143974, 0.98347259,
             0.24894028, 0.05653947, 0.09210693, 0.34447385, 0.20429882]

    def test_cdf(self):
        pytest.skip("Not implemented.")

    def test_rvs(self):
        super().test_rvs()

        chi2t, rvs = check_cop_rvs(
            self.copula, rvs=self.rvs, nobs=2000, k=10, use_pdf=True
        )
        assert chi2t.pvalue > 0.1
        tau = stats.kendalltau(*rvs.T)[0]
        tau_cop = self.copula.tau()
        assert_allclose(tau, tau_cop, rtol=0.05)


class TestClaytonCopula(CheckModernCopula):
    copula = ClaytonCopula(theta=1.2)
    dim = 2
    pdf_u = [1.0119836, 0.2072728, 0.8148839, 0.9481976, 2.1419659,
             0.6828507, 0.2040454, 0.2838497, 0.8197787, 1.1096360]
    cdf_u = [0.28520375, 0.06101690, 0.17703377, 0.36848218, 0.97772088,
             0.24082057, 0.05811908, 0.09343934, 0.33012582, 0.18738753]


class TestFrankCopula(CheckModernCopula):
    copula = FrankCopula(theta=3)
    dim = 2
    pdf_u = [0.9646599, 0.5627195, 0.8941964, 0.8364614, 2.9570945,
             0.6665601, 0.5779906, 0.5241333, 0.7156741, 1.1074024]
    cdf_u = [0.27467496, 0.05492539, 0.15995939, 0.36750702, 0.97782283,
             0.23412757, 0.05196265, 0.08676979, 0.32803721, 0.16320730]


class TestGumbelCopula(CheckModernCopula):
    copula = GumbelCopula(theta=1.5)
    dim = 2
    pdf_u = [1.0391696, 0.6539579, 0.9878446, 0.8679504, 16.6030932,
             0.7542073, 0.6668307, 0.6275887, 0.7477991, 1.1564864]
    cdf_u = [0.27194634, 0.05484380, 0.15668190, 0.37098420, 0.98176346,
             0.23422865, 0.05188260, 0.08659615, 0.33086960, 0.15803914]
