"""
Created on Thu Jan 14 23:32:57 2021

Author: Josef Perktold
License: BSD-3

"""

from statsmodels.compat.pytest import pytest_warns

from unittest import mock
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import pytest
from scipy import stats

from statsmodels.distributions.copula.archimedean import (
    ArchimedeanCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    _debyem1_expansion,
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
from statsmodels.distributions.copula.other_copulas import (
    IndependenceCopula,
    rvs_kernel,
)
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
    [
        trev.transform_tawn,
        0.5,
        0.9,
        (0.5, 0.5, 0.5),
        0.4724570876035117,
        [0.8952847075210475, 0.8535533905932737, 0.8952847075210475],
    ],
    # abvevd(c(0.25, 0.5, 0.75), dep=0.25, asy = c(0.5, 0.75), model = "alog")
    [
        trev.transform_tawn,
        0.5,
        0.9,
        (0.5, 0.75, 0.25),
        0.4724570876035117,
        [0.8753426223607659, 0.7672861240893745, 0.8182268471629245],
    ],
    [
        trev.transform_tawn2,
        0.4,
        0.9,
        (0.3, 0.2),
        0,
        [0.8968750000000001, 0.8500000000000000, 0.8781249999999999],
    ],
    # # note evd has asymmetry reversed, interchange variables - NOT anymore
    # [trev.transform_tawn2, 0.9, 0.5, (0.25, 0.05), 0.464357480263932],
    # [trev.transform_tawn2, 0.9, 0.5, (0.5, 0.25), 0.4916117128670654],
    # [trev.transform_tawn2, 0.5, 0.9, (0.5, 0.25), 0.48340673415789],
    # # note evd has parameter for hr 1/lmbda (inverse of our parameter)
    [
        trev.transform_hr,
        0.5,
        0.9,
        (1 / 2,),
        0.4551235014298542,
        [0.7774638908611127, 0.6914624612740130, 0.7774638908611127],
    ],
    # [trev.transform_joe, 0.5, 0.9, (0.5, 0.75, 1/0.25), 0.4543698299835434],
    # [trev.transform_joe, 0.9, 0.5, (0.5, 0.75, 1/0.25), 0.4539773435983587],
    # > abvevd(c(0.25, 0.5, 0.75), dep=0.75, asy=c(0.5, 0.75), model="aneglog")
    # [1] 0.9139915932031195 0.8803412942173715 0.8993537417026507
    [
        trev.transform_joe,
        0.5,
        0.9,
        (0.5, 0.75, 1 / 0.75),
        0.0,
        [0.9139915932031195, 0.8803412942173715, 0.8993537417026507],
    ],
]


cop_list = [
    [
        tra.TransfFrank,
        [0.5, 0.9],
        (2,),
        0.4710805107852225,
        0.9257812360337806,
        FrankCopula,
    ],
    [
        tra.TransfGumbel,
        [0.5, 0.9],
        (2,),
        0.4960348880595387,
        0.3973548776136501,
        GumbelCopula,
    ],
    [
        tra.TransfClayton,
        [0.5, 0.9],
        (2,),
        0.485954322440435,
        0.8921974147432954,
        ClaytonCopula,
    ],
    [tra.TransfIndep, [0.5, 0.5], (), 0.25, 1, IndependenceCopula],
]


# separate mv list because test_copulas_distr not yet adjusted
copk_list = [
    # k_dim = 3
    [
        tra.TransfGumbel,
        [0.6, 0.5, 0.9],
        (2,),
        0.4200146617837097,
        0.7507987484870147,
        GumbelCopula,
    ],
    [
        tra.TransfClayton,
        [0.6, 0.5, 0.9],
        (2,),
        0.4078289289864994,
        1.430033358494079,
        ClaytonCopula,
    ],
    [
        tra.TransfFrank,
        [0.6, 0.5, 0.9],
        (2,),
        0.3397845258821868,
        1.123811705698149,
        FrankCopula,
    ],
    # k_dim = 4
    [
        tra.TransfGumbel,
        [0.6, 0.5, 0.9, 0.1],
        (2,),
        0.08538643946528957,
        0.05130542596740889,
        GumbelCopula,
    ],
    [
        tra.TransfClayton,
        [0.6, 0.5, 0.9, 0.1],
        (2,),
        0.09758427058689817,
        0.00428071573295176,
        ClaytonCopula,
    ],
    [
        tra.TransfFrank,
        [0.6, 0.5, 0.9, 0.1],
        (2,),
        0.05456579067435671,
        0.4089534511841545,
        FrankCopula,
    ],
    [tra.TransfIndep, [0.5, 0.5, 0.5, 0.5], (), 0.0625, 1, IndependenceCopula],
]

# archimedean with pdf only for k_dim <= 4
cop_2d = [
    [tra.TransfFrank, (2,), FrankCopula],
    [tra.TransfGumbel, (2,), GumbelCopula],
    # [tra.TransfClayton, (2,), ClaytonCopula],
    # [tra.TransfIndep, (), IndependenceCopula],
]

gev_list = [
    # [cop.transform_tawn, 0.5, 0.9, (0.5, 0.5, 0.5), 0.4724570876035117],
    # > pbvevd(c(0.5,0.9), dep = 0.25, asy = c(0.5, 0.5), model = "alog")
    # [trev.transform_tawn, 0.5, 0.9, (0.5, 0.5, 0.25),
    #  0.4386367545837274, 0.12227570158361],
    [
        trev.transform_tawn,
        0.5,
        0.9,
        (0.5, 0.75, 0.25),
        0.4868879662205026,
        0.4646154226541540,
        0.1897142141905908,
    ],
    [
        trev.transform_tawn2,
        0.4,
        0.9,
        (0.3, 0.2),
        0.3838690483829361,
        0.3989785485000293,
        0.1084278364284748,
    ],
    # [trev.transform_tawn2, 0.5, 0.5, (0.5, 0.25), 0.387629940606913,
    # 0.1383277275273335],
    # [trev.transform_tawn2, 0.9, 0.5, (0.5, 0.25), 0.4519820720233402,
    # 0.1162545305128522],  # fails in pdf
    # note evd has parameter for hr 1/lmbda (inverse of our parameter)
    [
        trev.transform_hr,
        0.4,
        0.9,
        (2,),
        0.36459381872178737,
        0.34879372499897571,
        0.09305880295825367,
    ],
    # [trev.transform_joe, 0.5, 0.9, (0.5, 0.75, 1/0.25), 0.3700584213780548,
    # 0.08992436735088952],
    [
        trev.transform_joe,
        0.4,
        0.9,
        (0.5, 0.75, 1 / 0.25),
        0.36391125216656162,
        0.34752631779552950,
        0.09316705199822513,
    ],
]


def check_cop_rvs(cop, rvs=None, nobs=2000, k=10, use_pdf=True, rng=None):
    if rvs is None:
        rvs = cop.rvs(nobs, rng=rng)
    else:
        nobs = rvs.shape[0]
    freq = frequencies_fromdata(rvs, k, use_ranks=True)
    pdfg = approx_copula_pdf(
        cop, k_bins=k, force_uniform=True, use_pdf=use_pdf, rng=rng
    )
    count_pdf = pdfg * nobs

    freq = freq.ravel()
    count_pdf = count_pdf.ravel()
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
    # Smoke test
    copula_bv_ev([v1, v2], ev_tr, args=args)
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
    ev_tr, v1, v2, args, res1, res2 = case
    t = np.array([0.25, 0.5, 0.75])
    df = ev_tr(t, *args)
    assert_allclose(df, res2, rtol=1e-13)


@pytest.mark.parametrize("case", cop_list + copk_list)
def test_copulas(case):
    # check ev copulas, cdf and transform against R `copula` package
    cop_tr, u, args, cdf2, pdf2, cop = case
    k_dim = np.asarray(u).shape[-1]
    ca = ArchimedeanCopula(cop_tr(), k_dim=k_dim)
    cdf1 = ca.cdf(u, args=args)
    pdf1 = ca.pdf(u, args=args)
    assert_allclose(cdf1, cdf2, rtol=1e-13)
    assert_allclose(pdf1, pdf2, rtol=1e-13)
    assert cdf1.shape == ()

    logpdf1 = ca.logpdf(u, args=args)
    assert_allclose(logpdf1, np.log(pdf2), rtol=1e-13)

    # compare with specific copula class
    ca2 = cop(k_dim=k_dim)
    cdf3 = ca2.cdf(u, args=args)
    pdf3 = ca2.pdf(u, args=args)
    logpdf3 = ca2.logpdf(u, args=args)
    assert_allclose(cdf3, cdf2, rtol=1e-13)
    assert_allclose(pdf3, pdf2, rtol=1e-13)
    assert_allclose(logpdf3, np.log(pdf2), rtol=1e-13)
    assert cdf3.shape == ()
    assert pdf3.shape == ()  # currently fails


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


@pytest.mark.parametrize("case", cop_list + copk_list)
def test_copulas_distr(case):
    # check ev copulas, cdf and transform against R `copula` package
    cop_tr, u, args, cdf2, pdf2, cop = case
    k_dim = np.asarray(u).shape[-1]

    ca = ArchimedeanCopula(cop_tr(), k_dim=k_dim)
    cdf1 = ca.cdf(u, args=args)
    pdf1 = ca.pdf(u, args=args)

    marginals = [uniform] * k_dim
    cad = CopulaDistribution(ca, marginals, cop_args=args)
    # TODO: check also for specific archimedean classes
    # cad = CopulaDistribution(cop(k_dim=k_dim), marginals, cop_args=args)
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

        cdfmv = ca.cdf(list(u) + [1], args=args)
    assert_allclose(cdfmv, cdf1, rtol=1e-13)
    assert cdfd.shape == (3,)


@pytest.mark.parametrize("case", cop_2d)
@pytest.mark.parametrize("k_dim", [5, 6])
def test_copulas_raise(case, k_dim):
    cop_tr, args, cop = case
    u = [0.5] * k_dim

    ca = ArchimedeanCopula(cop_tr(), k_dim=k_dim)

    with pytest.raises(NotImplementedError):
        ca.rvs(u, args=args)

    with pytest.raises(NotImplementedError):
        ca.pdf(u, args=args)


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
        case = [tra.TransfFrank, 0.5, 0.9, (2,), 0.4710805107852225, 0.9257812360337806]
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

    def test_tau(self):
        copula = FrankCopula(k_dim=2)

        theta = [2, 1, 1e-2, 1e-4, 1e-5, 1e-6]
        # > tau(frankCopula(param = 2, dim = 2))
        tau_r = [
            0.2138945692196201,
            0.110018536448993,
            0.001111110000028503,
            1.111110992013664e-05,
            1.111104651951855e-06,
            1.108825244955369e-07,
        ]

        tau_cop = [copula.tau(th) for th in theta]
        assert_allclose(tau_cop[:-1], tau_r[:-1], rtol=1e-5)
        # relative precision at very small tau is not very high

        # check debye function
        taud = 1 + 4 * _debyem1_expansion(theta) / theta
        assert_allclose(taud, tau_cop, rtol=1e-5)


# The reference results are coming from the R package Copula.
# See ``copula_r_tests.rst`` for more details.


class CheckCopula:
    """Generic tests for copula."""

    copula = None
    dim = None
    u = np.array(
        [
            [0.33706249, 0.6075078],
            [0.62232507, 0.06241089],
            [0.2001457, 0.54027684],
            [0.77166391, 0.40610225],
            [0.98534253, 0.99212789],
            [0.72755898, 0.25913165],
            [0.05943888, 0.61044613],
            [0.0962475, 0.67585563],
            [0.35496733, 0.79584436],
            [0.44513594, 0.23050014],
        ]
    )
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
        assert_allclose(self.cdf_u, cdf_u_test, rtol=6e-5)

    def test_validate_params(self):
        pass

    def test_rvs(self):
        nobs = 2000
        rng = np.random.RandomState(27658622)
        self.rvs = rvs = self.copula.rvs(nobs, rng=rng)
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

    def test_seed_default_rng(self):
        seed1 = np.random.default_rng()
        seed2 = np.random.default_rng()
        seed2.bit_generator.state = seed1.bit_generator.state
        nobs = 2000
        rvs1 = self.copula.rvs(nobs, rng=seed1)
        rvs2 = self.copula.rvs(nobs, rng=seed2)
        assert_allclose(rvs1, rvs2)

    @pytest.mark.parametrize(
        "rng", [None, 0, np.random.RandomState(0), np.random.default_rng(0)]
    )
    def test_rng_types(self, rng):
        nobs = 2000
        rvs = self.copula.rvs(nobs, rng=rng)
        assert isinstance(rvs, np.ndarray)
        assert np.issubdtype(rvs.dtype, np.float64)

    @pytest.mark.parametrize("seed", ["random_state", "generator", "qmc", 0])
    def test_seed(self, seed):
        if seed == "random_state":
            seed1 = np.random.RandomState(0)
            seed2 = np.random.RandomState(0)
        elif seed == "generator":
            seed1 = np.random.default_rng(0)
            seed2 = 0
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
            rvs1 = self.copula.rvs(nobs, rng=seed1)
        with pytest_warns(FutureWarning):
            rvs2 = self.copula.rvs(nobs, random_state=seed2)
        assert_allclose(rvs1, rvs2)


class TestIndependenceCopula(CheckCopula):
    copula = IndependenceCopula()
    dim = 2
    pdf_u = np.ones(10)
    cdf_u = np.prod(CheckCopula.u, axis=1)


class CheckRvsDim:
    # class to check rvs for larger k_dim
    def test_rvs(self):
        nobs = 2000
        use_pdf = getattr(self, "use_pdf", False)
        # seed adjusted to avoid test failures with rvs numbers
        rng = np.random.RandomState(97651629)  # 27658622)
        rvs = self.copula.rvs(nobs, rng=rng)
        chi2t, rvs = check_cop_rvs(
            self.copula, rvs=rvs, nobs=nobs, k=10, use_pdf=use_pdf, rng=rng
        )
        assert chi2t.pvalue > 0.1

        k = self.dim
        assert k == rvs.shape[1]

        tau_cop = self.copula.tau()

        if np.ndim(tau_cop) == 2:
            # elliptical copula with tau matrix
            tau = np.eye(k)
            for i in range(k):
                for j in range(i + 1, k):
                    tau_ij = stats.kendalltau(rvs[..., i], rvs[..., j])[0]
                    tau[i, j] = tau[j, i] = tau_ij
            atol = 0.05
        else:
            taus = [
                stats.kendalltau(rvs[..., i], rvs[..., j])[0]
                for i in range(k)
                for j in range(i + 1, k)
            ]
            tau = np.mean(taus)
            atol = 0

        assert_allclose(tau, tau_cop, rtol=0.05, atol=atol)
        theta_est = self.copula.fit_corr_param(rvs)
        # specific to archimedean
        assert_allclose(theta_est, self.copula.args[0], rtol=0.1, atol=atol)

    @pytest.mark.parametrize(
        "rng", [0, np.random.RandomState(0), np.random.default_rng(0)]
    )
    def test_rng_types(self, rng):
        nobs = 2000
        rvs = self.copula.rvs(nobs, rng=rng)
        assert isinstance(rvs, np.ndarray)
        assert np.issubdtype(rvs.dtype, np.float64)

        with pytest_warns(FutureWarning):
            self.copula.rvs(nobs, random_state=rng)


class TestGaussianCopula(CheckModernCopula):
    copula = GaussianCopula(corr=[[1.0, 0.8], [0.8, 1.0]])
    dim = 2
    pdf_u = [
        1.03308741,
        0.06507279,
        0.72896012,
        0.65389439,
        16.45012399,
        0.34813218,
        0.06768115,
        0.08168840,
        0.40521741,
        1.26723470,
    ]
    cdf_u = [
        0.31906854,
        0.06230196,
        0.19284669,
        0.39952707,
        0.98144792,
        0.25677003,
        0.05932818,
        0.09605404,
        0.35211017,
        0.20885480,
    ]

    def test_rvs(self):
        # copied from student t test,
        # currently inconsistent with non-elliptical copulas
        super().test_rvs()
        rs = np.random.RandomState(97651627)
        chi2t, rvs = check_cop_rvs(
            self.copula, rvs=self.rvs, nobs=2000, k=10, use_pdf=True, rng=rs
        )
        assert chi2t.pvalue > 0.1
        tau = stats.kendalltau(*rvs.T)[0]
        tau_cop = self.copula.tau()
        assert_allclose(tau, tau_cop, rtol=0.05)

        corr_est = self.copula.fit_corr_param(rvs)
        assert_allclose(corr_est, 0.8, rtol=0.1)


class TestGaussianCopula3d(CheckRvsDim):
    copula = GaussianCopula(
        corr=[[1.0, 0.8, 0.1], [0.8, 1.0, 0.3], [0.1, 0.3, 1.0]], k_dim=3
    )
    dim = 3
    use_pdf = False


class TestStudentTCopula(CheckModernCopula):
    copula = StudentTCopula(corr=[[1.0, 0.8], [0.8, 1.0]], df=2)
    dim = 2
    pdf_u = [
        0.8303065,
        0.1359839,
        0.5157746,
        0.4776421,
        26.2173959,
        0.3070661,
        0.1349173,
        0.1597064,
        0.3303230,
        1.0482301,
    ]
    cdf_u = [
        0.31140349,
        0.05942746,
        0.18548601,
        0.39143974,
        0.98347259,
        0.24894028,
        0.05653947,
        0.09210693,
        0.34447385,
        0.20429882,
    ]

    def test_cdf(self):
        pytest.skip("Not implemented.")

    def test_rvs(self):
        super().test_rvs()
        rs = np.random.RandomState(97651625)
        chi2t, rvs = check_cop_rvs(
            self.copula, rvs=self.rvs, nobs=2000, k=10, use_pdf=True, rng=rs
        )
        assert chi2t.pvalue > 0.1
        tau = stats.kendalltau(*rvs.T)[0]
        tau_cop = self.copula.tau()
        assert_allclose(tau, tau_cop, rtol=0.05)


class TestStudentTCopula3d(CheckRvsDim):
    copula = StudentTCopula(
        corr=[[1.0, 0.8, 0.1], [0.8, 1.0, 0.3], [0.1, 0.3, 1.0]], k_dim=3, df=10
    )
    dim = 3
    use_pdf = True


class TestClaytonCopula(CheckModernCopula):
    copula = ClaytonCopula(theta=1.2)
    dim = 2
    pdf_u = [
        1.0119836,
        0.2072728,
        0.8148839,
        0.9481976,
        2.1419659,
        0.6828507,
        0.2040454,
        0.2838497,
        0.8197787,
        1.1096360,
    ]
    cdf_u = [
        0.28520375,
        0.06101690,
        0.17703377,
        0.36848218,
        0.97772088,
        0.24082057,
        0.05811908,
        0.09343934,
        0.33012582,
        0.18738753,
    ]


class TestClaytonCopula_3d(CheckRvsDim):
    # currently only checks rvs
    copula = ClaytonCopula(theta=1.2, k_dim=3)
    dim = 3


class TestFrankCopula(CheckModernCopula):
    copula = FrankCopula(theta=3)
    dim = 2
    pdf_u = [
        0.9646599,
        0.5627195,
        0.8941964,
        0.8364614,
        2.9570945,
        0.6665601,
        0.5779906,
        0.5241333,
        0.7156741,
        1.1074024,
    ]
    cdf_u = [
        0.27467496,
        0.05492539,
        0.15995939,
        0.36750702,
        0.97782283,
        0.23412757,
        0.05196265,
        0.08676979,
        0.32803721,
        0.16320730,
    ]


class TestFrankCopula_3d(CheckRvsDim):
    copula = FrankCopula(theta=3, k_dim=3)
    dim = 3


class TestGumbelCopula(CheckModernCopula):
    copula = GumbelCopula(theta=1.5)
    dim = 2
    pdf_u = [
        1.0391696,
        0.6539579,
        0.9878446,
        0.8679504,
        16.6030932,
        0.7542073,
        0.6668307,
        0.6275887,
        0.7477991,
        1.1564864,
    ]
    cdf_u = [
        0.27194634,
        0.05484380,
        0.15668190,
        0.37098420,
        0.98176346,
        0.23422865,
        0.05188260,
        0.08659615,
        0.33086960,
        0.15803914,
    ]


class TestGumbelCopula_3d(CheckRvsDim):
    copula = GumbelCopula(theta=1.5, k_dim=3)
    dim = 3


# ---------------------------------------------------------------------
# Copula.plot_scatter, Copula.plot_pdf, Copula.tau_simulated,
# GaussianCopula/StudentTCopula.dependence_tail, StudentTCopula.spearmans_rho
# and other_copulas.rvs_kernel.
# ---------------------------------------------------------------------


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_plot_scatter_generated_sample(close_figures):
    copula = GaussianCopula(corr=0.5)
    fig, sample = copula.plot_scatter(nobs=200, rng=np.random.default_rng(0))
    assert sample.shape == (200, 2)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "u"
    assert ax.get_ylabel() == "v"
    # the plotted scatter points should be exactly the returned sample
    offsets = np.asarray(ax.collections[0].get_offsets())
    assert_allclose(offsets, sample)


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_plot_scatter_given_sample(close_figures):
    copula = GaussianCopula(corr=0.5)
    sample = copula.rvs(nobs=100, rng=np.random.default_rng(1))
    fig, returned = copula.plot_scatter(sample=sample)
    # a provided sample is used as-is, not regenerated
    assert returned is sample
    ax = fig.axes[0]
    offsets = np.asarray(ax.collections[0].get_offsets())
    assert_allclose(offsets, sample)


def test_plot_scatter_raises_for_higher_dim():
    cop3 = GaussianCopula(
        corr=[[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]], k_dim=3
    )
    with pytest.raises(ValueError):
        cop3.plot_scatter()


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_plot_pdf_grid_matches_pdf(close_figures):
    copula = GaussianCopula(corr=0.6)
    # grid parameters hardcoded in Copula.plot_pdf
    eps = 1e-4
    n_samples = 100
    uu, vv = np.meshgrid(
        np.linspace(eps, 1 - eps, n_samples),
        np.linspace(eps, 1 - eps, n_samples),
    )
    expected_points = np.vstack([uu.ravel(), vv.ravel()]).T

    with mock.patch.object(copula, "pdf", wraps=copula.pdf) as mock_pdf:
        fig = copula.plot_pdf()

    # plot_pdf should evaluate copula.pdf exactly once, on the expected grid
    assert mock_pdf.call_count == 1
    used_points = mock_pdf.call_args[0][0]
    assert_allclose(used_points, expected_points)

    ax = fig.axes[0]
    assert ax.get_xlabel() == "u"
    assert ax.get_ylabel() == "v"
    assert_allclose(ax.get_xlim(), (0, 1))
    assert_allclose(ax.get_ylim(), (0, 1))

    # the contour color range should be the 5th/95th percentile of the
    # pdf evaluated on that same grid
    expected_pdf = copula.pdf(expected_points).T.reshape(uu.shape)
    expected_min = np.nanpercentile(expected_pdf, 5)
    expected_max = np.nanpercentile(expected_pdf, 95)

    clim = None
    for artist in fig.findobj():
        if hasattr(artist, "get_clim"):
            clim = artist.get_clim()
            break
    assert clim is not None
    assert_allclose(clim, (expected_min, expected_max))


tau_simulated_cases = [
    GaussianCopula(corr=0.6),
    StudentTCopula(corr=0.6, df=4),
    ClaytonCopula(theta=1.5),
    FrankCopula(theta=3),
    GumbelCopula(theta=1.5),
]
tau_simulated_ids = ["gaussian", "student_t", "clayton", "frank", "gumbel"]


@pytest.mark.parametrize("copula", tau_simulated_cases, ids=tau_simulated_ids)
def test_tau_simulated(copula):
    # tau_simulated should recover each copula's own exact/closed-form
    # Kendall's tau (``.tau()``) up to Monte Carlo error.
    rng = np.random.default_rng(987654321)
    tau_sim = copula.tau_simulated(nobs=4000, rng=rng)
    tau_exact = copula.tau()
    assert_allclose(tau_sim, tau_exact, atol=0.05)


def test_gaussian_copula_dependence_tail():
    # Gaussian copula has zero tail dependence by construction,
    # Joe (2014) p. 182.
    copula = GaussianCopula(corr=0.9)
    assert copula.dependence_tail() == (0, 0)
    # the corr argument is documented as ignored
    assert copula.dependence_tail(corr=0.1) == (0, 0)


@pytest.mark.parametrize("corr,df", [(0.5, 4), (0.8, 2), (-0.3, 8), (0.0, 6)])
def test_student_t_copula_dependence_tail_formula(corr, df):
    # hand-computed closed form, Joe (2014) p. 182:
    # lambda = 2 * t_{df+1}(-sqrt((df+1)(1-rho)/(1+rho)))
    copula = StudentTCopula(corr=corr, df=df)
    lower, upper = copula.dependence_tail()

    t_stat = -np.sqrt((df + 1) * (1 - corr) / (1 + corr))
    expected = 2 * stats.t.cdf(t_stat, df + 1)
    assert_allclose(lower, expected, rtol=1e-12)
    # the bivariate t copula is exchangeable, so lower == upper
    assert_allclose(upper, expected, rtol=1e-12)


def test_student_t_copula_dependence_tail_limits():
    # as df -> infinity the t copula converges to the Gaussian copula,
    # whose tail dependence is exactly zero.
    copula = StudentTCopula(corr=0.5, df=10000)
    lower, upper = copula.dependence_tail()
    assert lower < 1e-3
    assert upper < 1e-3

    # tail dependence increases toward 1 as corr -> 1
    copula_high = StudentTCopula(corr=0.999, df=4)
    lower_high, upper_high = copula_high.dependence_tail()
    assert lower_high > 0.9


@pytest.mark.parametrize("corr", [-0.7, -0.2, 0.0, 0.4, 0.9])
def test_student_t_copula_spearmans_rho_formula(corr):
    # hand-computed closed form, Joe (2014) p. 182:
    # rho_s = (6 / pi) * arcsin(rho / 2)
    copula = StudentTCopula(corr=corr, df=5)
    rho_s = copula.spearmans_rho()
    expected = 6 * np.arcsin(corr / 2) / np.pi
    assert_allclose(rho_s, expected, rtol=1e-12)


def test_student_t_copula_spearmans_rho_simulated():
    # cross check the closed form against an independent, simulation-based
    # Spearman's rho.
    corr = 0.6
    copula = StudentTCopula(corr=corr, df=6)
    rng = np.random.default_rng(2024)
    sample = copula.rvs(nobs=4000, rng=rng)
    rho_hat = stats.spearmanr(sample[:, 0], sample[:, 1])[0]
    rho_exact = copula.spearmans_rho()
    assert_allclose(rho_hat, rho_exact, atol=0.05)


def test_rvs_kernel_small_bandwidth_stays_near_source():
    # for a very small bandwidth the Beta-kernel perturbation concentrates
    # tightly around the resampled source point: the mean of
    # Beta(x/bw, (1-x)/bw + 1) is x / (1 + bw) -> x as bw -> 0.
    rng = np.random.default_rng(0)
    sample = rng.uniform(0.05, 0.95, size=(50, 2))
    krvs, idx, xi = rvs_kernel(sample, size=500, bw=1e-4, rng=rng, return_extras=True)
    assert krvs.shape == (500, 2)
    assert_allclose(xi, sample[idx])
    assert_allclose(krvs, xi, atol=0.05)


def test_rvs_kernel_reproducible_with_rng():
    # rvs_kernel's docstring promises that passing `rng` fully determines
    # the output, matching every other ``rvs``-like function in this module.
    sample = np.random.default_rng(0).uniform(0.05, 0.95, size=(50, 2))
    r1 = rvs_kernel(sample, size=20, bw=0.5, rng=np.random.default_rng(123))
    r2 = rvs_kernel(sample, size=20, bw=0.5, rng=np.random.default_rng(123))
    assert_allclose(r1, r2)


def test_rvs_kernel_preserves_dependence():
    # a beta-kernel bootstrap (small bandwidth) of a sample drawn from a
    # copula with known Kendall's tau should approximately reproduce that
    # same tau, and its margins should stay close to Uniform(0, 1).
    rng = np.random.default_rng(20240815)
    corr = 0.6
    copula = GaussianCopula(corr=corr)
    sample = copula.rvs(nobs=3000, rng=rng)

    krvs = rvs_kernel(sample, size=5000, bw=0.001, rng=rng)
    assert krvs.shape == (5000, 2)
    assert np.all((krvs >= 0) & (krvs <= 1))

    tau_hat = stats.kendalltau(krvs[:, 0], krvs[:, 1])[0]
    tau_exact = copula.tau()
    assert_allclose(tau_hat, tau_exact, atol=0.05)
    assert_allclose(krvs.mean(axis=0), 0.5, atol=0.05)
