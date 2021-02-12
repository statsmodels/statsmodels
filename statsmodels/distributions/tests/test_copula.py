# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 23:32:57 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from numpy.testing import assert_allclose
from scipy import stats
import pytest

from statsmodels.distributions.copula.copulas import CopulaDistribution
from statsmodels.distributions.copula.archimedean import (
    ArchimedeanCopula, FrankCopula)
from statsmodels.distributions.copula.extreme_value import (
    ExtremeValueCopula, copula_bv_ev)
import statsmodels.distributions.copula.transforms as tra
import statsmodels.distributions.copula.depfunc_ev as trev

uniform = stats.uniform


ev_list = [
    [trev.transform_bilogistic, 0.5, 0.9, (0.25, 0.05), 0.5],
    [trev.transform_tawn, 0.5, 0.9, (0.5, 0.5, 0.5), 0.4724570876035117],
    # note evd has asymmetry reversed, interchange variables
    [trev.transform_tawn2, 0.9, 0.5, (0.25, 0.05), 0.464357480263932],
    [trev.transform_tawn2, 0.9, 0.5, (0.5, 0.25), 0.4916117128670654],
    [trev.transform_tawn2, 0.5, 0.9, (0.5, 0.25), 0.48340673415789],
    # note evd has parameter for hr 1/lmbda (inverse of our parameter)
    [trev.transform_hr, 0.5, 0.9, (2,), 0.4551235014298542],
    [trev.transform_joe, 0.5, 0.9, (0.5, 0.75, 1/0.25), 0.4543698299835434],
    [trev.transform_joe, 0.9, 0.5, (0.5, 0.75, 1/0.25), 0.4539773435983587],

    # tev is against R `copula` package
    # > cop = tevCopula(0.8, df = 4)
    # > pCopula(c(0.5, 0.75), cop)
    # [1] 0.456807960674953
    # > pCopula(c(0.5, 0.9), cop)
    # [1] 0.4911039761533587
    [trev.transform_tev, 0.5, 0.75, (0.8, 4), 0.456807960674953],
    [trev.transform_tev, 0.5, 0.9, (0.8, 4), 0.4911039761533587],
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
    [trev.transform_tawn, 0.5, 0.9, (0.5, 0.5, 0.25), 0.4386367545837274],
    ]


@pytest.mark.parametrize("case", ev_list)
def test_ev_copula(case):
    # check ev copulas, cdf and transform against R `evt` package
    ev_tr, v1, v2, args, res1 = case
    res = copula_bv_ev([v1, v2], ev_tr, args=args)
    assert_allclose(res, res1, rtol=1e-13)


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
    # check ev copulas, cdf and transform against R `evt` package
    ev_tr, v1, v2, args, res1 = case
    u = [v1, v2]
    res = copula_bv_ev(u, ev_tr, args=args)
    assert_allclose(res, res1, rtol=1e-13)

    cev = ExtremeValueCopula(ev_tr)
    cdf1 = cev.cdf(u, args)
    assert_allclose(cdf1, res1, rtol=1e-13)

    cev = CopulaDistribution([uniform, uniform], cev, copargs=args)
    cdfd = cev.cdf(np.array(u), args=args)
    assert_allclose(cdfd, res1, rtol=1e-13)
    assert cdfd.shape == ()

    # using list u
    cdfd = cev.cdf(u, args=args)
    assert_allclose(cdfd, res1, rtol=1e-13)
    assert cdfd.shape == ()

    # check vector values for u
    # bilogistic is not vectorized, uses integrate.quad
    if ev_tr != trev.transform_bilogistic:
        cdfd = cev.cdf(np.array(u) * np.ones((3, 1)), args=args)
        assert_allclose(cdfd, res1, rtol=1e-13)
        assert cdfd.shape == (3, )


@pytest.mark.parametrize("case", cop_list)
def test_copulas_distr(case):
    # check ev copulas, cdf and transform against R `copula` package
    cop_tr, v1, v2, args, cdf2, pdf2 = case
    u = [v1, v2]
    ca = ArchimedeanCopula(cop_tr())
    cdf1 = ca.cdf(u, args=args)
    pdf1 = ca.pdf(u, args=args)

    cad = CopulaDistribution([uniform, uniform], ca, copargs=args)
    cdfd = cad.cdf(np.array(u), args=args)
    assert_allclose(cdfd, cdf1, rtol=1e-13)
    assert cdfd.shape == ()

    # check pdf
    pdfd = cad.pdf(np.array(u), args=args)
    assert_allclose(pdfd, pdf1, rtol=1e-13)
    assert cdfd.shape == ()

    # using list u
    cdfd = cad.cdf(u, args=args)
    assert_allclose(cdfd, cdf1, rtol=1e-13)
    assert cdfd.shape == ()

    assert_allclose(cdf1, cdf2, rtol=1e-13)
    assert_allclose(pdf1, pdf2, rtol=1e-13)

    # check vector values for u
    cdfd = cad.cdf(np.array(u) * np.ones((3, 1)), args=args)
    assert_allclose(cdfd, cdf2, rtol=1e-13)
    assert cdfd.shape == (3, )

    # check mv, check at marginal cdf
    cdfmv = ca.cdf([v1, v2, 1], args=args)
    assert_allclose(cdfmv, cdf1, rtol=1e-13)
    assert cdfd.shape == (3, )


@pytest.mark.parametrize("case", gev_list)
def test_gev_genextreme(case):
    gev = stats.genextreme(0)
    # check ev copulas, cdf and transform against R `evt` package
    ev_tr, v1, v2, args, res1 = case
    y = [v1, v2]
    u = gev.cdf(y)
    res = copula_bv_ev(u, ev_tr, args=args)
    assert_allclose(res, res1, rtol=1e-13)

    cev = ExtremeValueCopula(ev_tr)
    cdf1 = cev.cdf(u, args)
    assert_allclose(cdf1, res, rtol=1e-13)

    cev = CopulaDistribution([gev, gev], cev, copargs=args)
    cdfd = cev.cdf(np.array(y), args=args)
    assert_allclose(cdfd, res1, rtol=1e-13)


class TestFrank(object):

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
