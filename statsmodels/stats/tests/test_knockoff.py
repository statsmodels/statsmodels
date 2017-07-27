import numpy as np
from statsmodels.stats import knockoff_regeffects as kr
from statsmodels.stats._knockoff import (RegressionFDR,
                                         _design_knockoff_equi,
                                         _design_knockoff_sdp)
from numpy.testing import assert_allclose, assert_array_equal
from numpy.testing.decorators import slow

try:
    import cvxopt
    has_cvxopt = True
except:
    has_cvxopt = False


def test_equi():
    # Test the structure of the equivariant knockoff construction.

    np.random.seed(2342)
    exog = np.random.normal(size=(10, 4))

    exog1, exog2, sl = _design_knockoff_equi(exog)

    exoga = np.concatenate((exog1, exog2), axis=1)

    gmat = np.dot(exoga.T, exoga)

    cm1 = gmat[0:4, 0:4]
    cm2 = gmat[4:, 4:]
    cm3 = gmat[0:4, 4:]

    assert_allclose(cm1, cm2, rtol=1e-4, atol=1e-4)
    assert_allclose(cm1 - cm3, np.diag(sl * np.ones(4)), rtol=1e-4, atol=1e-4)


def test_sdp():
    # Test the structure of the SDP knockoff construction.

    if not has_cvxopt:
        return

    np.random.seed(2342)
    exog = np.random.normal(size=(10, 4))

    exog1, exog2, sl = _design_knockoff_sdp(exog)

    exoga = np.concatenate((exog1, exog2), axis=1)

    gmat = np.dot(exoga.T, exoga)

    cm1 = gmat[0:4, 0:4]
    cm2 = gmat[4:, 4:]
    cm3 = gmat[0:4, 4:]

    assert_allclose(cm1, cm2, rtol=1e-4, atol=1e-4)
    assert_allclose(cm1 - cm3, np.diag(sl * np.ones(4)), rtol=1e-5, atol=1e-5)


def test_testers():
    # Smoke test

    np.random.seed(2432)

    n = 200
    p = 50

    y = np.random.normal(size=n)
    x = np.random.normal(size=(n, p))

    testers = [kr.CorrelationEffects(),
               kr.ForwardEffects(pursuit=False),
               kr.ForwardEffects(pursuit=True),
               kr.OLSEffects()]

    for method in "equi", "sdp":

        if method == "sdp" and not has_cvxopt:
            continue

        for tv in testers:
            RegressionFDR(y, x, tv, design_method=method)


@slow
def test_sim():
    # This function assesses the performance of the knockoff approach
    # relative to its theoretical claims.

    np.random.seed(43234)
    npos = 30
    target_fdr = 0.2
    nrep = 10

    testers = [[kr.CorrelationEffects(), 300, 100, 6],
               [kr.ForwardEffects(pursuit=False), 300, 100, 3.5],
               [kr.ForwardEffects(pursuit=True), 300, 100, 3.5],
               [kr.OLSEffects(), 3000, 200, 3.5]]

    for method in "equi", "sdp":

        if method == "sdp" and not has_cvxopt:
            continue

        for tester_info in testers:

            fdr = 0
            power = 0
            tester = tester_info[0]
            n = tester_info[1]
            p = tester_info[2]
            es = tester_info[3]

            for k in range(nrep):

                x = np.random.normal(size=(n, p))
                x /= np.sqrt(np.sum(x*x, 0))

                coeff = es * (-1)**np.arange(npos)
                y = np.dot(x[:, 0:npos], coeff) + np.random.normal(size=n)

                kn = RegressionFDR(y, x, tester)

                tr = kn.threshold(target_fdr)
                cp = np.sum(kn.stats >= tr)
                cp = max(cp, 1)
                fp = np.sum(kn.stats[npos:] >= tr)
                fdr += fp/cp
                power += np.mean(kn.stats[0:npos] >= tr)

                estimated_fdr = (np.sum(kn.stats <= -tr) /
                                 (1 + np.sum(kn.stats >= tr)))
                assert_array_equal(estimated_fdr < target_fdr, True)

            power /= nrep
            fdr /= nrep

            assert_array_equal(power > 0.6, True)
            assert_array_equal(fdr < target_fdr + 0.05, True)
