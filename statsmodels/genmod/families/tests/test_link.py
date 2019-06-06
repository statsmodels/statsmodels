"""
Test functions for genmod.families.links
"""
from statsmodels.compat.python import range
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import statsmodels.genmod.families as families
from statsmodels.tools import numdiff as nd

# Family instances
links = families.links
logit = links.Logit()
inverse_power = links.inverse_power()
sqrt = links.sqrt()
inverse_squared = links.inverse_squared()
identity = links.identity()
log = links.log()
probit = links.probit()
cauchy = links.cauchy()
cloglog = links.CLogLog()
negbinom = links.NegativeBinomial()

# TODO: parametrize all these tess
Links = [logit, inverse_power, sqrt, inverse_squared, identity,
         log, probit, cauchy, cloglog, negbinom]


def get_domainvalue(link):
    """
    Get a value in the domain for a given family.
    """
    z = -np.log(np.random.uniform(0, 1))
    if isinstance(link, links.CLogLog):  # prone to overflow
        z = min(z, 3)
    elif isinstance(link, links.NegativeBinomial):
        # domain is negative numbers
        z = -z
    return z


def test_inverse():
    # Logic check that link.inverse(link) and link(link.inverse)
    # are the identity.
    np.random.seed(3285)

    for link in Links:
        for k in range(10):
            p = np.random.uniform(0, 1)  # In domain for all families
            d = link.inverse(link(p))
            assert_allclose(d, p, atol=1e-8, err_msg=str(link))

            z = get_domainvalue(link)
            d = link(link.inverse(z))
            assert_allclose(d, z, atol=1e-8, err_msg=str(link))


def test_deriv():
    # Check link function derivatives using numeric differentiation.

    np.random.seed(24235)

    for link in Links:
        for k in range(10):
            p = np.random.uniform(0, 1)
            d = link.deriv(p)
            da = nd.approx_fprime(np.r_[p], link)
            assert_allclose(d, da, rtol=1e-6, atol=1e-6,
                            err_msg=str(link))


def test_deriv2():
    # Check link function second derivatives using numeric differentiation.

    np.random.seed(24235)

    for link in Links:
        # TODO: Resolve errors with the numeric derivatives
        if isinstance(link, links.probit):
            continue
        for k in range(10):
            p = np.random.uniform(0, 1)
            p = np.clip(p, 0.01, 0.99)
            if isinstance(link, links.cauchy):
                p = np.clip(p, 0.03, 0.97)
            d = link.deriv2(p)
            da = nd.approx_fprime(np.r_[p], link.deriv)
            assert_allclose(d, da, rtol=1e-6, atol=1e-6,
                            err_msg=str(link))


def test_inverse_deriv():
    # Logic check that inverse_deriv equals 1/link.deriv(link.inverse)

    np.random.seed(24235)

    for link in Links:
        for k in range(10):
            z = -np.log(np.random.uniform())  # In domain for all families
            d = link.inverse_deriv(z)
            f = 1 / link.deriv(link.inverse(z))
            assert_allclose(d, f, rtol=1e-8, atol=1e-10,
                            err_msg=str(link))


def test_invlogit_stability():
    z = [1123.4910007309222, 1483.952316802719, 1344.86033748641,
         706.339159002542, 1167.9986375146532, 663.8345826933115,
         1496.3691686913917, 1563.0763842182257, 1587.4309332296314,
         697.1173174974248, 1333.7256198289665, 1388.7667560586933,
         819.7605431778434, 1479.9204150555015, 1078.5642245164856,
         480.10338454985896, 1112.691659145772, 534.1061908007274,
         918.2011296406588, 1280.8808515887802, 758.3890788775948,
         673.503699841035, 1556.7043357878208, 819.5269028006679,
         1262.5711060356423, 1098.7271535253608, 1482.811928490097,
         796.198809756532, 893.7946963941745, 470.3304989319786,
         1427.77079226037, 1365.2050226373822, 1492.4193201661922,
         871.9922191949931, 768.4735925445908, 732.9222777654679,
         812.2382651982667, 495.06449978924525]
    zinv = logit.inverse(z)
    assert_equal(zinv, np.ones_like(z))
