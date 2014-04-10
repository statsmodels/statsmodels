"""
Test functions for genmod.families.links
"""
from statsmodels.compat.python import range
import numpy as np
from numpy.testing import assert_allclose
import statsmodels.genmod.families as families

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

Links = [logit, inverse_power, sqrt, inverse_squared, identity, log, probit, cauchy,
         cloglog, negbinom]


def test_inverse():

    ## Logic check that link.inverse(link) is the identity
    for link in Links:
        for k in range(10):
            p = np.random.uniform() # In domain for all families
            d = p - link.inverse(link(p))
            assert_allclose(d, 0, atol=1e-8)


def test_inverse_deriv():

    ## Logic check that inverse_deriv equals 1/link.deriv(link.inverse)
    for link in Links:
        for k in range(10):
            z = -np.log(np.random.uniform()) # In domain for all families
            d = link.inverse_deriv(z)
            f = 1 / link.deriv(link.inverse(z))
            assert_allclose(d, f, rtol=1e-8, atol=1e-10)


if __name__=="__main__":
    #run_module_suite()
    #taken from Fernando Perez:
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb'],
                       exit=False)
