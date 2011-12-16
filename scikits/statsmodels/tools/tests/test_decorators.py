import numpy.testing as npt
from numpy import array
from scikits.statsmodels.tools.decorators import transform2, set_transform

def _olsen_reparam(params):
    """
    Go from true parameters to gamma and theta of Olsen

    gamma = beta/sigma
    theta = 1/sigma
    """
    beta, sigma  = params[:-1], params[-1]
    theta = 1./sigma
    gamma = beta/sigma
    return gamma, theta

class A(object):

    #self._transformation = _olsen_reparam

    #_transform = transform_factory(_olsen_reparam)

    @transform2(_olsen_reparam)
    #@_transform
    def loglike(self, params, extra=None):
        """
        I am the help of the loglike
        """
        return params

    @set_transform
    def fit(self, params):
        """
        I am the help of the fit function.
        """
        params = self.loglike(params)
        return params

def test_transform():
    a = A()
    res = a.fit(array([1.,2,3]))
    npt.assert_equal(res[0], array([1./3., 2/3.]))
    npt.assert_equal(res[1], 1/3.)

