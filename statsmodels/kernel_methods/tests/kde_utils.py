import numpy as np
from .. import kde_methods as km
from ..kde_utils import namedtuple, Grid
from scipy import stats, linalg
from .. import kernels
from ...compat.numpy import NumpyVersion
import scipy

class sp_multivariate_normal(object):
    """
    minimal version of multivariate_normal that just handle rvs and pdf with a covariance matrix
    """
    def __init__(self, mean=None, cov=1):
        cov = np.atleast_2d(cov)
        if cov.ndim != 2 or np.any(cov != cov.T):
            raise ValueError("The covariance matrix must be a symmetric, positive, matrix")
        if mean is None:
            mean = np.array([0.]*cov.shape[0])
        else:
            mean = np.atleast_1d(mean)
            if mean.ndim != 1:
                raise ValueError("The mean must be at most a 1D array")
            if cov.shape[0] != mean.shape[0]:
                raise ValueError("Error, the dimension of the covariance and the mean must be the same")
        self.trans = linalg.sqrtm(cov)
        self.inv_cov = linalg.inv(cov)
        ndim = cov.shape[0]
        self.factor = 1. / np.sqrt((2*np.pi)**ndim * linalg.det(self.inv_cov))
        self.ndim = ndim
        self.norm = stats.norm(0, 1)
        self.mean = mean[None, :]

    def rvs(self, N):
        xs = self.norm.rvs(N*self.ndim).reshape((N, self.ndim))
        return self.mean + np.dot(xs, self.trans)

    def pdf(self, xs):
        xs = np.atleast_2d(xs)
        if xs.ndim != 2 or xs.shape[1] != self.ndim:
            raise ValueError("The evaluation points must have shape (N,D) or (D,), "
                             " with D the dimension of the normal.")
        xs = xs - self.mean
        return self.factor * np.exp(-0.5*np.sum(xs * np.dot(xs, self.inv_cov), axis=1))

if NumpyVersion(scipy.__version__) < NumpyVersion('0.14.0'):
    multivariate_normal = sp_multivariate_normal
else:
    multivariate_normal = stats.multivariate_normal

def generate(dist, N, low, high):
    start = dist.cdf(low)
    end = dist.cdf(high)
    xs = np.linspace(1 - start, 1 - end, N)
    return dist.isf(xs)

def generate_nd(dist, N):
    np.random.seed(1)
    return dist.rvs(N)

def generate_nc(dist, N):
    np.random.seed(1)
    return dist.rvs(N)

def generate_multivariate(N, *dists):
    return np.vstack([d.rvs(N) for d in dists]).T

def setupClass_norm(cls):
    """
    Setup the class for a 1D normal distribution
    """
    cls.dist = stats.norm(0, 1)
    cls.sizes = [128, 256, 201]
    cls.vs = [generate(cls.dist, s, -5, 5) for s in cls.sizes]
    cls.args = {}
    cls.weights = [cls.dist.pdf(v) for v in cls.vs]
    cls.adjust = [1 - ws for ws in cls.weights]
    cls.xs = np.r_[-5:5:512j]
    cls.lower = -5
    cls.upper = 5
    cls.methods = methods_1d

def setupClass_lognorm(cls):
    cls.dist = stats.lognorm(1)
    cls.sizes = [128, 256, 201]
    cls.args = {}
    cls.vs = [generate(cls.dist, s, 0.001, 20) for s in cls.sizes]
    cls.vs = [v[v < 20] for v in cls.vs]
    cls.xs = np.r_[0:20:512j]
    cls.weights = [cls.dist.pdf(v) for v in cls.vs]
    cls.adjust = [1 - ws for ws in cls.weights]
    cls.lower = 0
    cls.upper = 20
    cls.methods = methods_log

def setupClass_normnd(cls, ndim):
    """
    Setting up the class for a nD normal distribution
    """
    cls.dist = multivariate_normal(cov=np.eye(ndim))
    cls.sizes = [32, 64, 128]
    cls.vs = [generate_nd(cls.dist, s) for s in cls.sizes]
    cls.weights = [cls.dist.pdf(v) for v in cls.vs]
    cls.adjust = [1 - ws for ws in cls.weights]
    cls.xs = [np.r_[-5:5:512j]] * ndim
    cls.lower = [-5] * ndim
    cls.upper = [5] * ndim
    cls.methods = methods_nd
    cls.args = {}

def setupClass_nc(cls):
    """
    Setting up the class for a nC poisson distribution
    """
    cls.dist = stats.poisson(12)
    cls.sizes = [128, 256, 201]
    cls.vs = [generate_nc(cls.dist, s) for s in cls.sizes]
    cls.weights = [cls.dist.pmf(v) for v in cls.vs]
    cls.args = {}
    cls.methods = methods_nc

def setupClass_multivariate(cls):
    """
    Setting up the class with a poisson distribution and two normals
    """
    cls.d1 = stats.norm(0, 3)
    cls.d2 = stats.poisson(12)
    cls.sizes = [64, 128, 101]
    cls.vs = [generate_multivariate(s, cls.d1, cls.d2) for s in cls.sizes]
    cls.weights = [cls.d1.pdf(v[:, 0]) for v in cls.vs]
    cls.upper = [5, max(v[:, 1].max() for v in cls.vs)]
    cls.lower = [-5, 0]
    cls.args = {}
    cls.methods1 = methods_1d + methods_nc + methods_nc
    cls.methods2 = methods_nc + methods_1d + methods_nc[::-1]
    cls.nb_methods = len(cls.methods1)

test_method = namedtuple('test_method',
                         ['instance', 'accuracy', 'grid_accuracy',
                          'normed_accuracy', 'bound_low', 'bound_high'])

methods_1d = [test_method(km.KDE1DMethod, 1e-5, 1e-4, 1e-5, False, False),
              test_method(km.Reflection1D, 1e-5, 1e-4, 1e-5, True, True),
              test_method(km.Cyclic1D, 1e-5, 1e-3, 1e-4, True, True),
              test_method(km.Renormalization, 1e-5, 1e-4, 1e-2, True, True),
              test_method(km.LinearCombination, 1e-1, 1e-1, 1e-1, True, False)]
methods_log = [test_method(km.Transform1D(km.LogTransform), 1e-5, 1e-4, 1e-5, True, False)]

methods_nd = [test_method(km.Cyclic, 1e-5, 1e-4, 1e-5, True, True),
              test_method(km.Cyclic, 1e-5, 1e-4, 1e-5, False, False),
              test_method(km.KDEnDMethod, 1e-5, 1e-4, 1e-5, False, False)]

methods_nc = [test_method(km.Ordered, 1e-5, 1e-4, 1e-5, False, False),
              test_method(km.Unordered, 1e-5, 1e-4, 1e-5, False, False)]

test_kernel = namedtuple('test_kernel', ['cls', 'precision_factor', 'var', 'positive'])

kernels1d = [test_kernel(kernels.normal1d, 1, 1, True),
             test_kernel(kernels.tricube, 1, 1, True),
             test_kernel(kernels.Epanechnikov, 10, 1, True),
             test_kernel(kernels.normal_order4, 10, 0, False),  # Bad for precision because of high frequencies
             test_kernel(kernels.Epanechnikov_order4, 1000, 0, False)]  # Bad for precision because of high frequencies

kernelsnc = [test_kernel(kernels.AitchisonAitken, 1, 1, True),
             test_kernel(kernels.WangRyzin, 1, 1, True)]

kernelsnd = [test_kernel(kernels.normal, 1, 1, True)]
