import numpy as np
from collections import namedtuple
from .. import kde_methods as km
from ..kde_utils import Grid
from scipy import stats, linalg
from .. import kernels
import scipy

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

class Parameters(object):
    """Empty class to hold values."""
    pass

def createParams_norm():
    """
    Create the parameters to test using a 1D Gaussian distribution
    """
    params = Parameters()
    params.dist = stats.norm(0, 1)
    params.sizes = [128, 256, 201]
    params.vs = [generate(params.dist, s, -5, 5) for s in params.sizes]
    params.args = {}
    params.weights = [params.dist.pdf(v) for v in params.vs]
    params.adjust = [1 - ws for ws in params.weights]
    params.xs = np.r_[-5:5:512j]
    params.lower = -5
    params.upper = 5
    params.methods = methods_1d
    return params

def createParams_lognorm():
    """
    Create the parameters to test using a 1D log normal distribution
    """
    params = Parameters()
    params.dist = stats.lognorm(1)
    params.sizes = [128, 256, 201]
    params.args = {}
    params.vs = [generate(params.dist, s, 0.001, 20) for s in params.sizes]
    params.vs = [v[v < 20] for v in params.vs]
    params.xs = np.r_[0:20:512j]
    params.weights = [params.dist.pdf(v) for v in params.vs]
    params.adjust = [1 - ws for ws in params.weights]
    params.lower = 0
    params.upper = 20
    params.methods = methods_log
    return params

def createParams_normnd(ndim):
    """
    Create the paramters to test using a nD Gaussian distribution
    """
    params = Parameters()
    params.dist = stats.multivariate_normal(cov=np.eye(ndim))
    params.sizes = [32, 64, 128]
    params.vs = [generate_nd(params.dist, s) for s in params.sizes]
    params.weights = [params.dist.pdf(v) for v in params.vs]
    params.adjust = [1 - ws for ws in params.weights]
    params.xs = [np.r_[-5:5:512j]] * ndim
    params.lower = [-5] * ndim
    params.upper = [5] * ndim
    params.methods = methods_nd
    params.args = {}
    return params

def createParams_nc():
    """
    Create the parameters to test using  a nC poisson distribution
    """
    params = Parameters()
    params.dist = stats.poisson(12)
    params.sizes = [128, 256, 201]
    params.vs = [generate_nc(params.dist, s) for s in params.sizes]
    params.weights = [params.dist.pmf(v) for v in params.vs]
    params.args = {}
    params.methods = methods_nc
    return params

def createParams_multivariate():
    """
    Create the parameters to test using a poisson distribution and two normals
    """
    params = Parameters()
    params.d1 = stats.norm(0, 3)
    params.d2 = stats.poisson(12)
    params.sizes = [64, 128, 101]
    params.vs = [generate_multivariate(s, params.d1, params.d2) for s in params.sizes]
    params.weights = [params.d1.pdf(v[:, 0]) for v in params.vs]
    params.upper = [5, max(v[:, 1].max() for v in params.vs)]
    params.lower = [-5, 0]
    params.args = {}
    params.methods1 = methods_1d + methods_nc + methods_nc
    params.methods2 = methods_nc + methods_1d + methods_nc[::-1]
    params.nb_methods = len(params.methods1)
    return params

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

kernels1d = [test_kernel(kernels.Gaussian1D, 1, 1, True),
             test_kernel(kernels.TriCube, 1, 1, True),
             test_kernel(kernels.Epanechnikov, 10, 1, True),
             test_kernel(kernels.GaussianOrder4, 10, 0, False),  # Bad for precision because of high frequencies
             test_kernel(kernels.EpanechnikovOrder4, 1000, 0, False)]  # Bad for precision because of high frequencies

kernelsnc = [test_kernel(kernels.AitchisonAitken, 1, 1, True),
             test_kernel(kernels.WangRyzin, 1, 1, True)]

kernelsnd = [test_kernel(kernels.Gaussian, 1, 1, True)]
