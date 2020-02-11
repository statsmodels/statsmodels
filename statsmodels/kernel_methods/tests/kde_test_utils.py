import pytest
import numpy as np
from collections import namedtuple
from .. import kde_methods as km
from scipy import stats
from .. import kernels, kde


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


test_method = namedtuple('test_method', [
    'instance', 'accuracy', 'grid_accuracy', 'normed_accuracy', 'bound_low',
    'bound_high'
])

methods_1d = [
    test_method(km.KDE1DMethod, 1e-5, 1e-4, 1e-5, False, False),
    test_method(km.Reflection1D, 1e-5, 1e-4, 1e-5, True, True),
    test_method(km.Cyclic1D, 1e-5, 1e-3, 1e-4, True, True),
    test_method(km.Renormalization, 1e-5, 1e-4, 1e-2, True, True),
    test_method(km.LinearCombination, 1e-1, 1e-1, 1e-1, True, False)
]
methods_log = [
    test_method(km.Transform1D(km.LogTransform), 1e-5, 1e-4, 1e-5, True, False)
]

methods_nd = [
    test_method(km.Cyclic, 1e-5, 1e-4, 1e-5, True, True),
    test_method(km.Cyclic, 1e-5, 1e-4, 1e-5, False, False),
    test_method(km.KDEnDMethod, 1e-5, 1e-4, 1e-5, False, False)
]

methods_nc = [
    test_method(km.Ordered, 1e-5, 1e-4, 1e-5, False, False),
    test_method(km.Unordered, 1e-5, 1e-4, 1e-5, False, False)
]

test_kernel = namedtuple('test_kernel',
                         ['cls', 'precision_factor', 'var', 'positive'])

kernels1d = [
    test_kernel(kernels.Gaussian1D, 1, 1, True),
    test_kernel(kernels.TriCube, 1, 1, True),
    test_kernel(kernels.Epanechnikov, 10, 1, True),
    test_kernel(kernels.GaussianOrder4, 10, 0,
                False),  # Bad for precision because of high frequencies
    test_kernel(kernels.EpanechnikovOrder4, 1000, 0, False)
]  # Bad for precision because of high frequencies

kernelsnc = [
    test_kernel(kernels.AitchisonAitken, 1, 1, True),
    test_kernel(kernels.WangRyzin, 1, 1, True)
]

kernelsnd = [test_kernel(kernels.Gaussian, 1, 1, True)]


class Parameters(object):
    """Empty class to hold values."""
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return 'Parameters({0}, ...)'.format(self.name)


dataset = namedtuple('dataset', ['vs', 'weights', 'adjust', 'lower', 'upper'])


def createTestSets_norm():
    """
    Create the parameters to test using a 1D Gaussian distribution dataset.
    """
    params = Parameters('norm')
    dist = stats.norm(0, 1)
    params.sizes = [128, 256, 201]
    params.xs = np.r_[-5:5:512j]
    params.methods = methods_1d
    params.can_adjust = True

    vs = [generate(dist, s, -5, 5) for s in params.sizes]
    weights = [dist.pdf(v) for v in vs]
    adjust = [1 - ws for ws in weights]
    params.dataset = dataset(vs, weights, adjust, lower=-5, upper=5)

    return params


def createTestSets_lognorm():
    """
    Create the parameters to test using a 1D log normal distribution dataset.
    """
    params = Parameters('lognorm')
    dist = stats.lognorm(1)
    params.sizes = [128, 256, 201]
    params.methods = methods_log
    params.xs = np.r_[0:20:513j][1:]
    params.can_adjust = True
    vs = [generate(dist, s, 0.001, 20) for s in params.sizes]
    vs = [v[v < 20] for v in vs]
    weights = [dist.pdf(v) for v in vs]
    adjust = [1 - ws for ws in weights]
    params.dataset = dataset(vs, weights, adjust, lower=0, upper=20)
    return params


def createTestSets_normnd(ndim):
    """
    Create the parameters to test using a nD Gaussian distribution dataset.
    """
    params = Parameters('normnd{0}'.format(ndim))
    dist = stats.multivariate_normal(cov=np.eye(ndim))
    params.sizes = [32, 64, 128]
    params.xs = [np.r_[-5:5:512j]] * ndim
    params.methods = methods_nd
    params.can_adjust = False

    vs = [generate_nd(dist, s) for s in params.sizes]
    weights = [dist.pdf(v) for v in vs]
    params.dataset = dataset(vs,
                             weights,
                             adjust=None,
                             lower=[-5] * ndim,
                             upper=[5] * ndim)

    return params


def createTestSets_nc():
    """
    Create the parameters to test using  a nC poisson distribution dataset.
    """
    params = Parameters('nc')
    dist = stats.poisson(12)
    params.sizes = [128, 256, 201]
    params.methods = methods_nc
    params.can_adjust = False

    vs = [generate_nc(dist, s) for s in params.sizes]
    weights = [dist.pmf(v) for v in vs]
    params.dataset = dataset(vs, weights, adjust=None, lower=None, upper=None)

    return params


def createTestSets_multivariate():
    """
    Create the parameters to test using a poisson distribution and two normals
    as dataset.
    """
    params = Parameters('multivariate')
    params.d1 = stats.norm(0, 3)
    params.d2 = stats.poisson(12)
    params.sizes = [64, 128, 101]
    params.args = {}
    params.methods1 = methods_1d + methods_nc + methods_nc
    params.methods2 = methods_nc + methods_1d + methods_nc[::-1]
    params.methods = zip(params.methods1, params.methods2)
    params.nb_methods = len(params.methods1)
    params.can_adjust = False

    vs = [generate_multivariate(s, params.d1, params.d2) for s in params.sizes]
    weights = [params.d1.pdf(v[:, 0]) for v in vs]
    upper = [5, max(v[:, 1].max() for v in vs)]
    lower = [-5, 0]
    params.dataset = dataset(vs,
                             weights,
                             adjust=None,
                             lower=lower,
                             upper=upper)

    return params


knownTestSets = dict(norm=createTestSets_norm(),
                     norm2d=createTestSets_normnd(2),
                     lognorm=createTestSets_lognorm(),
                     nc=createTestSets_nc(),
                     multivariate=createTestSets_multivariate())


def createKDE(vs, lower, upper, method):
    """
    Create a new KDE object.

    Arguments:
        parameters: A value stored in `knownTestSets`.
        vs: an ndarray containing the data points.
        lower: the lower bound of the data set.
        upper: the upper bound of the data set.
        method: Either an instance of `test_method` or a list of instances (for
            multivariate KDE), defining the methods to use for the KDE.
    """
    k = kde.KDE(vs)
    if isinstance(method, test_method):
        if method.instance is None:
            del k.method
        else:
            k.method = method.instance
        if method.bound_low:
            k.lower = lower
        else:
            del k.lower
        if method.bound_high:
            k.upper = upper
        else:
            del k.upper
    else:
        mv = km.Multivariate()
        k.method = mv

        n = len(method)
        axis_type = ''
        real_lower = [-np.inf] * n
        real_upper = [np.inf] * n
        for i, m in enumerate(method):
            method_instance = m.instance()
            mv.methods[i] = method_instance
            axis_type += str(method_instance.axis_type)
            if method_instance.axis_type != 'C':
                vs[:, i] = np.round(vs[:, i])
            if m.bound_low:
                real_lower[i] = lower[i]
            if m.bound_high:
                real_upper[i] = upper[i]
        k.exog = vs
        k.axis_type = axis_type
        k.lower = real_lower
        k.upper = real_upper
    return k


def kde_tester(check):
    """
    Decorator for a method needing a created KDE as input.

    The decorated method must accept three arguments:

        1. The KDE itself, pre-built with the data.
        2. The `test_method` object used to generate the KDE.
        3. The `data` object containing the data used to generate the KDE.

    The produced function is meant to be parametrized by the output of
    `generate_methods_data`, using the list of arguments in `kde_tester_args`.
    In addition, the `datasets` fixture must be imported in the local scope.
    """
    def fct(self, name, data, index, method, with_adjust, with_weights,
            method_name):
        params = knownTestSets[name]
        k = createKDE(data.vs[index], data.lower, data.upper, method)
        if with_adjust:
            k.adjust = data.adjust[index]
        if with_weights:
            k.weights = data.weights[index]
        # We expect a lot of division by zero, and that is fine.
        with np.errstate(divide='ignore'):
            check(self, k, method, data)

    return fct


kde_tester_args = 'name,index,data,method,with_adjust,with_weights,method_name'


def make_name(method):
    if isinstance(method, test_method):
        return method.instance.name
    return ":".join(m.instance.name for m in method)


def generate_methods_data(parameter_names, indices=None):
    """
    Generate the set of parameters needed to create tests for all these types
    of distributions.
    """
    global parameters
    result = []
    for name in parameter_names:
        params = knownTestSets[name]
        if indices is None:
            local_indices = range(len(params.sizes))
        else:
            try:
                local_indices = indices[name]
            except (KeyError, TypeError):
                local_indices = indices
        adjusts = [False, True] if params.can_adjust else [False]
        result += [(name, index, params.dataset, method, with_adjust,
                    with_weights, make_name(method)) for index in local_indices
                   for method in params.methods for with_adjust in adjusts
                   for with_weights in [False, True]]
    return result
