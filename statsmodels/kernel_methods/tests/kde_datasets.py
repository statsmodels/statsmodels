import numpy as np
from collections import namedtuple
from .. import kde_1d, kde_nd, kde_nc, kde_multivariate
from scipy import stats
from .. import kernelsnd, kernels1d, kernelsnc, kde


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
    test_method(kde_1d.KDE1DMethod, 1e-5, 1e-4, 1e-5, False, False),
    test_method(kde_1d.Reflection1D, 1e-5, 1e-4, 1e-5, True, True),
    test_method(kde_1d.Cyclic1D, 1e-5, 1e-3, 1e-4, True, True),
    test_method(kde_1d.Renormalization, 1e-5, 1e-4, 1e-2, True, True),
    test_method(kde_1d.LinearCombination, 1e-1, 1e-1, 1e-1, True, False)
]

methods_log = [
    test_method(kde_1d.Transform1D(kde_1d.LogTransform), 1e-5, 1e-4, 1e-5,
                True, False)
]

methods_nd = [
    test_method(kde_nd.Cyclic, 1e-5, 1e-4, 1e-5, True, True),
    test_method(kde_nd.Cyclic, 1e-5, 1e-4, 1e-5, False, False),
    test_method(kde_nd.KDEnDMethod, 1e-5, 1e-4, 1e-5, False, False)
]

methods_nc = [
    test_method(kde_nc.Ordered, 1e-5, 1e-4, 1e-5, False, False),
    test_method(kde_nc.Unordered, 1e-5, 1e-4, 1e-5, False, False)
]

test_kernel = namedtuple('test_kernel',
                         ['cls', 'precision_factor', 'var', 'positive'])

kernels_1d = [
    test_kernel(kernels1d.Gaussian1D, 1, 1, True),
    test_kernel(kernels1d.TriCube, 1, 1, True),
    test_kernel(kernels1d.Epanechnikov, 10, 1, True),
    test_kernel(kernels1d.GaussianOrder4, 10, 0,
                False),  # Bad for precision because of high frequencies
    test_kernel(kernels1d.EpanechnikovOrder4, 1000, 0, False)
]  # Bad for precision because of high frequencies

kernels_nc = [
    test_kernel(kernelsnc.AitchisonAitken, 1, 1, True),
    test_kernel(kernelsnc.WangRyzin, 1, 1, True)
]

kernels_nd = [test_kernel(kernelsnd.Gaussian, 1, 1, True)]

# Tuple to store datasets
# * method - Method to use for the KDE
# * xs - Points on which the KDE can be safely evaluated
# * exog - The dataset itself
# * weights - A list of valid weights, None if no weights are to be used.
# * adjust - A list of valid adjustments, None if the method doesn't handle it.
# * lower - A lower bound for vs
# * upper - An upper bound for vs
dataset = namedtuple(
    'dataset',
    ['name', 'method', 'xs', 'exog', 'weights', 'adjust', 'lower', 'upper'])


def _boolean_selector(data):
    if data is None:
        return [False]
    return [False, True]


def _make_name(method, size, has_weights, has_adjust):
    if isinstance(method, test_method):
        method_name = method.instance.name
    else:
        method_name = ":".join(m.instance.name for m in method)
    return "{0}/{1}{2}w{3}a".format(method_name, size,
                                    "+" if has_weights else "-",
                                    "+" if has_adjust else "-")


def _make_dataset(methods, xs, exogs, weights, adjusts, lower, upper):
    return [
        dataset(_make_name(method, len(exogs[i]), has_weights, has_adjust),
                method, xs, exogs[i], weights[i] if has_weights else None,
                adjusts[i] if has_adjust else None, lower, upper)
        for method in methods for i in range(len(exogs))
        for has_weights in _boolean_selector(weights)
        for has_adjust in _boolean_selector(adjusts)
    ]


class DataSets(object):
    """Class grouping static methods generating known test sets."""
    @staticmethod
    def norm(sizes=(128, 256, 201)):
        """
        Create the parameters to test using a 1D Gaussian distribution dataset.
        """
        dist = stats.norm(0, 1)
        exogs = [generate(dist, s, -5, 5) for s in sizes]
        weights = [dist.pdf(v) for v in exogs]
        adjusts = [1 - ws for ws in weights]
        return _make_dataset(methods_1d,
                             np.r_[-5:5:512j],
                             exogs,
                             weights,
                             adjusts,
                             lower=-5,
                             upper=5)

    @staticmethod
    def lognorm(sizes=(128, 256, 201)):
        """
        Create the parameters to test using a 1D log normal distribution
        dataset.
        """
        dist = stats.lognorm(1)
        xs = np.r_[0:20:513j][1:]
        vs = [generate(dist, s, 0.001, 20) for s in sizes]
        vs = [v[v < 20] for v in vs]
        weights = [dist.pdf(v) for v in vs]
        adjusts = [1 - ws for ws in weights]
        return _make_dataset(methods_log,
                             xs,
                             vs,
                             weights,
                             adjusts,
                             lower=0,
                             upper=20)

    @staticmethod
    def normnd(ndim, sizes=(32, 64, 128)):
        """
        Create the parameters to test using a nD Gaussian distribution dataset.
        """
        dist = stats.multivariate_normal(cov=np.eye(ndim))
        xs = [np.r_[-5:5:512j]] * ndim
        vs = [generate_nd(dist, s) for s in sizes]
        weights = [dist.pdf(v) for v in vs]
        return _make_dataset(methods_nd,
                             xs,
                             vs,
                             weights,
                             adjusts=None,
                             lower=[-5] * ndim,
                             upper=[5] * ndim)

    @staticmethod
    def poissonnc(sizes=(128, 256, 201)):
        """
        Create the parameters to test using  a nC poisson distribution dataset.
        """
        dist = stats.poisson(12)
        vs = [generate_nc(dist, s) for s in sizes]
        weights = [dist.pmf(v) for v in vs]
        return _make_dataset(methods_nc,
                             None,
                             vs,
                             weights,
                             adjusts=None,
                             lower=None,
                             upper=None)

    @staticmethod
    def multivariate(sizes=(64, 128, 101)):
        """
        Create the parameters to test using a poisson distribution and two
        normals as dataset.
        """
        d1 = stats.norm(0, 3)
        d2 = stats.poisson(12)
        sizes = [64, 128, 101]
        m1d = methods_1d[:-1]  # Remove Linear combination
        methods1 = m1d + methods_nc + methods_nc
        methods2 = methods_nc + m1d + methods_nc[::-1]
        methods = zip(methods1, methods2)
        vs = [generate_multivariate(s, d1, d2) for s in sizes]
        weights = [d1.pdf(v[:, 0]) for v in vs]
        upper = [5, max(v[:, 1].max() for v in vs)]
        lower = [-5, 0]
        return _make_dataset(methods,
                             None,
                             vs,
                             weights,
                             adjusts=None,
                             lower=lower,
                             upper=upper)


def createKDE(data, **kde_kwargs):
    """
    Create a new KDE object.

    Arguments:
        data: An instance of `dataset`.
    """
    k = kde.KDE(data.exog, **kde_kwargs)
    if isinstance(data.method, test_method):
        if data.method.instance is None:
            del k.method
        else:
            k.method = data.method.instance
        if data.method.bound_low:
            k.lower = data.lower
        else:
            del k.lower
        if data.method.bound_high:
            k.upper = data.upper
        else:
            del k.upper
    else:
        mv = kde_multivariate.Multivariate()
        k.method = mv
        exog = data.exog[...]

        n = len(data.method)
        axis_type = ''
        lower = [-np.inf] * n
        upper = [np.inf] * n
        for i, m in enumerate(data.method):
            method_instance = m.instance()
            mv.methods[i] = method_instance
            axis_type += str(method_instance.axis_type)
            if method_instance.axis_type != 'C':
                exog[:, i] = np.round(exog[:, i])
            if m.bound_low:
                lower[i] = data.lower[i]
            if m.bound_high:
                upper[i] = data.upper[i]
        k.exog = exog
        k.axis_type = axis_type
        k.lower = lower
        k.upper = upper
    if data.weights is not None:
        k.weights = data.weights
    if data.adjust is not None:
        k.adjust = data.adjust
    return k
