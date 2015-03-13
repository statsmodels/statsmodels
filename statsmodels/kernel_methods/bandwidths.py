"""
Module containing the methods for continuous, univariate, KDE estimations.

:Author: Barbier de Reuille, Pierre

"""
from __future__ import division, absolute_import, print_function
import numpy as np
from scipy import fftpack, optimize, linalg
from .kde_utils import large_float, finite, atleast_2df, AxesType
from statsmodels.compat.python import range

def _spread(X):
    """
    Returns the smaller of std(X, ddof=1) or normalized IQR(X) over axis 0.

    References
    ----------
    Silverman (1986) p.47
    """
    Q1, Q3 = np.percentile(X, [25, 75], axis=0)
    IQR = (Q3 - Q1) / 1.349
    return np.minimum(np.std(X, axis=0, ddof=1), IQR)

def full_variance(factor, exog):
    r"""
    Returns the bandwidth matrix:

    .. math::

        \mathcal{C} = \tau cov(X)^{1/2}

    where :math:`\tau` is a correcting factor that depends on the method.
    """
    d = exog.shape[1]
    if d == 1:
        spread = _spread(exog)
    else:
        spread = np.atleast_2d(linalg.sqrtm(np.cov(exog, rowvar=0, bias=False)))
    return spread * factor

def diagonal_variance(factor, exog):
    r"""
    Return the diagonal covariance matrix according to Silverman's rule
    """
    return _spread(exog) * factor

def silverman(model):
    r"""
    The Silverman bandwidth is defined as a variance bandwidth with factor:

    .. math::

        \tau = \left( n \frac{d+2}{4} \right)^\frac{-1}{d+4}
    """
    exog = atleast_2df(model.exog)
    n, d = exog.shape
    return diagonal_variance(0.9 * (n ** (-1. / (d + 4.))), exog)


def scotts(model):
    r"""
    The Scotts bandwidth is defined as a variance bandwidth with factor:

    .. math::

        \tau = n^\frac{-1}{d+4}
    """
    exog = atleast_2df(model.exog)
    n, d = exog.shape
    return diagonal_variance((n * (d + 2.) / 4.) ** (-1. / (d + 4.)), exog)

def scotts_full(model):
    """
    Scotts bandwidths, based on covariance only, and returning a full matrix
    """
    exog = atleast_2df(model.exog)
    n, d = exog.shape
    return full_variance((n * (d + 2.) / 4.) ** (-1. / (d + 4.)), exog)

def silverman_full(model):
    """
    Silverman bandwidths, based on covariance only, and returning a full matrix
    """
    exog = atleast_2df(model.exog)
    n, d = exog.shape
    return full_variance(0.9 * (n ** (-1. / (d + 4.))), exog)

def _botev_fixed_point(t, M, I, a2):
    l = 7
    I = large_float(I)
    M = large_float(M)
    a2 = large_float(a2)
    f = 2 * np.pi ** (2 * l) * np.sum(I ** l * a2 *
                                      np.exp(-I * np.pi ** 2 * t))
    for s in range(l, 1, -1):
        K0 = np.prod(np.arange(1, 2 * s, 2)) / np.sqrt(2 * np.pi)
        const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
        time = (2 * const * K0 / M / f) ** (2 / (3 + 2 * s))
        f = 2 * np.pi ** (2 * s) * \
            np.sum(I ** s * a2 * np.exp(-I * np.pi ** 2 * time))
    return t - (2 * M * np.sqrt(np.pi) * f) ** (-2 / 5)


class botev(object):
    """
    Implementation of the KDE bandwidth selection method outline in:

    Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
    estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

    Based on the implementation of Daniel B. Smith, PhD.

    The object is a callable returning the bandwidth for a 1D kernel.
    """
    def __init__(self, N=None, **kword):
        if 'lower' in kword or 'upper' in kword:
            print("Warning, using 'lower' and 'upper' for botev bandwidth is "
                  "deprecated. Argument is ignored")
        self.N = N

    def __call__(self, model):
        """
        Returns the optimal bandwidth based on the data
        """
        data = model.exog
        N = 2 ** 10 if self.N is None else int(2 ** np.ceil(np.log2(self.N)))
        lower = getattr(model, 'lower', None)
        upper = getattr(model, 'upper', None)
        if not finite(lower) or not finite(upper):
            minimum = np.min(data)
            maximum = np.max(data)
            span = maximum - minimum
            lower = minimum - span / 10 if not finite(lower) else lower
            upper = maximum + span / 10 if not finite(upper) else upper
        # Range of the data
        span = upper - lower

        # Histogram of the data to get a crude approximation of the density
        weights = model.weights
        if not weights.shape:
            weights = None
        M = len(data)
        DataHist, bins = np.histogram(data, bins=N, range=(lower, upper), weights=weights)
        DataHist = DataHist / M
        DCTData = fftpack.dct(DataHist, norm=None)

        I = np.arange(1, N, dtype=int) ** 2
        SqDCTData = (DCTData[1:] / 2) ** 2
        guess = 0.1

        try:
            t_star = optimize.brentq(_botev_fixed_point, 0, guess,
                                     args=(M, I, SqDCTData))
        except ValueError:
            t_star = .28 * N ** (-.4)

        return np.sqrt(t_star) * span

class KDE1DAdaptor(object):
    """
    Adaptor class to view a nD KDE estimator as a 1D estimator for a given dimension
    """
    def __init__(self, kde, axis=None):
        self._axis = axis
        self._kde = kde

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, val):
        val = int(val)
        if val < 0 or val >= self._kde.ndim:
            raise ValueError("Error, invalid axis")
        self._axis = val

    @property
    def ndim(self):
        return 1

    def fit(self):
        raise NotImplementedError()

    @property
    def exog(self):
        return self._kde.exog[..., self._axis]

    _list_attributes = ['lower', 'upper', 'axis_type', 'kernel', 'bandwidth']

    _constant_attributes = ['weights', 'adjust', 'total_weights', 'npts']

def _add_fwd_list_attr(cls, attr):
    def getter(self):
        value = getattr(self._kde, attr)
        try:
            return getattr(self._kde, attr)[self._axis]
        except:
            return value
    setattr(cls, attr, property(getter))

def _add_fwd_attr(cls, attr):
    def getter(self):
        return getattr(self._kde, attr)
    setattr(cls, attr, property(getter))

for attr in KDE1DAdaptor._list_attributes:
    _add_fwd_list_attr(KDE1DAdaptor, attr)

for attr in KDE1DAdaptor._constant_attributes:
    _add_fwd_attr(KDE1DAdaptor, attr)


class Multivariate(object):
    def __init__(self):
        self.continuous = scotts
        self.ordered = 0.1
        self.unordered = 0.1

    def __call__(self, model):
        res = np.zeros(model.ndim, dtype=float)
        if len(model.axis_type) == 1:
            axis_type = AxesType(model.axis_type[0] * model.ndim)
        else:
            axis_type = AxesType(model.axis_type)
        c = np.nonzero(axis_type == 'C')[0]
        o = np.nonzero(axis_type == 'O')[0]
        u = np.nonzero(axis_type == 'U')[0]
        adapt = KDE1DAdaptor(model)
        if callable(self.continuous):
            for d in c:
                adapt.axis = d
                res[d] = self.continuous(adapt)
        else:
            res[c] = self.continuous
        if callable(self.ordered):
            for d in c:
                adapt.axis = d
                res[d] = self.ordered(adapt)
        else:
            res[o] = self.ordered
        if callable(self.unordered):
            for d in c:
                adapt.axis = d
                res[d] = self.unordered(adapt)
        else:
            res[u] = self.unordered
        return res

from .bw_crossvalidation import lsq_crossvalidation, ContinuousIMSE  # NoQA
