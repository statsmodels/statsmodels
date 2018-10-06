"""
Module containing the methods to compute the bandwidth of the KDE.

:Author: Barbier de Reuille, Pierre
"""
from __future__ import division, absolute_import, print_function
import numpy as np
from scipy import fftpack, optimize, linalg
from .kde_utils import large_float, finite, atleast_2df, AxesType
from ..compat.python import range


def _spread(X):
    """
    Returns the smaller of std(X, ddof=1) or normalized IQR(X) over axis 0.

    References
    ----------
    [KM6] Silverman (1986) p.47
    """
    Q1, Q3 = np.percentile(X, [25, 75], axis=0)
    IQR = (Q3 - Q1) / 1.349
    return np.minimum(np.std(X, axis=0, ddof=1), IQR)


def full_variance(factor, exog):
    r"""
    Returns the bandwidth matrix:

    .. math::

        \mathcal{C} = \tau \left(\Sigma_X\right)^{1/2}

    where :math:`\tau` is a correcting factor and :math:`\Sigma_X` is the
    covariance matrix of X.
    """
    d = exog.shape[1]
    if d == 1:
        spread = _spread(exog)
    else:
        spread = np.atleast_2d(linalg.sqrtm(np.cov(exog, rowvar=0, bias=False)))
    return spread * factor


def diagonal_variance(factor, exog):
    r"""
    Return the diagonal covariance matrix according to Silverman's rule. The
    variance of each dimension is computed as:

    .. math::

        \mathcal{C} = \tau \min(\sigma_X, IQR(X) / 1.349)

    where :math:`\tau` is the correcting factor, :math:`\sigma_X` is the
    unbiased standard deviation of X and :math:`IQR(X)` is the intequartile
    range of X.
    """
    return _spread(exog) * factor


def silverman(model):
    r"""
    The Silverman bandwidth is defined as a variance bandwidth with factor:

    .. math::

        \tau = .9 n^{-\frac{1}{d+4}}

    where n is the number of points and d the dimension of the model.

    .. seealso::

        :py:func:`diagonal_variance`
        :py:func:`silverman_full`
    """
    exog = atleast_2df(model.exog)
    n, d = exog.shape
    return diagonal_variance(0.9 * (n ** (-1. / (d + 4.))), exog)


def silverman_full(model):
    r"""
    Silverman bandwidths, based on covariance only, and returning a full matrix

    .. math::

        \tau = .9 n^{-\frac{1}{d+4}}

    where n is the number of points and d the dimension of the model.

    .. seealso::

        :py:func:`full_variance`
        :py:func:`silverman`
    """
    exog = atleast_2df(model.exog)
    n, d = exog.shape
    return full_variance(0.9 * (n ** (-1. / (d + 4.))), exog)


def scotts(model):
    r"""
    The Scotts bandwidth is defined as a variance bandwidth with factor:

    .. math::

        \tau = \left( n \frac{d+2}{4} \right)^\frac{-1}{d+4}

    where n is the number of points and d the dimension of the model.

    .. seealso::

        :py:func:`diagonal_variance`
        :py:func:`scotts_full`
    """
    exog = atleast_2df(model.exog)
    n, d = exog.shape
    return diagonal_variance((n * (d + 2.) / 4.) ** (-1. / (d + 4.)), exog)


def scotts_full(model):
    r"""
    Scotts bandwidths, based on covariance only, and returning a full matrix

    .. math::

        \tau = \left( n \frac{d+2}{4} \right)^\frac{-1}{d+4}

    where n is the number of points and d the dimension of the model.

    .. seealso::

        :py:func:`full_variance`
        :py:func:`scotts`
    """
    exog = atleast_2df(model.exog)
    n, d = exog.shape
    return full_variance((n * (d + 2.) / 4.) ** (-1. / (d + 4.)), exog)


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
        if model.ndim != 1:
            raise ValueError('Botev bandwidth selection is only supported for '
                             '1D KDEs')
        data = model.exog.reshape((-1,))
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

    def copy(self):
        return KDE1DAdaptor(self._kde.copy(), self._axis)

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
    """
    Object computing the bandwidth for each axis of a multi-variate dataset.
    """
    def __init__(self):
        self._defaults = dict(c=scotts,
                              o=0.1,
                              u=0.1)
        self._bandwidths = {}

    @property
    def continuous(self):
        """
        Default bandwidth for a continuous axis (Default: :py:func:`scotts`)
        """
        return self._defaults['c']

    @continuous.setter
    def continuous(self, val):
        if not callable(val):
            self._defaults['c'] = float(val)
        else:
            self._defaults['c'] = val

    @property
    def ordered(self):
        """
        Default bandwidth for an ordered axis (Default: 0.1)
        """
        return self._defaults['o']

    @ordered.setter
    def ordered(self, val):
        if not callable(val):
            self._defaults['o'] = float(val)
        else:
            self._defaults['o'] = val

    @property
    def unordered(self):
        """
        Default bandwidth for an unordered axis (Default: 0.1)
        """
        return self._defaults['u']

    @unordered.setter
    def unordered(self, val):
        if not callable(val):
            self._defaults['u'] = float(val)
        else:
            self._defaults['u'] = val

    @property
    def bandwidths(self):
        """
        Dictionnary holding explicit methods or values for specific axes, by index.
        """
        return self._bandwidths

    def __call__(self, model):
        """
        Compute the bandwidths for all the axes of the model

        Parameters
        ----------
        model: object
            An object similar to a KDE, containing the axis types, exog data
            and other properties required by each method

        Returns
        -------
        ndarray of shape (D,)
            An array with one bandwidth per dimension
        """
        res = np.zeros(model.ndim, dtype=float)
        if len(model.axis_type) == 1:
            axis_type = AxesType(model.axis_type[0] * model.ndim)
        else:
            axis_type = AxesType(model.axis_type)
        bandwidths = self._bandwidths
        defaults = self._defaults
        adapt = KDE1DAdaptor(model)
        for d, axis in enumerate(axis_type):
            bw = bandwidths.get(d, None)
            if bw is None:
                bw = defaults[axis]
            if callable(bw):
                adapt.axis = d
                res[d] = bw(adapt)
            else:
                res[d] = float(bw)
        return res

from .bw_crossvalidation import crossvalidation, CVFunc, CV_IMSE, CV_LogLikelihood, leave_some_out  # NoQA
