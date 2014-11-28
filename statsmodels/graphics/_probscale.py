import numpy as np
import matplotlib
from matplotlib.transforms import Transform
from matplotlib.scale import ScaleBase
from matplotlib.ticker import (
    FixedLocator,
    NullLocator,
    Formatter,
    NullFormatter,
    FuncFormatter
)
import matplotlib.pyplot as plt
from scipy import stats


def _get_probs(nobs):
    '''Returns the x-axis labels for a probability plot based
    on the number of observations (`nobs`)
    '''
    order = int(np.floor(np.log10(nobs)))
    base_probs = np.array([10., 20., 30., 40., 50., 60., 70., 80., 90.])

    axis_probs = base_probs.copy()
    for n in range(order):
        if n <= 2:
            lower_fringe = np.array([1., 2., 5.])
            upper_fringe = np.array([5., 8., 9.])
        else:
            lower_fringe = np.array([1.])
            upper_fringe = np.array([9.])

        new_lower = lower_fringe/10.**(n)
        new_upper = upper_fringe/10.**(n) + axis_probs.max()
        axis_probs = np.hstack([new_lower, axis_probs, new_upper])

    return axis_probs


def _sig_figs(val, nsf, expthresh=5):
    '''Formats a number into a string with the correct number of sig figs.

    Parameters
    -------s---
    val : numeric
        the number you want to round
    nsf : int
        the number of sig figs it should have

    Example
    -------
    >>> print(_sig_figs(1247.15, 3))
    1250
    >>> print(_sig_figs(1247.15, 7))
    1247.150

    '''
    # check on the number provided
    if not np.isfinite(val):
        raise ValueError("`val` must be a finite number")

    else:

        # check on the _sig_figs
        if nsf < 1:
            raise ValueError("number of sig figs must be greater than zero!")

        # logic to do all of the rounding
        elif val != 0.0:
            order = np.floor(np.log10(np.abs(val)))

            if -1.0 * expthresh <= order <= expthresh:
                decimal_places = int(nsf - 1 - order)

                if decimal_places <= 0:
                    out = '{0:,.0f}'.format(round(val, decimal_places))

                else:
                    fmt = '{0:,.%df}' % decimal_places
                    out = fmt.format(val)

            else:
                decimal_places = nsf - 1
                fmt = '{0:.%de}' % decimal_places
                out = fmt.format(val)

        else:
            out = str(round(val, nsf))


    return out


class ProbFormatter(Formatter):
    def __call__(self, x, pos=None):
        if x < 10:
            out = _sig_figs(x, 1)
        elif x <= 99:
            out =  _sig_figs(x, 2)
        else:
            order = np.ceil(np.round(np.abs(np.log10(100 - x)), 6))
            out = _sig_figs(x, order + 2)

        return '{}'.format(out)


class ProbTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, dist):
        Transform.__init__(self)
        self.dist = dist

    def transform_non_affine(self, a):
        return self.dist.ppf(a / 100.)

    def inverted(self):
        return InvertedProbTransform(self.dist)


class InvertedProbTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, dist):
        self.dist = dist
        Transform.__init__(self)

    def transform_non_affine(self, a):
        return self.dist.cdf(a) * 100.

    def inverted(self):
        return ProbTransform(self.dist)


class ProbScale(ScaleBase):
    """
    A probability scale.  Care is taken so non-positive
    values are not plotted.

    For computational efficiency (to push as much as possible to Numpy
    C code in the common cases), this scale provides different
    transforms depending on the base of the logarithm:

    """
    name = 'prob'

    def __init__(self, axis, **kwargs):
        self.dist = kwargs.pop('dist', stats.norm)
        self._transform = ProbTransform(self.dist)

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to specialized versions for
        log scaling.
        """
        axis.set_major_locator(FixedLocator(_get_probs(1e10)))
        axis.set_major_formatter(FuncFormatter(ProbFormatter()))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())

    def get_transform(self):
        """
        Return a :class:`~matplotlib.transforms.Transform` instance
        appropriate for the given logarithm base.
        """
        return self._transform

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Limit the domain to positive values.
        """
        return (vmin <= 0.0 and minpos or vmin,
                vmax <= 0.0 and minpos or vmax)
