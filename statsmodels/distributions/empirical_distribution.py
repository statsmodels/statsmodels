"""
Empirical CDF Functions
"""
import numpy as np
from scipy.interpolate import interp1d

def _conf_set(F, alpha=.05):
    r"""
    Constructs a Dvoretzky-Kiefer-Wolfowitz confidence band for the eCDF.

    Parameters
    ----------
    F : array-like
        The empirical distributions
    alpha : float
        Set alpha for a (1 - alpha) % confidence band.

    Notes
    -----
    Based on the DKW inequality.

    .. math:: P \left( \sup_x \left| F(x) - \hat(F)_n(X) \right| > \epsilon \right) \leq 2e^{-2n\epsilon^2}

    References
    ----------
    Wasserman, L. 2006. `All of Nonparametric Statistics`. Springer.
    """
    nobs = len(F)
    epsilon = np.sqrt(np.log(2./alpha) / (2 * nobs))
    lower = np.clip(F - epsilon, 0, 1)
    upper = np.clip(F + epsilon, 0, 1)
    return lower, upper

class StepFunction(object):
    """
    A basic step function.

    Values at the ends are handled in the simplest way possible:
    everything to the left of x[0] is set to ival; everything
    to the right of x[-1] is set to y[-1].

    Parameters
    ----------
    x : array-like
    y : array-like
    ival : float
        ival is the value given to the values to the left of x[0]. Default
        is 0.
    sorted : bool
        Default is False.
    side : {'left', 'right'}, optional
        Default is 'left'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.distributions.empirical_distribution import StepFunction
    >>>
    >>> x = np.arange(20)
    >>> y = np.arange(20)
    >>> f = StepFunction(x, y)
    >>>
    >>> print(f(3.2))
    3.0
    >>> print(f([[3.2,4.5],[24,-3.1]]))
    [[  3.   4.]
     [ 19.   0.]]
    >>> f2 = StepFunction(x, y, side='right')
    >>>
    >>> print(f(3.0))
    2.0
    >>> print(f2(3.0))
    3.0
    """

    def __init__(self, x, y, ival=0., sorted=False, side='left'):

        if side.lower() not in ['right', 'left']:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side

        _x = np.asarray(x)
        _y = np.asarray(y)

        if _x.shape != _y.shape:
            msg = "x and y do not have the same shape"
            raise ValueError(msg)
        if len(_x.shape) != 1:
            msg = 'x and y must be 1-dimensional'
            raise ValueError(msg)

        self.x = np.r_[-np.inf, _x]
        self.y = np.r_[ival, _y]

        if not sorted:
            asort = np.argsort(self.x)
            self.x = np.take(self.x, asort, 0)
            self.y = np.take(self.y, asort, 0)
        self.n = self.x.shape[0]

    def __call__(self, time):

        tind = np.searchsorted(self.x, time, self.side) - 1
        return self.y[tind]

class ECDF(StepFunction):
    """
    Return the Empirical CDF of an array as a step function.

    Parameters
    ----------
    x : array-like
        Observations
    side : {'left', 'right'}, optional
        Default is 'right'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Returns
    -------
    Empirical CDF as a step function.

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.distributions.empirical_distribution import ECDF
    >>>
    >>> ecdf = ECDF([3, 3, 1, 4])
    >>>
    >>> ecdf([3, 55, 0.5, 1.5])
    array([ 0.75,  1.  ,  0.  ,  0.25])
    """
    def __init__(self, x, side='right'):
        x = np.array(x, copy=True)
        x.sort()
        nobs = len(x)
        y = np.linspace(1./nobs,1,nobs)
        super(ECDF, self).__init__(x, y, side=side, sorted=True)
        # TODO: make `step` an arg and have a linear interpolation option?
        # This is the path with `step` is True
        # If `step` is False, a previous version of the code read
        #  `return interp1d(x,y,drop_errors=False,fill_values=ival)`
        # which would have raised a NameError if hit, so would need to be
        # fixed.  See GH#5701.


def monotone_fn_inverter(fn, x, vectorized=True, **keywords):
    """
    Given a monotone function fn (no checking is done to verify monotonicity)
    and a set of x values, return an linearly interpolated approximation
    to its inverse from its values on x.
    """
    x = np.asarray(x)
    if vectorized:
        y = fn(x, **keywords)
    else:
        y = []
        for _x in x:
            y.append(fn(_x, **keywords))
        y = np.array(y)

    a = np.argsort(y)

    return interp1d(y[a], x[a])

if __name__ == "__main__":
    #TODO: Make sure everything is correctly aligned and make a plotting
    # function
    from statsmodels.compat.python import urlopen
    import matplotlib.pyplot as plt
    nerve_data = urlopen('http://www.statsci.org/data/general/nerve.txt')
    nerve_data = np.loadtxt(nerve_data)
    x = nerve_data / 50. # was in 1/50 seconds
    cdf = ECDF(x)
    x.sort()
    F = cdf(x)
    plt.step(x, F, where='post')
    lower, upper = _conf_set(F)
    plt.step(x, lower, 'r', where='post')
    plt.step(x, upper, 'r', where='post')
    plt.xlim(0, 1.5)
    plt.ylim(0, 1.05)
    plt.vlines(x, 0, .05)
    plt.show()
