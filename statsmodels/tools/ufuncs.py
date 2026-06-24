import numpy as np
from math import exp


def _s_exp(x, b):
    """
    Overflow-protected exp

    Parameters
    ----------
    x : float
        Input value
    b : float
        Must be positive (not checked)

    Returns
    -------
    y : float
        Protected exponential of x

    Notes
    -----
    If y = exp(x) if -b <= x <= b. For values outside of this range, if x > b,
    then y = exp(b) * (1 + (x-b) + (x-b)**2/2) which is a second order Taylor
    expansion.  If x < -b, then y = 1 / s_exp(-x, b) which is the Taylor
    expansion for x > b inverted.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-3, 3, 100, endpoint=True)
    >>> y = s_exp(x, 2)
    >>> plt.plot(x, y, x, np.exp(x))
    >>> plt.show()

    >>> x = np.linspace(-5, 5, 100, endpoint=True)
    >>> y = s_exp(x, 2)
    >>> plt.plot(x, np.log(y), x, x)
    >>> plt.show()
    """
    if x < -b:
        d = abs(x) - b
        return 1 / (exp(b) * (1 + d + d ** 2 / 2))
    elif x > b:
        d = x - b
        return exp(b) * (1 + d + d ** 2 / 2)
    else:
        return exp(x)


def _o_exp(x, b, c=None):
    """
    Periodic, Overflow-protected exp

    Parameters
    ----------
    x : float
    b : float
    c : float

    Returns
    -------
    y : float
        Smoothed exponential of x

    Notes
    -----
    y = c * exp(x * b)

    This function is accurate until around (+/-) 2/b.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-50, 50, 100, endpoint=True)
    >>> b= 1 / (40 / (pi / 2))
    >>> y = o_exp(x, b)
    >>> plt.plot(x, y, x, exp(x))
    >>> plt.gca().set_ylim(0, 1.1 * y.max())
    >>> plt.show()

    >>> plt.plot(x, np.log(y), x, x)
    >>> plt.show()
    """
    c = 1. / b if c is None else c
    return exp(c * np.sin(x * b))


s_exp = np.vectorize(_s_exp, otypes=[np.double])
o_exp = np.vectorize(_o_exp, otypes=[np.double])
