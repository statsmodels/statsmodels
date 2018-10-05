from __future__ import absolute_import
import numpy as np
from scipy import integrate

NumpyVersion = np.lib.NumpyVersion


def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)
            # Quickly find next power of 2 >= quotient
            p2 = 2 ** ((quotient - 1).bit_length())

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


if NumpyVersion(scipy.__version__) < NumpyVersion('0.13.0'):
    from functools import partial

    def sp_integrate_nquad(func, ranges, args=None, opts=None):
        """
        Integration over multiple variables.

        Wraps `quad` to enable integration over multiple variables.
        Various options allow improved integration of discontinuous functions, as
        well as the use of weighted integration, and generally finer control of the
        integration process.

        Parameters
        ----------
        func : callable
            The function to be integrated. Has arguments of ``x0, ... xn``,
            ``t0, tm``, where integration is carried out over ``x0, ... xn``, which
            must be floats.  Function signature should be
            ``func(x0, x1, ..., xn, t0, t1, ..., tm)``.  Integration is carried out
            in order.  That is, integration over ``x0`` is the innermost integral,
            and ``xn`` is the outermost.
        ranges : iterable object
            Each element of ranges may be either a sequence  of 2 numbers, or else
            a callable that returns such a sequence.  ``ranges[0]`` corresponds to
            integration over x0, and so on.  If an element of ranges is a callable,
            then it will be called with all of the integration arguments available.
            e.g. if ``func = f(x0, x1, x2)``, then ``ranges[0]`` may be defined as
            either ``(a, b)`` or else as ``(a, b) = range0(x1, x2)``.
        args : iterable object, optional
            Additional arguments ``t0, ..., tn``, required by `func`.
        opts : iterable object or dict, optional
            Options to be passed to `quad`.  May be empty, a dict, or
            a sequence of dicts or functions that return a dict.  If empty, the
            default options from scipy.integrate.quadare used.  If a dict, the same
            options are used for all levels of integraion.  If a sequence, then each
            element of the sequence corresponds to a particular integration. e.g.
            opts[0] corresponds to integration over x0, and so on. The available
            options together with their default values are:

              - epsabs = 1.49e-08
              - epsrel = 1.49e-08
              - limit  = 50
              - points = None
              - weight = None
              - wvar   = None
              - wopts  = None

            The ``full_output`` option from `quad` is unavailable, due to the
            complexity of handling the large amount of data such an option would
            return for this kind of nested integration.  For more information on
            these options, see `quad` and `quad_explain`.

        Returns
        -------
        result : float
            The result of the integration.
        abserr : float
            The maximum of the estimates of the absolute error in the various
            integration results.

        See Also
        --------
        quad : 1-dimensional numerical integration
        dblquad, tplquad : double and triple integrals
        fixed_quad : fixed-order Gaussian quadrature
        quadrature : adaptive Gaussian quadrature

        Examples
        --------
        >>> from scipy import integrate
        >>> func = lambda x0,x1,x2,x3 : x0**2 + x1*x2 - x3**3 + np.sin(x0) + (
        ...                                 1 if (x0-.2*x3-.5-.25*x1>0) else 0)
        >>> points = [[lambda (x1,x2,x3) : 0.2*x3 + 0.5 + 0.25*x1], [], [], []]
        >>> def opts0(*args, **kwargs):
        ...     return {'points':[0.2*args[2] + 0.5 + 0.25*args[0]]}
        >>> integrate.nquad(func, [[0,1], [-1,1], [.13,.8], [-.15,1]],
        ...                 opts=[opts0,{},{},{}])
        (1.5267454070738633, 2.9437360001402324e-14)

        >>> scale = .1
        >>> def func2(x0, x1, x2, x3, t0, t1):
        ...     return x0*x1*x3**2 + np.sin(x2) + 1 + (1 if x0+t1*x1-t0>0 else 0)
        >>> def lim0(x1, x2, x3, t0, t1):
        ...     return [scale * (x1**2 + x2 + np.cos(x3)*t0*t1 + 1) - 1,
        ...             scale * (x1**2 + x2 + np.cos(x3)*t0*t1 + 1) + 1]
        >>> def lim1(x2, x3, t0, t1):
        ...     return [scale * (t0*x2 + t1*x3) - 1,
        ...             scale * (t0*x2 + t1*x3) + 1]
        >>> def lim2(x3, t0, t1):
        ...     return [scale * (x3 + t0**2*t1**3) - 1,
        ...             scale * (x3 + t0**2*t1**3) + 1]
        >>> def lim3(t0, t1):
        ...     return [scale * (t0+t1) - 1, scale * (t0+t1) + 1]
        >>> def opts0(x1, x2, x3, t0, t1):
        ...     return {'points' : [t0 - t1*x1]}
        >>> def opts1(x2, x3, t0, t1):
        ...     return {}
        >>> def opts2(x3, t0, t1):
        ...     return {}
        >>> def opts3(t0, t1):
        ...     return {}
        >>> integrate.nquad(func2, [lim0, lim1, lim2, lim3], args=(0,0),
                            opts=[opts0, opts1, opts2, opts3])
        (25.066666666666666, 2.7829590483937256e-13)

        """
        depth = len(ranges)
        ranges = [rng if callable(rng) else _sp_RangeFunc(rng) for rng in ranges]
        if args is None:
            args = ()
        if opts is None:
            opts = [dict([])] * depth

        if isinstance(opts, dict):
            opts = [opts] * depth
        else:
            opts = [opt if callable(opt) else _sp_OptFunc(opt) for opt in opts]

        return _sp_NQuad(func, ranges, opts).integrate(*args)

    class _sp_RangeFunc(object):
        def __init__(self, range_):
            self.range_ = range_

        def __call__(self, *args):
            """Return stored value.

            *args needed because range_ can be float or func, and is called with
            variable number of parameters.
            """
            return self.range_

    class _sp_OptFunc(object):
        def __init__(self, opt):
            self.opt = opt

        def __call__(self, *args):
            """Return stored dict."""
            return self.opt

    class _sp_NQuad(object):
        def __init__(self, func, ranges, opts):
            self.abserr = 0
            self.func = func
            self.ranges = ranges
            self.opts = opts
            self.maxdepth = len(ranges)

        def integrate(self, *args, **kwargs):
            depth = kwargs.pop('depth', 0)
            if kwargs:
                raise ValueError('unexpected kwargs')

            # Get the integration range and options for this depth.
            ind = -(depth + 1)
            fn_range = self.ranges[ind]
            low, high = fn_range(*args)
            fn_opt = self.opts[ind]
            opt = dict(fn_opt(*args))

            if 'points' in opt:
                opt['points'] = [x for x in opt['points'] if low <= x <= high]
            if depth + 1 == self.maxdepth:
                f = self.func
            else:
                f = partial(self.integrate, depth=depth+1)

            value, abserr = integrate.quad(f, low, high, args=args, **opt)
            self.abserr = max(self.abserr, abserr)
            if depth > 0:
                return value
            else:
                # Final result of n-D integration with error
                return value, self.abserr
else:
    sp_integrate_nquad = integrate.nquad


def _valarray(shape, value=np.nan, typecode=None):
    """Return an array of all value.
    """

    out = np.ones(shape, dtype=bool) * value
    if typecode is not None:
        out = out.astype(typecode)
    if not isinstance(out, np.ndarray):
        out = np.asarray(out)
    return out


def _lazywhere(cond, arrays, f, fillvalue=None, f2=None):
    """
    np.where(cond, x, fillvalue) always evaluates x even where cond is False.
    This one only evaluates f(arr1[cond], arr2[cond], ...).
    For example,
    >>> a, b = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])
    >>> def f(a, b):
        return a*b
    >>> _lazywhere(a > 2, (a, b), f, np.nan)
    array([ nan,  nan,  21.,  32.])
    Notice it assumes that all `arrays` are of the same shape, or can be
    broadcasted together.
    """
    if fillvalue is None:
        if f2 is None:
            raise ValueError("One of (fillvalue, f2) must be given.")
        else:
            fillvalue = np.nan
    else:
        if f2 is not None:
            raise ValueError("Only one of (fillvalue, f2) can be given.")

    arrays = np.broadcast_arrays(*arrays)
    temp = tuple(np.extract(cond, arr) for arr in arrays)
    tcode = np.mintypecode([a.dtype.char for a in arrays])
    out = _valarray(np.shape(arrays[0]), value=fillvalue, typecode=tcode)
    np.place(out, cond, f(*temp))
    if f2 is not None:
        temp = tuple(np.extract(~cond, arr) for arr in arrays)
        np.place(out, ~cond, f2(*temp))

    return out


# Work around for complex chnges in gammaln in 1.0.0.
#   loggamma introduced in 0.18.
try:
    from scipy.special import loggamma  # noqa:F401
except ImportError:
    from scipy.special import gammaln  # noqa:F401
    loggamma = gammaln

# Work around for factorial changes in 1.0.0

try:
    from scipy.special import factorial, factorial2  # noqa:F401
except ImportError:
    from scipy.misc import factorial, factorial2  # noqa:F401

# Moved in 1.0 to special
try:
    from scipy.special import logsumexp  # noqa:F401
except:
    from scipy.misc import logsumexp  # noqa:F401
