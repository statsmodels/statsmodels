from __future__ import absolute_import
import numpy as np
import scipy

def _bit_length_26(x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return len(bin(x)) - 2


try:
    from scipy.lib._version import NumpyVersion
except ImportError:
    import re
    from .python import string_types

    class NumpyVersion():
        """Parse and compare numpy version strings.

        Numpy has the following versioning scheme (numbers given are examples; they
        can be >9) in principle):

        - Released version: '1.8.0', '1.8.1', etc.
        - Alpha: '1.8.0a1', '1.8.0a2', etc.
        - Beta: '1.8.0b1', '1.8.0b2', etc.
        - Release candidates: '1.8.0rc1', '1.8.0rc2', etc.
        - Development versions: '1.8.0.dev-f1234afa' (git commit hash appended)
        - Development versions after a1: '1.8.0a1.dev-f1234afa',
                                        '1.8.0b2.dev-f1234afa',
                                        '1.8.1rc1.dev-f1234afa', etc.
        - Development versions (no git hash available): '1.8.0.dev-Unknown'

        Comparing needs to be done against a valid version string or other
        `NumpyVersion` instance.

        Parameters
        ----------
        vstring : str
            Numpy version string (``np.__version__``).

        Notes
        -----
        All dev versions of the same (pre-)release compare equal.

        Examples
        --------
        >>> from scipy.lib._version import NumpyVersion
        >>> if NumpyVersion(np.__version__) < '1.7.0':
        ...     print('skip')
        skip

        >>> NumpyVersion('1.7')  # raises ValueError, add ".0"

        """

        def __init__(self, vstring):
            self.vstring = vstring
            ver_main = re.match(r'\d[.]\d+[.]\d+', vstring)
            if not ver_main:
                raise ValueError("Not a valid numpy version string")

            self.version = ver_main.group()
            self.major, self.minor, self.bugfix = [int(x) for x in
                                                   self.version.split('.')]
            if len(vstring) == ver_main.end():
                self.pre_release = 'final'
            else:
                alpha = re.match(r'a\d', vstring[ver_main.end():])
                beta = re.match(r'b\d', vstring[ver_main.end():])
                rc = re.match(r'rc\d', vstring[ver_main.end():])
                pre_rel = [m for m in [alpha, beta, rc] if m is not None]
                if pre_rel:
                    self.pre_release = pre_rel[0].group()
                else:
                    self.pre_release = ''

            self.is_devversion = bool(re.search(r'.dev-', vstring))

        def _compare_version(self, other):
            """Compare major.minor.bugfix"""
            if self.major == other.major:
                if self.minor == other.minor:
                    if self.bugfix == other.bugfix:
                        vercmp = 0
                    elif self.bugfix > other.bugfix:
                        vercmp = 1
                    else:
                        vercmp = -1
                elif self.minor > other.minor:
                    vercmp = 1
                else:
                    vercmp = -1
            elif self.major > other.major:
                vercmp = 1
            else:
                vercmp = -1

            return vercmp

        def _compare_pre_release(self, other):
            """Compare alpha/beta/rc/final."""
            if self.pre_release == other.pre_release:
                vercmp = 0
            elif self.pre_release == 'final':
                vercmp = 1
            elif other.pre_release == 'final':
                vercmp = -1
            elif self.pre_release > other.pre_release:
                vercmp = 1
            else:
                vercmp = -1

            return vercmp

        def _compare(self, other):
            if not isinstance(other, (string_types, NumpyVersion)):
                raise ValueError("Invalid object to compare with NumpyVersion.")

            if isinstance(other, string_types):
                other = NumpyVersion(other)

            vercmp = self._compare_version(other)
            if vercmp == 0:
                # Same x.y.z version, check for alpha/beta/rc
                vercmp = self._compare_pre_release(other)
                if vercmp == 0:
                    # Same version and same pre-release, check if dev version
                    if self.is_devversion is other.is_devversion:
                        vercmp = 0
                    elif self.is_devversion:
                        vercmp = -1
                    else:
                        vercmp = 1

            return vercmp

        def __lt__(self, other):
            return self._compare(other) < 0

        def __le__(self, other):
            return self._compare(other) <= 0

        def __eq__(self, other):
            return self._compare(other) == 0

        def __ne__(self, other):
            return self._compare(other) != 0

        def __gt__(self, other):
            return self._compare(other) > 0

        def __ge__(self, other):
            return self._compare(other) >= 0

        def __repr(self):
            return "NumpyVersion(%s)" % self.vstring


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
            try:
                p2 = 2 ** ((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2 ** _bit_length_26(quotient - 1)

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

from scipy import integrate
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
