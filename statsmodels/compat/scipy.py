from __future__ import absolute_import
import numpy as np


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


# Work around for complex chnges in gammaln in 1.0.0.  loggamma introduced in 0.18.
try:
    from scipy.special import loggamma
except ImportError:
    from scipy.special import gammaln
    loggamma = gammaln
