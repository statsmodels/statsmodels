"""Compatibility functions for numpy versions in lib

np.unique
---------
Behavior changed in 1.6.2 and doesn't work for structured arrays if
return_index=True.
Only needed for this case, use np.unique otherwise


License:

np_unique below is copied form the numpy source before the change and is
distributed under the BSD-3 license

Copyright (c) 2005-2009, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""


import numpy as np

try:
    from scipy.lib.version import NumpyVersion
except ImportError:
    import re
    from statsmodels.compat import string_types

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

if NumpyVersion(np.__version__) < '1.6.2':
    npc_unique = np.unique
else:

    def npc_unique(ar, return_index=False, return_inverse=False):
        """
        Find the unique elements of an array.

        Returns the sorted unique elements of an array. There are two optional
        outputs in addition to the unique elements: the indices of the input array
        that give the unique values, and the indices of the unique array that
        reconstruct the input array.

        Parameters
        ----------
        ar : array_like
            Input array. This will be flattened if it is not already 1-D.
        return_index : bool, optional
            If True, also return the indices of `ar` that result in the unique
            array.
        return_inverse : bool, optional
            If True, also return the indices of the unique array that can be used
            to reconstruct `ar`.

        Returns
        -------
        unique : ndarray
            The sorted unique values.
        unique_indices : ndarray, optional
            The indices of the unique values in the (flattened) original array.
            Only provided if `return_index` is True.
        unique_inverse : ndarray, optional
            The indices to reconstruct the (flattened) original array from the
            unique array. Only provided if `return_inverse` is True.

        See Also
        --------
        numpy.lib.arraysetops : Module with a number of other functions for
                                performing set operations on arrays.

        Examples
        --------
        >>> np.unique([1, 1, 2, 2, 3, 3])
        array([1, 2, 3])
        >>> a = np.array([[1, 1], [2, 3]])
        >>> np.unique(a)
        array([1, 2, 3])

        Return the indices of the original array that give the unique values:

        >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
        >>> u, indices = np.unique(a, return_index=True)
        >>> u
        array(['a', 'b', 'c'],
               dtype='|S1')
        >>> indices
        array([0, 1, 3])
        >>> a[indices]
        array(['a', 'b', 'c'],
               dtype='|S1')

        Reconstruct the input array from the unique values:

        >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
        >>> u, indices = np.unique(a, return_inverse=True)
        >>> u
        array([1, 2, 3, 4, 6])
        >>> indices
        array([0, 1, 4, 3, 1, 2, 1])
        >>> u[indices]
        array([1, 2, 6, 4, 2, 3, 2])

        """
        try:
            ar = ar.flatten()
        except AttributeError:
            if not return_inverse and not return_index:
                items = sorted(set(ar))
                return np.asarray(items)
            else:
                ar = np.asanyarray(ar).flatten()

        if ar.size == 0:
            if return_inverse and return_index:
                return ar, np.empty(0, np.bool), np.empty(0, np.bool)
            elif return_inverse or return_index:
                return ar, np.empty(0, np.bool)
            else:
                return ar

        if return_inverse or return_index:
            perm = ar.argsort()
            aux = ar[perm]
            flag = np.concatenate(([True], aux[1:] != aux[:-1]))
            if return_inverse:
                iflag = np.cumsum(flag) - 1
                iperm = perm.argsort()
                if return_index:
                    return aux[flag], perm[flag], iflag[iperm]
                else:
                    return aux[flag], iflag[iperm]
            else:
                return aux[flag], perm[flag]

        else:
            ar.sort()
            flag = np.concatenate(([True], ar[1:] != ar[:-1]))
            return ar[flag]

if NumpyVersion(np.__version__) >= '1.7.1':
    np_matrix_rank = np.linalg.matrix_rank
else:
    def np_matrix_rank(M, tol=None):
        """
        Return matrix rank of array using SVD method

        Rank of the array is the number of SVD singular values of the array that are
        greater than `tol`.

        Parameters
        ----------
        M : {(M,), (M, N)} array_like
            array of <=2 dimensions
        tol : {None, float}, optional
        threshold below which SVD values are considered zero. If `tol` is
        None, and ``S`` is an array with singular values for `M`, and
        ``eps`` is the epsilon value for datatype of ``S``, then `tol` is
        set to ``S.max() * max(M.shape) * eps``.

        Notes
        -----
        The default threshold to detect rank deficiency is a test on the magnitude
        of the singular values of `M`.  By default, we identify singular values less
        than ``S.max() * max(M.shape) * eps`` as indicating rank deficiency (with
        the symbols defined above). This is the algorithm MATLAB uses [1].  It also
        appears in *Numerical recipes* in the discussion of SVD solutions for linear
        least squares [2].

        This default threshold is designed to detect rank deficiency accounting for
        the numerical errors of the SVD computation.  Imagine that there is a column
        in `M` that is an exact (in floating point) linear combination of other
        columns in `M`. Computing the SVD on `M` will not produce a singular value
        exactly equal to 0 in general: any difference of the smallest SVD value from
        0 will be caused by numerical imprecision in the calculation of the SVD.
        Our threshold for small SVD values takes this numerical imprecision into
        account, and the default threshold will detect such numerical rank
        deficiency.  The threshold may declare a matrix `M` rank deficient even if
        the linear combination of some columns of `M` is not exactly equal to
        another column of `M` but only numerically very close to another column of
        `M`.

        We chose our default threshold because it is in wide use.  Other thresholds
        are possible.  For example, elsewhere in the 2007 edition of *Numerical
        recipes* there is an alternative threshold of ``S.max() *
        np.finfo(M.dtype).eps / 2. * np.sqrt(m + n + 1.)``. The authors describe
        this threshold as being based on "expected roundoff error" (p 71).

        The thresholds above deal with floating point roundoff error in the
        calculation of the SVD.  However, you may have more information about the
        sources of error in `M` that would make you consider other tolerance values
        to detect *effective* rank deficiency.  The most useful measure of the
        tolerance depends on the operations you intend to use on your matrix.  For
        example, if your data come from uncertain measurements with uncertainties
        greater than floating point epsilon, choosing a tolerance near that
        uncertainty may be preferable.  The tolerance may be absolute if the
        uncertainties are absolute rather than relative.

        References
        ----------
        .. [1] MATLAB reference documention, "Rank"
            http://www.mathworks.com/help/techdoc/ref/rank.html
        .. [2] W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery,
            "Numerical Recipes (3rd edition)", Cambridge University Press, 2007,
            page 795.

        Examples
        --------
        >>> from numpy.linalg import matrix_rank
        >>> matrix_rank(np.eye(4)) # Full rank matrix
        4
        >>> I=np.eye(4); I[-1,-1] = 0. # rank deficient matrix
        >>> matrix_rank(I)
        3
        >>> matrix_rank(np.ones((4,))) # 1 dimension - rank 1 unless all 0
        1
        >>> matrix_rank(np.zeros((4,)))
        0
        """
        M = np.asarray(M)
        if M.ndim > 2:
            raise TypeError('array should have 2 or fewer dimensions')
        if M.ndim < 2:
            return int(not all(M == 0))
        S = np.linalg.svd(M, compute_uv=False)
        if tol is None:
            tol = S.max() * max(M.shape) * np.finfo(S.dtype).eps
        return np.sum(S > tol)
