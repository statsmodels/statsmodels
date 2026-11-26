"""
Least squares engine and rank rule shared by the regression code.

Internal least squares problems should use the objects in this module so
that

* the numerical rank is decided by a single rule, `default_rcond`, and
* the parameter covariance is computed from the factorization of the design
  matrix and never from the cross-product ``x.T @ x``, which squares the
  condition number.

The engine is a thin QR factorization of the design matrix followed by the
singular value decomposition of the small triangular factor. It returns the
same minimum-norm solution as the pseudoinverse, including when the design
is rank deficient. Unlike a direct SVD of the design, the factorization is
cheap to reuse for several right-hand sides.
"""

from __future__ import annotations

import numpy as np

__all__ = ["QRSVD", "default_rcond", "numerical_rank"]


def default_rcond(nobs: int, ncols: int) -> float:
    """
    Relative cutoff below which singular values are treated as zero.

    Parameters
    ----------
    nobs : int
        The number of rows of the design matrix.
    ncols : int
        The number of columns of the design matrix.

    Returns
    -------
    float
        ``max(nobs, ncols) * eps``, the rule used by
        ``numpy.linalg.matrix_rank`` and by ``numpy.linalg.lstsq`` when
        ``rcond`` is None.

    Notes
    -----
    This is the only place that decides when a singular value is zero. The
    singular value of an exactly rank deficient design that should be zero is
    returned by the SVD as a small multiple of ``eps`` times the largest
    singular value. A cutoff that does not grow with the number of
    observations can fall below that floor and miss the deficiency.
    """
    return float(max(nobs, ncols) * np.finfo(float).eps)


def numerical_rank(singular_values, shape, rcond=None) -> int:
    """
    Numerical rank implied by the singular values of a matrix.

    Parameters
    ----------
    singular_values : array_like
        The singular values of the matrix.
    shape : tuple of int
        The shape of the matrix.
    rcond : float, optional
        Relative cutoff. Singular values less than or equal to
        ``rcond * max(singular_values)`` are treated as zero. If None,
        `default_rcond` is used.

    Returns
    -------
    int
        The number of singular values above the cutoff.
    """
    s = np.asarray(singular_values, dtype=float)
    if s.size == 0:
        return 0
    if rcond is None:
        rcond = default_rcond(*shape)
    return int(np.count_nonzero(s > rcond * s.max()))


class QRSVD:
    """
    Thin QR factorization of a matrix followed by an SVD of the R factor.

    With ``x = Q @ R`` and ``R = U @ diag(s) @ Vt`` the singular values of
    ``x`` are ``s`` and its pseudoinverse is ``V @ diag(1 / s) @ U.T @ Q.T``.
    Singular values at or below the cutoff are treated as zero, so the
    solutions are the minimum-norm least squares solutions.

    Parameters
    ----------
    x : array_like
        The 2-d matrix to factor, usually the (whitened) design matrix.

    Attributes
    ----------
    shape : tuple of int
        The shape of ``x``.
    singular_values : ndarray
        The singular values of ``x`` in decreasing order.

    Notes
    -----
    The rank and every quantity derived from it take the ``rcond`` argument.
    If it is None the rule in `default_rcond` applies.
    """

    def __init__(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be 2-dimensional.")
        self.shape = x.shape
        self._q, r = np.linalg.qr(x)
        self._u, self.singular_values, self._vt = np.linalg.svd(r, full_matrices=False)

    def rank(self, rcond=None) -> int:
        """
        Numerical rank of the factored matrix.

        Parameters
        ----------
        rcond : float, optional
            Relative cutoff for the singular values.

        Returns
        -------
        int
            The numerical rank.
        """
        return numerical_rank(self.singular_values, self.shape, rcond)

    def _inverse_singular_values(self, rcond):
        s = self.singular_values
        sinv = np.zeros_like(s)
        if s.size:
            if rcond is None:
                rcond = default_rcond(*self.shape)
            keep = s > rcond * s.max()
            sinv[keep] = 1.0 / s[keep]
        return sinv

    def effects(self, y):
        """
        Project the data on the orthonormal basis of the column space.

        Parameters
        ----------
        y : array_like
            The data, 1-d or 2-d with ``nobs`` rows.

        Returns
        -------
        ndarray
            ``Q.T @ y``, with one row for each column of the thin ``Q``.
        """
        return self._q.T @ np.asarray(y, dtype=float)

    def solve(self, y, rcond=None):
        """
        Minimum-norm least squares solution.

        Parameters
        ----------
        y : array_like
            The data, 1-d or 2-d with ``nobs`` rows.
        rcond : float, optional
            Relative cutoff for the singular values.

        Returns
        -------
        ndarray
            The coefficients, with one row for each column of the matrix.
        """
        sinv = self._inverse_singular_values(rcond)
        return (self._vt.T * sinv) @ (self._u.T @ self.effects(y))

    def pinv(self, rcond=None):
        """
        Pseudoinverse of the factored matrix.

        Parameters
        ----------
        rcond : float, optional
            Relative cutoff for the singular values.

        Returns
        -------
        ndarray
            The pseudoinverse, with shape ``(ncols, nobs)``.
        """
        sinv = self._inverse_singular_values(rcond)
        return (self._vt.T * sinv) @ (self._q @ self._u).T

    def cov_unscaled(self, rcond=None):
        """
        Pseudoinverse of the cross-product, ``pinv(x.T @ x)``.

        Parameters
        ----------
        rcond : float, optional
            Relative cutoff for the singular values.

        Returns
        -------
        ndarray
            ``V @ diag(1 / s**2) @ V.T``, the unscaled covariance of the
            least squares coefficients, with shape ``(ncols, ncols)``.

        Notes
        -----
        The cross-product is never formed, so the accuracy depends on the
        condition number of ``x`` and not on its square.
        """
        sinv = self._inverse_singular_values(rcond)
        return (self._vt.T * sinv**2) @ self._vt
