"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This module contains kernels for non-continuous data.

Unlike with continuous kernels, these ones require explicitely the evaluation
point and the bandwidth.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class AitchisonAitken(object):
    r"""
    The Aitchison-Aitken kernel, used for unordered discrete random variables
    [KM2]_.

    See p.18 of [KM3]_ for details.  The value of the kernel L if
    :math:`X_{i}=x` is :math:`1-\lambda`, otherwise it is
    :math:`\frac{\lambda}{c-1}`. Here :math:`c` is the number of levels plus
    one of the RV.

    References
    ----------
    .. [KM2] J. Aitchison and C.G.G. Aitken, "Multivariate binary
             discrimination by the kernel method", Biometrika,
             Vol. 63, pp. 413-420, 1976.
    .. [KM3] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation
             and Trends in Econometrics: Vol 3: No 1, pp1-88., 2008.
    """
    def for_ndim(self, ndim):
        assert ndim == 1, "Error, this kernel only works in 1D"
        return self

    @property
    def ndim(self):
        return 1

    def cut(self, bw, epsilon):
        """
        Parameters
        ----------
        bw: float in [0,1]
            Bandwidth used
        epsilon: float
            Precision required

        Returns
        -------
        The number of categories to add on either direction to ensure no weight
        is lost.
        """
        return 0

    def pdf(self, x, Xi, bw, num_levels, out=None):
        """
        Compute the PDF on the points x

        Parameters
        ----------
        x: ndarray
            Points to evaluated the PDF at
        Xi: ndarray
            Training dataset
        bw: float
            Bandwidth
        num_levels: int
            Number of levels possible. The levels will range from 0 to
            num_levels-1.

        Returns
        -------
        Result of the pdf on x from Xi. If x and Xi are arrays on different
        dimension, the outer product will be performed
        """
        x = np.asfarray(x)
        bw = float(bw)
        Xi = np.asfarray(Xi)
        dx = Xi - x
        if dx.ndim == 0:
            dx.shape = (1,)
        if out is None:
            out = np.empty_like(dx)
        out[...] = bw / (num_levels-1)
        out[dx == 0] = 1 - bw
        return out

    def from_binned(self, mesh, bins, bw, dim=-1):
        num_levels = bins.shape[dim]
        all_vals = np.sum(bins, axis=dim, keepdims=True)
        result = bins*(1-bw)
        result += (all_vals - bins) * bw / (num_levels-1)
        return result


class WangRyzin(object):
    r"""
    The Wang-Ryzin kernel, used for ordered discrete random variables [KM5]_.

    Notes
    -----
    See p. 19 in [KM4]_ for details.  The value of the kernel L if
    :math:`X_{i}=x` is :math:`1-\lambda`, otherwise it is
    :math:`\frac{1-\lambda}{2}\lambda^{|X_{i}-x|}`, where :math:`\lambda` is
    the bandwidth.

    References
    ----------
    .. [KM4] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation
           and Trends in Econometrics: Vol 3: No 1, pp1-88., 2008.
           http://dx.doi.org/10.1561/0800000009
    .. [KM5] M.-C. Wang and J. van Ryzin, "A class of smooth estimators for
           discrete distributions", Biometrika, vol. 68, pp. 301-309, 1981.
    """
    def for_ndim(self, ndim):
        assert ndim == 1, "Error, this kernel only works in 1D"
        return self

    @property
    def ndim(self):
        return 1

    def cut(self, bw, epsilon):
        """
        Parameters
        ----------
        bw: float in [0,1]
            Bandwidth used
        epsilon: float
            Precision required

        Returns
        -------
        The number of categories to add on either direction to ensure no weight
        is lost.
        """
        return int(np.ceil(np.log(epsilon) / np.log(bw) - 1))

    def pdf(self, x, Xi, bw, num_levels, out=None):
        """
        Compute the PDF on the points x

        Parameters
        ----------
        x: float or 1-D ndarray (K,)
            Points to evaluated the PDF at
        Xi: ndarray of ints, shape (1,nobs) or (K, nobs)
            Training dataset
        bw: float or 1-D array of shape (K,)
            Bandwidth
        num_levels: int
            Number of levels possible. The levels will range from 0 to
            num_levels-1.

        Returns
        -------
        Result of the pdf on x from Xi. If x and Xi are arrays on different
        dimension, the outer product will be performed.
        """
        x = np.asfarray(x)
        bw = float(bw)
        Xi = np.asfarray(Xi)
        dx = Xi - x
        if dx.ndim == 0:
            dx.shape = (1,)
        if out is None:
            out = np.empty_like(dx)
        out[...] = (1 - bw)/2 * bw**abs(dx)
        out[dx == 0] = 1 - bw
        return out

    def from_binned(self, mesh, bins, bw, dim=-1):
        factor = (1-bw)/2
        result = factor*bins
        grid = mesh.sparse()[dim]
        selector = [slice(None)] * mesh.ndim
        for i, level in enumerate(mesh.grid[dim]):
            selector[dim] = i
            result[selector] += factor * np.sum(bw ** abs(grid - level) * bins,
                                                axis=dim)
        return result
