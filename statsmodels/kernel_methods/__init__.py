"""
Tools for nonparametric statistics, mainly density estimation and regression.

For an overview of this module, see docs/source/nonparametric.rst
"""

from . import (kde, kde_1d, kde_nd, kde_nc, kde_multivariate, kernels1d,
               kernelsnc, kernelsnd, bandwidths, kde_utils)

__all__ = ['kde', 'kde_1d', 'kde_nd', 'kde_nc', 'kde_multivariate',
           'kernels1d', 'kernelsnc', 'kernelsnd', 'bandwidths', 'kde_utils']
