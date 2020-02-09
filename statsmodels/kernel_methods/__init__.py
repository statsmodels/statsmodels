"""
Tools for nonparametric statistics, mainly density estimation and regression.

For an overview of this module, see docs/source/nonparametric.rst
"""

from . import kde, kde_methods, kernels, bandwidths, kde_utils

__all__ = ['kde', 'kde_methods', 'kernels', 'bandwidths', 'kde_utils']
