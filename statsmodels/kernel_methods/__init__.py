"""
Tools for nonparametric statistics, mainly density estimation and regression.

For an overview of this module, see docs/source/nonparametric.rst
"""

from statsmodels import NoseWrapper as Tester
test = Tester().test

__all__ = ['kde', 'kde_methods', 'kernels', 'bandwidths', 'kde_utils']
