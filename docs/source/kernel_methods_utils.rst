.. currentmodule:: statsmodels.kernel_methods

Utility modules
===============

List of modules:

* :py:mod:`statsmodels.kernel_methods.kde_utils`
* :py:mod:`statsmodels.kernel_methods.fast_linbin`

Module :py:mod:`kde_utils`
--------------------------

.. automodule:: statsmodels.kernel_methods.kde_utils

List of classes:

* :py:class:`Grid`
* :py:class:`GridInterpolator`

.. autoclass:: Grid
    :members: almost_equal, bin_sizes, bin_type, bin_volumes, bounds, copy,
              cum_integrate, dtype, edges, fromArrays, fromBounds, fromFull,
              fromSparse, full, grid, integrate, linear, ndim, shape, sparse,
              start_interval, start_volume, transform

.. autoclass:: GridInterpolator
    :members: __call__, ndim

.. currentmodule:: statsmodels.kernel_methods

Module :py:mod:`fast_linbin`
----------------------------

.. automodule:: statsmodels.kernel_methods.fast_linbin

.. currentmodule:: statsmodels.kernel_methods.fast_linbin

List of functions:

* :py:func:`fast_linbin`
* :py:func:`fast_bin`
* :py:func:`fast_linbin_nd`
* :py:func:`fast_bin_nd`

.. autofunction:: fast_linbin

.. autofunction:: fast_bin

.. autofunction:: fast_linbin_nd

.. autofunction:: fast_bin_nd

