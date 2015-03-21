.. currentmodule:: statsmodels.kernel_methods

Module :py:mod:`.kde_methods`
=============================

All the methods inherit the same base class. Although methods are not required
to inherit it, it is highly recommended, as it provides a number of services,
such as computed the total weights, the number of points or dimensions.

Note that all the classes and functions described here are aliased in the
`kde_methods` module.

.. autosummary::
    :toctree: generated/

    _kde_methods.KDEMethod

1D continuous KDE methods
-------------------------

First, there is a specialized base class for 1D continuous methods. It provide
services, such as computing the CDF, hazard, survival function, ... based on the
PDF only.

List of 1D continuous KDE methods:

.. autosummary::
    :toctree: generated/

    _kde1d_methods.KDE1DMethod
    _kde1d_methods.Reflection1D
    _kde1d_methods.Cyclic1D
    _kde1d_methods.Renormalization
    _kde1d_methods.LinearCombination
    _kde1d_methods.Transform1D

1D discrete KDE methods
-----------------------

List of discrete KDE methods:

.. autosummary::
    :toctree: generated/

    _kdenc_methods.Unordered
    _kdenc_methods.Ordered

Multi-dimensional KDE methods
-----------------------------

List of multi-dimensional KDE methods:

.. autosummary::
    :toctree: generated/

    _kdend_methods.KDEnDMethod
    _kdend_methods.Cyclic

Multi-variate methods
---------------------

List of multi-variate KDE methods:

.. autosummary::
    :toctree: generated/

    _kde_multivariate.Multivariate

Utility functions
-----------------

List of utility functions:

.. autosummary::
    :toctree: generated/

    _kde1d_methods.convolve
    _kde1d_methods.generate_grid1d
    _kdend_methods.generate_grid
    _kde_methods.filter_exog
    _kde1d_methods.create_transform
    _kde1d_methods.Transform
    _kde1d_methods.LogTransform
    _kde1d_methods.ExpTransform
