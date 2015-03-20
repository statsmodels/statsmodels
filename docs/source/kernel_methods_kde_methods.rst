.. currentmodule:: statsmodels.kernel_methods

Module :py:mod:`.kde_methods`
=============================

.. automodule:: statsmodels.kernel_methods.kde_methods

All the methods inherit the same base class. Although methods are not required
to inherit it, it is highly recommended, as it provides a number of services,
such as computed the total weights, the number of points or dimensions.

* :py:class:`KDEMethod`

1D continuous KDE methods
-------------------------

First, there is a specialized base class for 1D continuous methods. It provide
services, such as computing the CDF, hazard, survival function, ... based on the
PDF only.

List of 1D continuous KDE methods:

* :py:class:`KDE1DMethod`
* :py:class:`Reflection1D`
* :py:class:`Cyclic1D`
* :py:class:`Renormalization`
* :py:class:`LinearCombination`
* :py:class:`TransformKDE1D`

1D discrete KDE methods
-----------------------

List of discrete KDE methods:

* :py:class:`UnorderedKDE`
* :py:class:`OrderedKDE`

Multi-dimensional KDE methods
-----------------------------

List of multi-dimensional KDE methods:

* :py:class:`KDEnDMethod`
* :py:class:`Cyclic`

Multi-variate methods
---------------------

List of multi-variate KDE methods:

* :py:class:`MultivariateKDE`

Utility functions
-----------------

List of utility functions:

* :py:func:`convolve`
* :py:func:`generate_grid1d`
* :py:func:`generate_grid`
* :py:func:`filter_exog`
* :py:func:`create_transform`
* :py:class:`Transform`
* :py:data:`LogTransform`
* :py:data:`ExpTransform`

Module reference
----------------

.. autoclass:: KDEMethod
    :members: adjust, axis_type, bandwidth, exog, fitted, kernel, lower, ndim
              npts, total_weights, upper, weights

----------------

.. autoclass:: KDE1DMethod
    :members:
    :special-members: __call__

    Base class: :py:class:`KDEMethod`

----------------

.. autoclass:: Reflection1D

    Base class: :py:class:`KDE1DMethod`

----------------

.. autoclass:: Cyclic1D

    Base class: :py:class:`KDE1DMethod`

----------------

.. autoclass:: Renormalization

    Base class: :py:class:`KDE1DMethod`

----------------

.. autoclass:: LinearCombination

    Base class: :py:class:`KDE1DMethod`

----------------

.. autoclass:: TransformKDE1D
    :members: method

    Base class: :py:class:`KDE1DMethod`

----------------

.. autoclass:: UnorderedKDE
    :members:
    :special-members: __call__

    Base class: :py:class:`KDEMethod`

----------------

.. autoclass:: OrderedKDE
    :members:

    Base class: :py:class:`UnorderedKDE`

----------------

.. autoclass:: KDEnDMethod
    :members:
    :special-members: __call__

    Base class: :py:class:`KDEMethod`

----------------

.. autoclass:: Cyclic

    Base class: :py:class:`KDEnDMethod`

----------------

.. autoclass:: MultivariateKDE
    :members:
    :special-members: __call__

    Base class: :py:class:`KDEMethod`

----------------

.. autofunction:: convolve

----------------

.. autofunction:: generate_grid1d

----------------

.. autofunction:: generate_grid

----------------

.. autofunction:: filter_exog

----------------

.. autoclass:: Transform
    :members: __call__, inv, Dinv

----------------

.. autofunction:: create_transform

----------------

.. autodata:: LogTransform

----------------

.. autodata:: ExpTransform
