.. currentmodule:: statsmodels.kernel_methods

Module :py:mod:`kernels`
========================

.. automodule:: statsmodels.kernel_methods.kernels

Kernel base classes:

* :py:class:`Kernel1D`
* :py:class:`KernelnD`

List of 1D kernels:

* :py:class:`from1DPDF`
* :py:class:`normal1d`
* :py:class:`tricube`
* :py:class:`Epanechnikov`
* :py:class:`normal_order4`
* :py:class:`Epanechnikov_order4`

List of discrete kernels:

* :py:class:`WangRyzin`
* :py:class:`AitchisonAitken`

List of multi-dimensional kernels:

* :py:class:`normal`

List of utility functions:

* :py:func:`rfftfreq`
* :py:func:`rfftsize`
* :py:func:`fftsamples`
* :py:func:`rfftnfreq`
* :py:func:`rfftnsize`
* :py:func:`fftnsamples`
* :py:func:`dctfreq`
* :py:func:`dctsamples`
* :py:func:`dctnfreq`
* :py:func:`dctnsamples`

.. autoclass:: Kernel1D
    :members:
    :special-members: __call__

----------------

.. autoclass:: KernelnD
    :members:
    :special-members: __call__

----------------

.. autoclass:: normal1d
    :members:
    :special-members: __call__

    Base class: :py:class:`Kernel1D`

----------------

.. autoclass:: from1DPDF
    :members:
    :special-members: __call__

    Base class: :py:class:`Kernel1D`

----------------

.. autoclass:: tricube
    :members:
    :special-members: __call__

    Base class: :py:class:`Kernel1D`

----------------

.. autoclass:: Epanechnikov
    :members:
    :special-members: __call__

    Base class: :py:class:`Kernel1D`

----------------

.. autoclass:: normal_order4
    :members:
    :special-members: __call__

    Base class: :py:class:`Kernel1D`

----------------

.. autoclass:: Epanechnikov_order4
    :members:
    :special-members: __call__

    Base class: :py:class:`Kernel1D`

----------------

.. autoclass:: WangRyzin
    :members:
    :special-members: __call__

----------------

.. autoclass:: AitchisonAitken
    :members:
    :special-members: __call__

----------------

.. autoclass:: normal
    :members:
    :special-members: __call__

    Base class: :py:class:`KernelnD`

----------------

.. autofunction:: rfftfreq

----------------

.. autofunction:: rfftsize

----------------

.. autofunction:: fftsamples

----------------

.. autofunction:: rfftnfreq

----------------

.. autofunction:: rfftnsize

----------------

.. autofunction:: fftnsamples

----------------

.. autofunction:: dctfreq

----------------

.. autofunction:: dctsamples

----------------

.. autofunction:: dctnfreq

----------------

.. autofunction:: dctnsamples
