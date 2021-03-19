.. currentmodule:: statsmodels.kernel_methods

Module :py:mod:`.kernels`
=========================

Kernel base classes
-------------------

.. autosummary::
    :toctree: generated/

    kernels.Kernel1D
    kernels.KernelnD

1D kernels
----------

.. autosummary::
    :toctree: generated/

    kernels.from1DPDF
    kernels.normal1d
    _kernels1d.tricube
    _kernels1d.Epanechnikov
    _kernels1d.normal_order4
    _kernels1d.Epanechnikov_order4

Discrete kernels
----------------

.. autosummary::
    :toctree: generated/

    _kernelsnc.WangRyzin
    _kernelsnc.AitchisonAitken

Multi-dimensional kernels
-------------------------

.. autosummary::
    :toctree: generated/

    kernels.normal

Utility functions
-----------------

.. autosummary::
    :toctree: generated/

    kernels.rfftfreq
    kernels.rfftsize
    kernels.fftsamples
    kernels.rfftnfreq
    kernels.rfftnsize
    kernels.fftnsamples
    kernels.dctfreq
    kernels.dctsamples
    kernels.dctnfreq
    kernels.dctnsamples
