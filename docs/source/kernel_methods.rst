.. currentmodule:: statsmodels

.. _kernel_methods:

Kernel Methods :mod:`kernel_methods`
====================================

.. currentmodule:: statsmodels.kernel_methods

This modules currently implements a variety of methods for kernel density
estimation for Multi-variate data on bounded domains.

The main entry point of the module is the :py:class:`kde.KDE` object, which is
used to define the parameters of the density estimation. After construction, the
user needs to call the method :py:meth:`kde.KDE.fit` to obtain a fitted model
from which the density and other quantities can be computed. The exact list of
quantities being computed depends on the object fitted and the method used.

The module :py:mod:`.kde_methods` contains all the methods that are defined.
While fitting, each method is free is use a different, more adapted method given
the parameters. For example, the :py:class:`.kde_methods.MultivariateKDE` method
will return a univariate method if the data has a single dimension. Same for the
multi-dimensional methods returning the equivalent 1D method for efficiency
purposes.

Examples
--------

Here is a list of example notebooks:

.. toctree::

    examples/notebooks/generated/kernel_methods_kde1d
    examples/notebooks/generated/kernel_methods_crossvalidation
    examples/notebooks/generated/kernel_methods_kdend

References
----------

* B.W. Silverman, "Density Estimation for Statistics and Data Analysis"
* J.S. Racine, "Nonparametric Econometrics: A Primer," Foundation and
  Trends in Econometrics, Vol. 3, No. 1, pp. 1-88, 2008.
* Q. Li and J.S. Racine, "Nonparametric econometrics: theory and practice",
  Princeton University Press, 2006.
* Racine, J., Li, Q. "Nonparametric Estimation of Distributions
  with Categorical and Continuous Data." Working Paper. (2000)
* Racine, J. Li, Q. "Kernel Estimation of Multivariate Conditional
  Distributions Annals of Economics and Finance 5, 211-235 (2004)
* Liu, R., Yang, L. "Kernel estimation of multivariate
  cumulative distribution function." Journal of Nonparametric Statistics 
  (2008)
* Li, R., Ju, G. "Nonparametric Estimation of Multivariate CDF
  with Categorical and Continuous Data." Working Paper
* Li, Q., Racine, J. "Cross-validated local linear nonparametric
  regression" Statistica Sinica 14(2004), pp. 485-512


Module Reference
----------------

.. toctree::
    :maxdepth: 2

    kernel_methods_kde
    kernel_methods_kde_methods
    kernel_methods_bandwidths
    kernel_methods_kernels
    kernel_methods_utils
