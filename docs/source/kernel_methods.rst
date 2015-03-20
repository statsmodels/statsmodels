.. currentmodule:: statsmodels.kernel_methods

.. _kernel_methods:

Kernel Methods :mod:`kernel_methods`
====================================

:mod:`statsmodels.kernel_methods` collect kernel-based methods, currently only
directed toward estimation of density of probability. It currently handles
univariate, multivariate and multidimensional data, which can be continuous or
discrete.

The :py:class:`kde.KDE` object is the entrance point to all these methods. From there, the
user defines the data, its properties, and the methods to use for the
estimation. :py:class:`kde_utils.Grid`.

Univariate continuous KDE
-------------------------

This is the simplest, and default, case. As an example, let's look at the
distribution of annual flow of the Nile:

.. code-block:: python

    >>> from statsmodels import datasets
    >>> import statsmodels.kernel_methods.api as km
    >>> from matplotlib import pyplot as plt
    >>> plt.ion()
    >>> import pandas as pd
    >>> ds = datasets.nile.load()
    >>> df = pd.DataFrame(ds.data)
    >>> df.volume.describe()
    count     100.000000
    mean      919.350000
    std       169.227501
    min       456.000000
    25%       798.500000
    50%       893.500000
    75%      1032.500000
    max      1370.000000
    Name: volume, dtype: float64

We can simply use the default options for this::

    >>> m = km.KDE(df.volume)
    >>> est = m.fit()
    >>> xs, ys = est.grid()
    >>> f = plt.figure()
    >>> plt.plot(xs, ys)
    >>> plt.xlabel('Flow')
    >>> plt.ylabel('Frequency')
    >>> plt.show()

Bounded domains
^^^^^^^^^^^^^^^

Transformed axis
^^^^^^^^^^^^^^^^

Univariate discrete KDE
-----------------------

Multi-dimensional KDE
---------------------

Multivariate KDE
----------------

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

    kernel_methods_kde
    kernel_methods_kde_methods
    kernel_methods_bandwidths
    kernel_methods_kernels
    kernel_methods_utils
