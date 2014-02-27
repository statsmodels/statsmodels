:orphan:

===========
0.6 Release
===========

Release 0.6.0
=============

Release summary.

Major changes:

Addition of Generalized Estimating Equations GEE



Generalized Estimating Equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generalized Estimating Equations (GEE) provide an approach to handling
dependent data in a regression analysis.  Dependent data arise
commonly in practice, such as in a longitudinal study where repeated
observations are collected on subjects. GEE can be viewed as an
extension of the generalized linear modeling (GLM) framework to the
dependent data setting.  The familiar GLM families such as the
Gaussian, Poisson, and logistic families can be used to accommodate
dependent variables with various distributions.

Here is an example of GEE Poisson regression in a data set with four
count-type repeated measures per subject, and three explanatory
covariates.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from statsmodels.genmod.generalized_estimating_equations import GEE
   from statsmodels.genmod.dependence_structures import Independence
   from statsmodels.genmod.families import Poisson
   
   data_url = "http://vincentarelbundock.github.io/Rdatasets/csv/MASS/epil.csv"
   data = pd.read_csv(data_url)
   
   fam = Poisson()
   ind = Independence()
   md = GEE.from_formula("y ~ age + trt + base", data, groups=data["subject"],
                          covstruct=ind, family=fam)
   mdf = md.fit()
   print mdf.summary()


The dependence structure in a GEE is treated as a nuisance parameter
and is modeled in terms of a "working dependence structure".  The
statsmodels GEE implementation currently includes five working
dependence structures (independent, exchangeable, autoregressive,
nested, and a global odds ratio for working with categorical data).
Since the GEE estimates are not maximum likelihood estimates,
alternative approaches to some common inference procedures have been
developed.  The statsmodels GEE implementation currently provides
standard errors, Wald tests, score tests for arbitrary parameter
contrasts, and estimates and tests for marginal effects.  Several
forms of standard errors are provided, including robust standard
errors that are approximately correct even if the working dependence
structure is misspecified.



Seasonality Plots
~~~~~~~~~~~~~~~~~

Adding functionality to look at seasonality in plots. Two new functions are :func:`sm.graphics.tsa.month_plot` and :func:`sm.graphics.tsa.quarter_plot`. Another function :func:`sm.graphics.tsa.seasonal_plot` is available for power users.

.. code-block:: python

    import statsmodels.api as sm
    import pandas as pd

    dta = sm.datasets.elnino.load_pandas().data
    dta['YEAR'] = dta.YEAR.astype(int).astype(str)
    dta = dta.set_index('YEAR').T.unstack()
    dates = map(lambda x : pd.datetools.parse('1 '+' '.join(x)),
                                           dta.index.values)

    dta.index = pd.DatetimeIndex(dates, freq='M')
    fig = sm.tsa.graphics.month_plot(dta)

.. currentmodule:: statsmodels.tsa

Seasonal Decomposition
----------------------

We added a naive seasonal decomposition tool in the same vein as R's ``decompose``. This function can be found as :func:`sm.tsa.seasonal_decompose <tsa.seasonal.seasonal_decompose>`.


.. code-block:: python

    import statsmodels.api as sm

    dta = sm.datasets.co2.load_pandas().data
    # deal with missing values. see issue 
    dta.co2.interpolate(inplace=True)

    res = sm.tsa.seasonal_decompose(dta.co2)
    res.plot()


Other important new features
----------------------------

* The Kalman filter Cython code underlying AR(I)MA estimation has been substantially optimized. You can expect speed-ups of one to two orders of magnitude.

* Added :func:`sm.tsa.arma_order_select_ic`. A convenience function to quickly get the information criteria for use in tentative order selection of ARMA processes.

* Plotting functions for timeseries is now imported under the ``sm.tsa.graphics`` namespace in addition to ``sm.graphics.tsa``.

* **New datasets**: Added new :ref:`datasets <datasets>` for examples. ``sm.datasets.co2`` is a univariate time-series dataset of weekly co2 readings. It exhibits a trend and seasonality and has missing values.

* Added robust skewness and kurtosis estimators in :func:`sm.stats.stattools.robust_skewness` and :func:`sm.stats.stattools.robust_kurtosis`, respectively.  An alternative robust measure of skewness has been added in :func:`sm.stats.stattools.medcouple`.

* New functions added to correlation tools: `corr_nearest_factor`
  finds the closest factor-structured correlation matrix to a given
  square matrix in the Frobenius norm; `corr_thresholded` efficiently
  constructs a hard-thresholded correlation matrix using sparse matrix
  operations.

* A dotplot is a way to visualize a small dataset in a way that
  retains the identity of every point in the plot.  Dotplots are very
  commonly found in meta-analyses, where they are known as "forest
  plots", but can be used in many other settings as well.  Most tables
  that appear in research papers can be represented graphically as a
  dotplot.


Major Bugs fixed
----------------

* Bullet list of major bugs
* With a link to its github issue.
* Use the syntax ``:ghissue:`###```.

.. currentmodule:: statsmodels.tsa

Backwards incompatible changes and deprecations
-----------------------------------------------

* RegressionResults.norm_resid is now a readonly property, rather than a function.
* The function ``statsmodels.tsa.filters.arfilter`` has been removed. This did not compute a recursive AR filter but was instead a convolution filter. Two new functions have been added with clearer names :func:`sm.tsa.filters.recursive_filter <tsa.filters.filtertools.recursive_filter>` and :func:`sm.tsa.filters.convolution_filter <tsa.filters.filtertools.convolution_filter>`.

Development summary and credits
-------------------------------

A blurb about the number of changes and the contributors list.

.. note::

   Obtained by running ``git log v0.5.0..HEAD --format='* %aN <%aE>' | sed 's/@/\-at\-/' | sed 's/<>//' | sort -u``.

