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
md1 = GEE.from_formula("y ~ age + trt + base", data, groups=data["subject"],\
                       covstruct=ind, family=fam)
mdf1 = md1.fit()
print mdf1.summary()


The dependence structure in a GEE is treated as a nuisance parameter
and is modeled in terms of a "working dependence structure".  The
statsmodels GEE implementation currently includes five working
dependence structures (independent, exchangeable, autoregressive,
nested, and a global odds ratio for working with categorical data).
Since the GEE estimates are not maximum likelihood estimates,
alternative approaches to some common inference procedures have been
developed.  The statsmodels GEE implementation currently provides
standard errors and allows score tests for arbitrary parameter
contrasts.



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

ARMA Fitting Speed-ups
~~~~~~~~~~~~~~~~~~~~~~

The Cython code for the Kalman filter which the ARIMA model code relies on has been rewritten to be significantly faster. You can expect speed-ups from about 10 to 20x. As a consequence of these changes, Cython >= 0.18 is now required to build statsmodels from github source. The generated C files will still be included in all released source distributions, so, again, Cython is only required when building from github.

Other important new features
----------------------------

* Statsmodels now requires Cython >= 0.18 to build from source.
* Added :func:`sm.tsa.arma_order_select_ic`. A convenience function to quickly get the information criteria for use in tentative order selection of ARMA processes.

* Plotting functions for timeseries is now imported under the ``sm.tsa.graphics`` namespace in addition to ``sm.graphics.tsa``.

* **New datasets**: Added new :ref:`datasets <datasets>` for examples. ``sm.datasets.co2`` is a univariate time-series dataset of weekly co2 readings. It exhibits a trend and seasonality and has missing values.

Major Bugs fixed
----------------

* Bullet list of major bugs
* With a link to its github issue.
* Use the syntax ``:ghissue:`###```.

Backwards incompatible changes and deprecations
-----------------------------------------------

* RegressionResults.norm_resid is now a readonly property, rather than a function.

Development summary and credits
-------------------------------

A blurb about the number of changes and the contributors list.

.. note::

   Obtained by running ``git log v0.5.0..HEAD --format='* %aN <%aE>' | sed 's/@/\-at\-/' | sed 's/<>//' | sort -u``.

