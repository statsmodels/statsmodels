:orphan:

=============
Release 0.7.0
=============

Release summary
---------------

**Note:** This version has never been officially released. Several models have
been refactored, improved or bugfixed in 0.8.


The following major new features appear in this version.

Principal Component Analysis
----------------------------

Author: Kevin Sheppard

A new class-based Principal Component Analysis has been added.  This
class replaces the function-based PCA that previously existed in the
sandbox.  This change bring a number of new features, including:

* Options to control the standardization (demeaning/studentizing)
* Scree plotting
* Information criteria for selecting the number of factors
* R-squared plots to assess component fit
* NIPALS implementation when only a small number of components are required and the dataset is large
* Missing-value filling using the EM algorithm

.. code-block:: python

   import statsmodels.api as sm
   from statsmodels.multivariate.pca import PCA

   data = sm.datasets.fertility.load_pandas().data

   columns = map(str, range(1960, 2012))
   data.set_index('Country Name', inplace=True)
   dta = data[columns]
   dta = dta.dropna()

   pca_model = PCA(dta.T, standardize=False, demean=True)
   pca_model.plot_scree()

*Note* : A function version is also available which is compatible with the
call in the sandbox.  The function version is just a thin wrapper around the
class-based PCA implementation.

Regression graphics for GLM/GEE
-------------------------------

Author: Kerby Shedden

Added variable plots, partial residual plots, and CERES residual plots
are available for GLM and GEE models by calling the methods
`plot_added_variable`, `plot_partial_residuals`, and
`plot_ceres_residuals` that are attached to the results classes.

State Space Models
------------------

Author: Chad Fulton

State space methods provide a flexible structure for the estimation and
analysis of a wide class of time series models. The statsmodels implementation
allows specification of state models, fast Kalman filtering, and built-in
methods to facilitate maximum likelihood estimation of arbitrary models. One of
the primary goals of this module is to allow end users to create and estimate
their own models. Below is a short example demonstrating the ease with which a
local level model can be specified and estimated:

.. code-block:: python

   import numpy as np
   import statsmodels.api as sm
   import pandas as pd

   data = sm.datasets.nile.load_pandas().data
   data.index = pd.DatetimeIndex(data.year.astype(int).astype(str), freq='AS')

   # Setup the state space representation
   class LocalLevel(sm.tsa.statespace.MLEModel):
       def __init__(self, endog):
           # Initialize the state space model
           super(LocalLevel, self).__init__(
               endog, k_states=1, initialization='approximate_diffuse')

           # Setup known components of state space representation matrices
           self.ssm['design', :] = 1.
           self.ssm['transition', :] = 1.
           self.ssm['selection', :] = 1.

       # Describe how parameters enter the model
       def update(self, params, transformed=True):
           params = super(LocalLevel, self).update(params, transformed)
           self.ssm['obs_cov', 0, 0] = params[0]
           self.ssm['state_cov', 0, 0] = params[1]

       def transform_params(self, params):
           return params**2  # force variance parameters to be positive

       # Specify start parameters and parameter names
       @property
       def start_params(self):
           return [np.std(self.endog)]*2

       @property
       def param_names(self):
           return ['sigma2.measurement', 'sigma2.level']

   # Fit the model with maximum likelihood estimation
   mod = LocalLevel(data['volume'])
   res = mod.fit()
   print res.summary()

The documentation and example notebooks provide further examples of how to
form state space models. Included in this release is a full-fledged
model making use of the state space infrastructure to estimate SARIMAX
models. See below for more details.

Time Series Models (ARIMA) with Seasonal Effects
------------------------------------------------

Author: Chad Fulton

A model for estimating seasonal autoregressive integrated moving average models
with exogenous regressors (SARIMAX) has been added by taking advantage of the
new state space functionality. It can be used very similarly to the existing
`ARIMA` model, but works on a wider range of specifications, including:

* Additive and multiplicative seasonal effects
* Flexible trend specification
* Regression with SARIMA errors
* Regression with time-varying coefficients
* Measurement error in the endogenous variables

Below is a short example fitting a model with a number of these components,
including exogenous data, a linear trend, and annual multiplicative seasonal
effects.

.. code-block:: python

   import statsmodels.api as sm
   import pandas as pd

   data = sm.datasets.macrodata.load_pandas().data
   data.index = pd.DatetimeIndex(start='1959-01-01', end='2009-09-01',
                                 freq='QS')
   endog = data['realcons']
   exog = data['m1']

   mod = sm.tsa.SARIMAX(endog, exog=exog, order=(1,1,1),
                        trend='t', seasonal_order=(0,0,1,4))
   res = mod.fit()
   print res.summary()


Generalized Estimating Equations GEE
------------------------------------

Author: Kerby Shedden

Enhancements and performance improvements for GEE:

* EquivalenceClass covariance structure allows covariances to be specified by
  arbitrary collections of equality constraints #2188
* add weights #2090
* refactored margins #2158


MixedLM
-------

Author: Kerby Shedden with Saket Choudhary

Enhancements to MixedLM (#2363): added variance components support for
MixedLM allowing a wider range of random effects structures to be specified;
also performance improvements from use of sparse matrices internally for
random effects design matrices.


Other important new features
----------------------------

* GLM: add scipy-based gradient optimization to fit #1961 (Kerby Shedden)
* wald_test_terms: new method of LikelihoodModels to compute wald tests (F or chi-square)
  for terms or sets of coefficients #2132  (Josef Perktold)
* add cov_type with fixed scale in WLS to allow chi2-fitting #2137 #2143
  (Josef Perktold, Christoph Deil)
* VAR: allow generalized IRF and FEVD computation #2067 (Josef Perktold)
* get_prediction new method for full prediction results (new API convention)



Major Bugs fixed
----------------

* see github issues for a full list
* bug in ARMA/ARIMA predict with `exog` #2470
* bugs in VAR
* x13: python 3 compatibility



Backwards incompatible changes and deprecations
-----------------------------------------------

* List backwards incompatible changes


Development summary and credits
-------------------------------



.. note::

  Thanks to all of the contributors for the 0.7 release:

.. note::

   * Alex Griffing
   * Antony Lee
   * Chad Fulton
   * Christoph Deil
   * Daniel Sullivan
   * Hans-Martin von Gaudecker
   * Jan Schulz
   * Joey Stockermans
   * Josef Perktold
   * Kerby Shedden
   * Kevin Sheppard
   * Kiyoto Tamura
   * Louis-Philippe Lemieux Perreault
   * Padarn Wilson
   * Ralf Gommers
   * Saket Choudhary
   * Skipper Seabold
   * Tom Augspurger
   * Trent Hauck
   * Vincent Arel-Bundock
   * chebee7i
   * donbeo
   * gliptak
   * hlin117
   * jerry dumblauskas
   * jonahwilliams
   * kiyoto
   * neilsummers
   * waynenilsen

These lists of names are automatically generated based on git log, and may not be
complete.
