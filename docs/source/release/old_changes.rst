:orphan:

.. _old_changes:

Pre 0.5.0 Release History
=========================

0.5.0
-----
*Main Changes and Additions*
* Add patsy dependency

*Compatibility and Deprecation*

* cleanup of import paths (lowess)
*

*Bug Fixes*

* input shapes of tools.isestimable
*

*Enhancements and Additions*

* formula integration based on patsy (new dependency)
* Time series analysis
  - ARIMA modeling
  - enhanced forecasting based on pandas datetime handling
* expanded margins for discrete models
* OLS outlier test

* empirical likelihood - Google Summer of Code 2012 project
  - inference for descriptive statistics
  - inference for regression models
  - accelerated failure time models

* expanded probability plots
* improved graphics
  - plotcorr
  - acf and pacf
* new datasets
* new and improved tools
  - numdiff numerical differentiation



0.4.3
-----

The only change compared to 0.4.2 is for compatibility with python 3.2.3
(changed behavior of 2to3)


0.4.2
-----

This is a bug-fix release, that affects mainly Big-Endian machines.

*Bug Fixes*

* discrete_model.MNLogit fix summary method
* tsa.filters.hp_filter do not use umfpack on Big-Endian machine (scipy bug)
* the remaining fixes are in the test suite, either precision problems
  on some machines or incorrect testing on Big-Endian machines.



0.4.1
-----

This is a backwards compatible (according to our test suite) release with
bug fixes and code cleanup.

*Bug Fixes*

* build and distribution fixes
* lowess correct distance calculation
* genmod correction CDFlink derivative
* adfuller _autolag correct calculation of optimal lag
* het_arch, het_lm : fix autolag and store options
* GLSAR: incorrect whitening for lag>1

*Other Changes*

* add lowess and other functions to api and documentation
* rename lowess module (old import path will be removed at next release)
* new robust sandwich covariance estimators, moved out of sandbox
* compatibility with pandas 0.8
* new plots in statsmodels.graphics
  - ABLine plot
  - interaction plot


0.4.0
-----

*Main Changes and Additions*

* Added pandas dependency.
* Cython source is built automatically if cython and compiler are present
* Support use of dates in timeseries models
* Improved plots
  - Violin plots
  - Bean Plots
  - QQ Plots
* Added lowess function
* Support for pandas Series and DataFrame objects. Results instances return
  pandas objects if the models are fit using pandas objects.
* Full Python 3 compatibility
* Fix bugs in genfromdta. Convert Stata .dta format to structured array
  preserving all types. Conversion is much faster now.
* Improved documentation
* Models and results are pickleable via save/load, optionally saving the model
  data.
* Kernel Density Estimation now uses Cython and is considerably faster.
* Diagnostics for outlier and influence statistics in OLS
* Added El Nino Sea Surface Temperatures dataset
* Numerous bug fixes
* Internal code refactoring
* Improved documentation including examples as part of HTML

*Changes that break backwards compatibility*

* Deprecated scikits namespace. The recommended import is now::

      import statsmodels.api as sm

* model.predict methods signature is now (params, exog, ...) where before
  it assumed that the model had been fit and omitted the params argument.
* For consistency with other multi-equation models, the parameters of MNLogit
  are now transposed.
* tools.tools.ECDF -> distributions.ECDF
* tools.tools.monotone_fn_inverter -> distributions.monotone_fn_inverter
* tools.tools.StepFunction -> distributions.StepFunction


0.3.1
-----

* Removed academic-only WFS dataset.
* Fix easy_install issue on Windows.

0.3.0
-----

*Changes that break backwards compatibility*

Added api.py for importing. So the new convention for importing is::

    import statsmodels.api as sm

Importing from modules directly now avoids unnecessary imports and increases
the import speed if a library or user only needs specific functions.

* sandbox/output.py -> iolib/table.py
* lib/io.py -> iolib/foreign.py (Now contains Stata .dta format reader)
* family -> families
* families.links.inverse -> families.links.inverse_power
* Datasets' Load class is now load function.
* regression.py -> regression/linear_model.py
* discretemod.py -> discrete/discrete_model.py
* rlm.py -> robust/robust_linear_model.py
* glm.py -> genmod/generalized_linear_model.py
* model.py -> base/model.py
* t() method -> tvalues attribute (t() still exists but raises a warning)

*Main changes and additions*

* Numerous bugfixes.
* Time Series Analysis model (tsa)

  - Vector Autoregression Models VAR (tsa.VAR)
  - Autoregressive Models AR (tsa.AR)
  - Autoregressive Moving Average Models ARMA (tsa.ARMA)
    optionally uses Cython for Kalman Filtering
    use setup.py install with option --with-cython
  - Baxter-King band-pass filter (tsa.filters.bkfilter)
  - Hodrick-Prescott filter (tsa.filters.hpfilter)
  - Christiano-Fitzgerald filter (tsa.filters.cffilter)

* Improved maximum likelihood framework uses all available scipy.optimize solvers
* Refactor of the datasets sub-package.
* Added more datasets for examples.
* Removed RPy dependency for running the test suite.
* Refactored the test suite.
* Refactored codebase/directory structure.
* Support for offset and exposure in GLM.
* Removed data_weights argument to GLM.fit for Binomial models.
* New statistical tests, especially diagnostic and specification tests
* Multiple test correction
* General Method of Moment framework in sandbox
* Improved documentation
* and other additions


0.2.0
-----

*Main changes*

 * renames for more consistency
   RLM.fitted_values -> RLM.fittedvalues
   GLMResults.resid_dev -> GLMResults.resid_deviance
 * GLMResults, RegressionResults:
   lazy calculations, convert attributes to properties with _cache
 * fix tests to run without rpy
 * expanded examples in examples directory
 * add PyDTA to lib.io -- functions for reading Stata .dta binary files
   and converting
   them to numpy arrays
 * made tools.categorical much more robust
 * add_constant now takes a prepend argument
 * fix GLS to work with only a one column design

*New*

 * add four new datasets

   - A dataset from the American National Election Studies (1996)
   - Grunfeld (1950) investment data
   - Spector and Mazzeo (1980) program effectiveness data
   - A US macroeconomic dataset
 * add four new Maximum Likelihood Estimators for models with a discrete
   dependent variables with examples

   - Logit
   - Probit
   - MNLogit (multinomial logit)
   - Poisson

*Sandbox*

 * add qqplot in sandbox.graphics
 * add sandbox.tsa (time series analysis) and sandbox.regression (anova)
 * add principal component analysis in sandbox.tools
 * add Seemingly Unrelated Regression (SUR) and Two-Stage Least Squares
   for systems of equations in sandbox.sysreg.Sem2SLS
 * add restricted least squares (RLS)


0.1.0b1
-------
 * initial release
