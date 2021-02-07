|PyPI Version| |Conda Version| |License| |Azure CI Build Status|
|Coveralls Coverage| |PyPI downloads| |Conda downloads|

About statsmodels
=================

statsmodels is a Python package that provides a complement to scipy for
statistical computations including descriptive statistics and estimation
and inference for statistical models.


Documentation
=============

The documentation for the latest release is at

https://www.statsmodels.org/stable/

The documentation for the development version is at

https://www.statsmodels.org/dev/

Recent improvements are highlighted in the release notes

https://www.statsmodels.org/stable/release/version0.9.html

Backups of documentation are available at https://statsmodels.github.io/stable/
and https://statsmodels.github.io/dev/.


Main Features
=============

* Linear regression models:

  - Ordinary least squares
  - Generalized least squares
  - Weighted least squares
  - Least squares with autoregressive errors
  - Quantile regression
  - Recursive least squares

* Mixed Linear Model with mixed effects and variance components
* GLM: Generalized linear models with support for all of the one-parameter
  exponential family distributions
* Bayesian Mixed GLM for Binomial and Poisson
* GEE: Generalized Estimating Equations for one-way clustered or longitudinal data
* Discrete models:

  - Logit and Probit
  - Multinomial logit (MNLogit)
  - Poisson and Generalized Poisson regression
  - Negative Binomial regression
  - Zero-Inflated Count models

* RLM: Robust linear models with support for several M-estimators.
* Time Series Analysis: models for time series analysis

  - Complete StateSpace modeling framework

    - Seasonal ARIMA and ARIMAX models
    - VARMA and VARMAX models
    - Dynamic Factor models
    - Unobserved Component models

  - Markov switching models (MSAR), also known as Hidden Markov Models (HMM)
  - Univariate time series analysis: AR, ARIMA
  - Vector autoregressive models, VAR and structural VAR
  - Vector error correction model, VECM
  - exponential smoothing, Holt-Winters
  - Hypothesis tests for time series: unit root, cointegration and others
  - Descriptive statistics and process models for time series analysis

* Survival analysis:

  - Proportional hazards regression (Cox models)
  - Survivor function estimation (Kaplan-Meier)
  - Cumulative incidence function estimation

* Multivariate:

  - Principal Component Analysis with missing data
  - Factor Analysis with rotation
  - MANOVA
  - Canonical Correlation

* Nonparametric statistics: Univariate and multivariate kernel density estimators
* Datasets: Datasets used for examples and in testing
* Statistics: a wide range of statistical tests

  - diagnostics and specification tests
  - goodness-of-fit and normality tests
  - functions for multiple testing
  - various additional statistical tests

* Imputation with MICE, regression on order statistic and Gaussian imputation
* Mediation analysis
* Graphics includes plot functions for visual analysis of data and model results

* I/O

  - Tools for reading Stata .dta files, but pandas has a more recent version
  - Table output to ascii, latex, and html

* Miscellaneous models
* Sandbox: statsmodels contains a sandbox folder with code in various stages of
  development and testing which is not considered "production ready".  This covers
  among others

  - Generalized method of moments (GMM) estimators
  - Kernel regression
  - Various extensions to scipy.stats.distributions
  - Panel data models
  - Information theoretic measures

How to get it
=============
The master branch on GitHub is the most up to date code

https://www.github.com/statsmodels/statsmodels

Source download of release tags are available on GitHub

https://github.com/statsmodels/statsmodels/tags

Binaries and source distributions are available from PyPi

https://pypi.org/project/statsmodels/

Binaries can be installed in Anaconda

conda install statsmodels


Installing from sources
=======================

See INSTALL.txt for requirements or see the documentation

https://statsmodels.github.io/dev/install.html

Contributing
============
Contributions in any form are welcome, including:

* Documentation improvements
* Additional tests
* New features to existing models
* New models

https://www.statsmodels.org/stable/dev/test_notes

for instructions on installing statsmodels in *editable* mode.

License
=======

Modified BSD (3-clause)

Discussion and Development
==========================

Discussions take place on the mailing list

https://groups.google.com/group/pystatsmodels

and in the issue tracker. We are very interested in feedback
about usability and suggestions for improvements.

Bug Reports
===========

Bug reports can be submitted to the issue tracker at

https://github.com/statsmodels/statsmodels/issues

.. |Azure CI Build Status| image:: https://dev.azure.com/statsmodels/statsmodels-testing/_apis/build/status/statsmodels.statsmodels?branch=master
   :target: https://dev.azure.com/statsmodels/statsmodels-testing/_build/latest?definitionId=1&branch=master
.. |Coveralls Coverage| image:: https://coveralls.io/repos/github/statsmodels/statsmodels/badge.svg?branch=master
   :target: https://coveralls.io/github/statsmodels/statsmodels?branch=master
.. |PyPI downloads| image:: https://img.shields.io/pypi/dm/statsmodels.svg?label=Pypi%20downloads
   :target: https://pypi.org/project/statsmodels/
.. |Conda downloads| image:: https://img.shields.io/conda/dn/conda-forge/statsmodels.svg?label=Conda%20downloads
   :target: https://anaconda.org/conda-forge/statsmodels/
.. |PyPI Version| image:: https://img.shields.io/pypi/v/statsmodels.svg
   :target: https://pypi.org/project/statsmodels/
.. |Conda Version| image:: https://anaconda.org/conda-forge/statsmodels/badges/version.svg
   :target: https://anaconda.org/conda-forge/statsmodels/
.. |License| image:: https://img.shields.io/pypi/l/statsmodels.svg
   :target: https://github.com/statsmodels/statsmodels/blob/master/LICENSE.txt
