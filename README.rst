|Travis Build Status| |Appveyor Build Status| |Coveralls Coverage|

About Statsmodels
=================

Statsmodels is a Python package that provides a complement to scipy for
statistical computations including descriptive statistics and estimation
and inference for statistical models.


Documentation
=============

The documentation for the latest release is at

   http://www.statsmodels.org/stable/

The documentation for the development version is at

   http://www.statsmodels.org/dev/

Recent improvements are highlighted in the release notes

   http://www.statsmodels.org/stable/release/version0.8.html

Backups of documentation are available at http://statsmodels.github.io/stable/
and http://statsmodels.github.io/dev/.



Main Features
=============

* Linear regression models:

  - Ordinary least squares
  - Generalized least squares
  - Weighted least squares
  - Least squares with autoregressive errors
  - Quantile regression

* Mixed Linear Model with mixed effects and variance components
* GLM: Generalized linear models with support for all of the one-parameter
  exponential family distributions
* GEE: Generalized Estimating Equations for one-way clustered or longitudinal data
* Discrete models:

  - Logit and Probit
  - Multinomial logit (MNLogit)
  - Poisson regresion
  - Negative Binomial regression

* RLM: Robust linear models with support for several M-estimators.
* Time Series Analysis: models for time series analysis

  - Complete StateSpace modeling framework

    - Seasonal ARIMA and ARIMAX models
    - VARMA and VARMAX models
    - Dynamic Factor models

  - Markov switching models (MSAR), also known as Hidden Markov Models (HMM)
  - Univariate time series analysis: AR, ARIMA
  - Vector autoregressive models, VAR and structural VAR
  - Hypothesis tests for time series: unit root, cointegration and others
  - Descriptive statistics and process models for time series analysis

* Survival analysis:

  - Proportional hazards regression (Cox models)
  - Survivor function estimation (Kaplan-Meier)
  - Cumulative incidence function estimation

* Nonparametric statistics: (Univariate) kernel density estimators
* Datasets: Datasets used for examples and in testing
* Statistics: a wide range of statistical tests

  - diagnostics and specification tests
  - goodness-of-fit and normality tests
  - functions for multiple testing
  - various additional statistical tests

* Imputation with MICE and regression on order statistic
* Mediation analysis
* Principal Component Analysis with missing data
* I/O

  - Tools for reading Stata .dta files into numpy arrays.
  - Table output to ascii, latex, and html

* Miscellaneous models
* Sandbox: statsmodels contains a sandbox folder with code in various stages of
  developement and testing which is not considered "production ready".   This covers
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

    http://pypi.python.org/pypi/statsmodels/

Binaries can be installed in Anaconda

    conda install statsmodels

Development snapshots are also available in Anaconda (infrequently updated)

    conda install -c https://conda.binstar.org/statsmodels statsmodels

Installing from sources
=======================

See INSTALL.txt for requirements or see the documentation

    http://statsmodels.github.io/dev/install.html

License
=======

Modified BSD (3-clause)

Discussion and Development
==========================

Discussions take place on our mailing list.

    http://groups.google.com/group/pystatsmodels

We are very interested in feedback about usability and suggestions for
improvements.

Bug Reports
===========

Bug reports can be submitted to the issue tracker at

    https://github.com/statsmodels/statsmodels/issues

.. |Travis Build Status| image:: https://travis-ci.org/statsmodels/statsmodels.svg?branch=master
   :target: https://travis-ci.org/statsmodels/statsmodels
.. |Appveyor Build Status| image:: https://ci.appveyor.com/api/projects/status/gx18sd2wc63mfcuc/branch/master?svg=true
   :target: https://ci.appveyor.com/project/josef-pkt/statsmodels/branch/master
.. |Coveralls Coverage| image:: https://coveralls.io/repos/github/statsmodels/statsmodels/badge.svg?branch=master
   :target: https://coveralls.io/github/statsmodels/statsmodels?branch=master
