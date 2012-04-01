What it is
==========

Statsmodels is a Python package that provides a complement to scipy for statistical computations including descriptive statistics and estimation and inference for statistical models.

Main Features
=============

* linear regression models: Generalized least squares (including weighted least squares and
  least squares with autoregressive errors), ordinary least squares.
* glm: Generalized linear models with support for all of the one-parameter
  exponential family distributions.
* discrete: regression with discrete dependent variables, including Logit, Probit, MNLogit, Poisson, based on maximum likelihood estimators
* rlm: Robust linear models with support for several M-estimators.
* tsa: models for time series analysis
  - univariate time series analysis: AR, ARIMA
  - vector autoregressive models, VAR and structural VAR
  - descriptive statistics and process models for time series analysis
* nonparametric : (Univariate) kernel density estimators
* datasets: Datasets to be distributed and used for examples and in testing.
* stats: a wide range of statistical tests
  - diagnostics and specification tests
  - goodness-of-fit and normality tests
  - functions for multiple testing
  - various additional statistical tests
* iolib
  - Tools for reading Stata .dta files into numpy arrays.
  - printing table output to ascii, latex, and html
* miscellaneous models
* sandbox: statsmodels contains a sandbox folder with code in various stages of
  developement and testing which is not considered "production ready".
  This covers among others Mixed (repeated measures) Models, GARCH models, general method
  of moments (GMM) estimators, kernel regression, various extensions to scipy.stats.distributions,
  panel data models, generalized additive models and information theoretic measures.


Where to get it
===============

The master branch on GitHub is the most up to date code

    https://www.github.com/statsmodels/statsmodels

Source download of release tags are available on GitHub

    https://github.com/statsmodels/statsmodels/tags

Binaries and source distributions are available from PyPi

    http://pypi.python.org/pypi/statsmodels/


Installation from sources
=========================

See INSTALL.txt for requirements or see the documentation

    http://statsmodels.sf.net/devel/install.html


License
=======

Modified BSD (3-clause)


Documentation
=============

The official documentation is hosted on SourceForge

    http://statsmodels.sf.net/


Windows Help
============
The source distribution for Windows includes a htmlhelp file (statsmodels.chm).
This can be opened from the python interpreter ::

    >>> import statsmodels.api as sm
    >>> sm.open_help()


Discussion and Development
==========================

Discussions take place on our mailing list. 

    http://groups.google.com/group/pystatsmodels

We are very interested in feedback about usability and suggestions for improvements. 


Bug Reports
===========

Bug reports can be submitted to the issue tracker at

    https://github.com/statsmodels/statsmodels/issues
