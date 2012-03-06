What it is
==========

Statsmodels is a Python package that provides a complement to scipy for
statistical computations including descriptive statistics and
estimation of statistical models.

Main Features
=============

* regression: Generalized least squares (including weighted least squares and
  least squares with autoregressive errors), ordinary least squares.
* glm: Generalized linear models with support for all of the one-parameter
  exponential family distributions.
* discrete choice models: Poisson, probit, logit, multinomial logit
* rlm: Robust linear models with support for several M-estimators.
* tsa: Time series analysis models, including ARMA, AR, VAR
* nonparametric : (Univariate) kernel density estimators
* datasets: Datasets to be distributed and used for examples and in testing.
* PyDTA: Tools for reading Stata .dta files into numpy arrays.
* stats: a wide range of statistical tests
* sandbox: There is also a sandbox which contains code for generalized additive 
  models (untested), mixed effects models, cox proportional hazards model (both
  are untested and still dependent on the nipy formula framework), generating
  descriptive statistics, and printing table output to ascii, latex, and html.
  There is also experimental code for systems of equations regression,
  time series models, panel data estimators and information theoretic measures.  
  None of this code is considered "production ready".


Where to get it
===============

Development branches will be on Github. This is where to go to get the most
up to date code in the trunk branch. Experimental code is hosted here
in branches and in developer forks. This code is merged to master often. We 
try to make sure that the master branch is always stable.

https://www.github.com/statsmodels/statsmodels

Source download of stable tags will be on SourceForge.

https://sourceforge.net/projects/statsmodels/

or

PyPi: http://pypi.python.org/pypi/statsmodels/


Installation from sources
=========================

In the top directory, just do::

    python setup.py install

See INSTALL.txt for requirements or

http://statsmodels.sourceforge.net/

For more information.


License
=======

Simplified BSD


Documentation
=============

The official documentation is hosted on SourceForge.

http://statsmodels.sourceforge.net/

The sphinx docs are currently undergoing a lot of work. They are not yet
comprehensive, but should get you started.

Our blog will continue to be updated as we make progress on the code.

http://scipystats.blogspot.com


Windows Help
============
The source distribution for Windows includes a htmlhelp file (statsmodels.chm).
This can be opened from the python interpreter ::

>>> import statsmodels.api as sm
>>> sm.open_help()


Discussion and Development
==========================

All chatter will take place on the or scipy-user mailing list. We are very
interested in receiving feedback about usability, suggestions for improvements,
and bug reports via the mailing list or the bug tracker at

https://github.com/statsmodels/statsmodels/issues

There is also a google group at

http://groups.google.com/group/pystatsmodels

to discuss development and design issues that are deemed to be too specialized
for the scipy-dev/user list.


Python 3
========

statsmodels has been ported and tested for Python 3.2. Python 3
version of the code can be obtained by running 2to3.py over the entire
statsmodels source. The numerical core of statsmodels worked almost without
changes, however there can be problems with data input and plotting.
The STATA file reader and writer in iolib.foreign has not been ported yet.
And there are still some problems with the matplotlib version for Python 3
that was used in testing. Running the test suite with Python 3.2 shows some
errors related to foreign and matplotlib.
