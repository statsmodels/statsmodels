.. currentmodule:: scikits.statsmodels

************
Introduction
************

Background
----------

Scipy.stats.models was originally written by Jonathan Taylor.
For some time it was part of scipy but then removed from it. During
the Google Summer of Code 2009, stats.models was corrected, tested and
enhanced and released as a new package. Since then we have continued to
improve the existing models and added new statistical methods.


Current Status
--------------

statsmodels 0.3 is a pure python package, with one optional cython based
extension. However, future releases will depend on cython generated
extensions.

statsmodels includes:

  * regression: mainly OLS and generalized least squares, GLS
    including weighted least squares and least squares with AR
    errors.
  * glm: generalized linear models
  * rlm: robust linear models
  * discretemod: regression with discrete dependent variables, Logit, Probit,
    MNLogit, Poisson, based on maximum likelihood estimators
  * datasets: for examples and tests
  * univariate time series analysis: AR, ARIMA
  * vector autoregressive models
  * descriptive statistics and process models for time series analysis
  * diagnostics and specification tests
  * additional statistical tests and functions for multiple testing
  * miscellaneous models

statsmodels contains a sandbox folder, which includes some of the original
stats.models code that has not yet been rewritten and tested. The sandbox also
contains models and functions that we are currently developing. This code is
in various stages of development from early stages to almost finished, but
not sufficiently tested or with an API that is still in flux. Some of the code
in the advanced state covers among others GARCH models, general method of
moments (GMM) estimators, kernel regression and kernel density estimation, and
various extensions to scipy.stats.distributions.

The code is written for plain NumPy arrays so that statsmodels can be used
as a library for any kind of data structure users might have. However, in
order to make the data handling easier, some time series specific models
rely on pandas, and we have plans to integrate pandas in future releases of
statsmodels.

We have also included several datasets from the public domain and by
permission for tests and examples. The datasets are set up so that it is
easy to add more datasets.

Python 3
--------

scikits.statsmodels has been ported and tested for Python 3.2. Python 3
version of the code can be obtained by running 2to3.py over the entire
statsmodels source. The numerical core of statsmodels worked almost without
changes, however there can be problems with data input and plotting.
The STATA file reader and writer in iolib.foreign has not been ported yet.
And there are still some problems with the matplotlib version for Python 3
that was used in testing. Running the test suite with Python 3.2 shows some
errors related to foreign and matplotlib.

Testing
-------

Most results have been verified with at least one other statistical package: R,
Stata or SAS. The guiding principal for the initial rewrite and for continued
development is that all numbers have to be verified. Some statistical
methods are tested with Monte Carlo studies. While we strife to follow this
test driven approach, there is no guarantee that the code is bug-free and
always works. Some auxilliary function are still insufficiently tested, some
edge cases might not be correctly taken into account, and the possibility of
numerical problems is inherent to many of the statistical models. We
especially appreciate any help and reports for these kind of problems so we
can keep improving the existing models.




Looking Forward
---------------

We would like to invite everyone to give statsmodels a test drive, use it, and
report comments, possibilities for improvement and bugs to the statsmodels
mailing list http://groups.google.com/group/pystatsmodels or file tickets on our
issue tracker at https://github.com/statsmodels/statsmodels/issues

The source code is available from https://github.com/statsmodels/statsmodels.

Our plans for the future include improving the coverage of statistical
models, methods and tests that any basic statistics package should provide.
But the main direction for the expansion of statsmodels depends on the
requirements and interests of the developers and contributers.

The current maintainers are mostly interested in econometrics and time series
analysis, but we would like to invite any users or developers to contribute
their own extensions to existing models, or new models. To speed up
improvements that are waiting in the sandbox, any help with providing test
cases, reviewing or improving the code would be very appreciated.

Planned Extensions
~~~~~~~~~~~~~~~~~~

Two big changes that are planned for the next release will improve the
usability of statsmodels especially for interactive work.

* Metainformation about data and models: Currently the models essentially
  use no information about the design matrix and just treat it as numpy
  array.
* Merge Pandas into statsmodels which will provide a data structure and
  improved handling of time series data, together with additional time series
  specific models. (Wes McKinney)
* Formulas similar to R: This will provide a faster way to interactively
  define models and contrast matrices, and will provide additional
  information especially for categorical variables. (Nathaniel Smith)

Various models that are work in progress where the time to inclusion in
statsmodels proper will depend on the available developer time and interests:

Bayesian dynamic linear models (Wes)

more Kalman filter based time series analysis (Skipper)

New models (roughly in order of completeness):
general method of moments (GMM) estimators, kernel regression,
kernel density estimation, various extensions to scipy.stats.distributions,
GARCH models, copulas, system of equation models, panel data models,
more discrete choice models, mixed effects models, survival models.

New tests: multiple comparison, more diagnostics and outlier tests, additional
non-parametric tests

Resampling approaches like bootstrap and permutation for tests and estimator
statistics.


Code Stability
~~~~~~~~~~~~~~

The existing models are mostly settled in their user interface and we do not
expect many changes anymore. One area that will need adjustment is how
formulas and meta information are included. New models that have just been
included might require adjustments as we gain more experience and obtain
feedback by users. As we expand the range of models, we keep improving the
framework for different estimators and statistical tests, so further changes
will be necessary. In 0.3 we reorganized the internal location of the code and
import paths which will make future enhancements less interruptive.
Although there is no guarantee yet on API stability, we try to keep changes
that require adjustments by existing users to a minimal level.

Financial Support
-----------------

We are grateful for the financial support that we obtained for the
developement of scikits.statsmodels:

 Google `www.google.com <http://www.google.com/>`_ : two Google Summer of Code,
 GSOC 2009 and GSOC 2010

 AQR `www.aqr.com <http://www.aqr.com/>`_ : financial sponsor for the work on
 Vector Autoregressive Models (VAR) by Wes McKinney

We would also like to thank our hosting providers, `github
<http://github.com/>`_ for the public code repository, `sourceforge
<http://sourceforge.net/>`_ for hosting our documentation and `python.org
<http://python.org>`_ for making our downloads available on pypi.


Josef Perktold and Skipper Seabold,
Wes McKinney,
Mike Crow,
Vincent Davis,
