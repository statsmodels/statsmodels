:orphan:

=============
Release 0.8.0
=============

See also changes in the unreleased 0.7

Release summary
---------------

The main features of this release are several new time series models based
on the statespace framework, multiple imputation using MICE as well as many
other enhancements. The codebase also has been updated to be compatible with
recent numpy and pandas releases.

statsmodels is using now github to store the updated documentation which
is available under
https://www.statsmodels.org/stable for the last release, and
https://www.statsmodels.org/devel/ for the development version.

This is the last release that supports Python 2.6.


**Warning**

API stability is not guaranteed for new features, although even in this case
changes will be made in a backwards compatible way if possible. The stability
of a new feature depends on how much time it was already in statsmodels master
and how much usage it has already seen.
If there are specific known problems or limitations, then they are mentioned
in the docstrings.


The following major new features appear in this version.

Statespace Models
-----------------

Building on the statespace framework and models added in 0.7, this release
includes additional models that build on it.
Authored by Chad Fulton largely during GSOC 2015

Kalman Smoother
^^^^^^^^^^^^^^^

The Kalman smoother (introduced in #2434) allows making inference on the
unobserved state vector at each point in time using data from the entire
sample. In addition to this improved inference, the Kalman smoother is required
for future improvements such as simulation smoothing and the expectation
maximization (EM) algorithm.

As a result of this improvement, all state space models now inherit a `smooth`
method for producing results with smoothed state estimates. In addition, the
`fit` method will return results with smoothed estimates at the maximum
likelihood estimates.

Postestimation
^^^^^^^^^^^^^^

Improved post-estimation output is now available to all state space models
(introduced in #2566). This includes the new methods `get_prediction` and
`get_forecast`, providing standard errors and confidence intervals as well
as point estimates, `simulate`, providing simulation of time series following
the given state space process, and `impulse_responses`, allowing computation
of impulse responses due to innovations to the state vector.

Diagnostics
^^^^^^^^^^^

A number of general diagnostic tests on the residuals from state space
estimation are now available to all state space models (introduced in #2431).
These include:

* `test_normality` implements the Jarque-Bera test for normality of residuals
* `test_heteroskedasticity` implements a test for homoskedasticity of
  residuals similar to the Goldfeld-Quandt test
* `test_serial_correlation` implements the Ljung-Box (or Box-Pierce) test for
  serial correlation of residuals

These test statistics are also now included in the `summary` method output. In
addition, a `plot_diagnostics` method is available which provides four plots
to visually assess model fit.

Unobserved Components
^^^^^^^^^^^^^^^^^^^^^

The class of univariate Unobserved Components models (also known as structural
time series models) are now available (introduced in #2432). This includes as
special cases the local level model and local linear trend model. Generically
it allows decomposing a time series into trend, cycle, seasonal, and
irregular components, optionally with exogenous regressors and / or
autoregressive errors.

Multivariate Models
^^^^^^^^^^^^^^^^^^^

Two standard multivariate econometric models - vector autoregressive
moving-average model with exogenous regressors (VARMAX) and Dynamic Factors
models - are now available (introduced in #2563). The first is a popular
reduced form method of exploring the covariance in several time series, and the
second is a popular reduced form method of extracting a small number of common
factors from a large dataset of observed series.

Recursive least squares
^^^^^^^^^^^^^^^^^^^^^^^

A model for recursive least squares, also known as expanding-window OLS, is
now available in `statsmodels.regression` (introduced in #2830).

Miscellaneous
^^^^^^^^^^^^^

Other improvements to the state space framework include:

* Improved missing data handling #2770, #2809
* Ongoing refactoring and bug fixes in fringes and corner cases


Time Series Analysis
--------------------

Markov Switching Models
^^^^^^^^^^^^^^^^^^^^^^^

Markov switching dynamic regression and autoregression models are now
available (introduced in #2980 by Chad Fulton). These models allow regression
effects and / or autoregressive dynamics to differ depending on an unobserved
"regime"; in Markov switching models, the regimes are assumed to transition
according to a Markov process.

Statistics
^^^^^^^^^^

* KPSS stationarity, unit root test #2775 (N-Wouda)
* The Brock Dechert Scheinkman (BDS) test for nonlinear dependence is now
  available (introduced in #934 by Chad Fulton)
* Augmented Engle/Granger cointegration test (refactor hidden function) #3146 (Josef Perktold)


New functionality in statistics
-------------------------------

Contingency Tables #2418 (Kerby Shedden)

Local FDR, multiple testing #2297 (Kerby Shedden)

Mediation Analysis #2352 (Kerby Shedden)

Confidence intervals for multinomial proportions #3162 (Sebastien Lerique, Josef Perktold)

other:

* weighted quantiles in DescrStatsW #2707 (Kerby Shedden)


Duration
--------

Kaplan Meier Survival Function #2614 (Kerby Shedden)

Cumulative incidence rate function #3016 (Kerby Shedden)

other:

* frequency weights in Kaplan-Meier #2992 (Kerby Shedden)
* entry times for Kaplan-Meier #3126 (Kerby Shedden)
* intercept handling for PHReg #3095 (Kerby Shedden)


Imputation
----------

new subpackage in `statsmodels.imputation`

MICE #2076  (Frank Cheng GSOC 2014 and Kerby Shedden)

Imputation by regression on Order Statistic  #3019 (Paul Hobson)


Penalized Estimation
--------------------

Elastic net: fit_regularized with L1/L2 penalization has been added to
OLS, GLM and PHReg (Kerby Shedden)


GLM
---

Tweedie is now available as new family #2872 (Peter Quackenbush, Josef Perktold)

other:

* frequency weights for GLM (currently without full support) #
* more flexible convergence options #2803 (Peter Quackenbush)


Multivariate
------------

new subpackage that currently contains PCA

PCA was added in 0.7 to statsmodels.tools and is now in statsmodels.multivariate


Documentation
-------------

New doc build with latest jupyter and Python 3 compatibility (Tom Augspurger)


Other important improvements
----------------------------

several existing functions have received improvements


* seasonal_decompose: improved periodicity handling #2987 (ssktotoro ?)
* tools add_constant, add_trend: refactoring and pandas compatibility #2240 (Kevin Sheppard)
* acf, pacf, acovf: option for missing handling #3020 (joesnacks ?)
* acf, pacf plots: allow array of lags #2989 (Kevin Sheppard)
* pickling support for ARIMA #3412 (zaemyung)
* io SimpleTable (summary): allow names with special characters #3015 (tvanessa ?)
* tsa tools lagmat, lagmat2ds: pandas support #2310 #3042 (Kevin Sheppard)
* CompareMeans: from_data, summary methods #2754 (Valery Tyumen)
* API cleanup for robust, sandwich covariances #3162 (Josef Perktold)
* influence plot used swapped arguments (bug) #3158



Major Bugs fixed
----------------

* see github issues

While most bugs are usability problems, there is now a new label `type-bug-wrong`
for bugs that cause that silently incorrect numbers are returned.
https://github.com/statsmodels/statsmodels/issues?q=label%3Atype-bug-wrong+is%3Aclosed



Backwards incompatible changes and deprecations
-----------------------------------------------

* predict now returns a pandas Series if the exog argument is a DataFrame,
  including missing/NaN values
* PCA moved to multivariate compared to 0.7


Development summary and credits
-------------------------------

Besides receiving contributions for new and improved features and for bugfixes,
important contributions to general maintenance came from

* Kevin Sheppard
* Pierre Barbier de Reuille
* Tom Augsburger

and the general maintainer and code reviewer

* Josef Perktold

Additionally, many users contributed by participation in github issues and
providing feedback.

Thanks to all of the contributors for the 0.8 release (based on git log):

.. note::

   * Ashish
   * Brendan
   * Brendan Condon
   * BrianLondon
   * Chad Fulton
   * Chris Fonnesbeck
   * Christian Lorentzen
   * Christoph T. Weidemann
   * James Kerns
   * Josef Perktold
   * Kerby Shedden
   * Kevin Sheppard
   * Leoyzen
   * Matthew Brett
   * Niels Wouda
   * Paul Hobson
   * Pierre Barbier de Reuille
   * Pietro Battiston
   * Ralf Gommers
   * Roman Ring
   * Skipper Seabold
   * Soren Fuglede Jorgensen
   * Thomas Cokelaer
   * Tom Augspurger
   * ValeryTyumen
   * Vanessa
   * Yaroslav Halchenko
   * dhpiyush
   * joesnacks
   * kokes
   * matiumerca
   * rlan
   * ssktotoro
   * thequackdaddy
   * vegcev

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
