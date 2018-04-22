:orphan:

===========
0.9 Release
===========

Release 0.9.0
=============

Release summary
---------------

Statsmodels is using now github to store the updated documentation which
is available under
http://www.statsmodels.org/stable for the last release, and
http://www.statsmodels.org/devel/ for the development version.


**Warning**

API stability is not guaranteed for new features, although even in
this case changes will be made in a backwards compatible way if
possible. The stability of a new feature depends on how much time it
was already in statsmodels master and how much usage it has already
seen.  If there are specific known problems or limitations, then they
are mentioned in the docstrings.


The following major new features appear in this version.

Generalized linear mixed models
-------------------------------

Limited support for GLIMMIX models is now included in the genmod
module.  Binomial and Poisson models with independent random effects
can be fit using Bayesian methods (Laplace and mean field
approximations to the posterior).

Multiple imputation
-------------------

Multiple imputation using a multivariate Gaussian model is now
included in the imputation module.  The model is fit via Gibbs
sampling from the joint posterior of the mean vector, covariance
matrix, and missing data values.  A convenience function for fitting a
model to the multiply imputed data sets and combining the results is
provided.  This is an alternative to the existing MICE (Multiple
Imputation via Chained Equations) procedures.

Exponential smoothing models
----------------------------

Exponential smoothing models are now available (introduced in #4176 by
Terence L van Zyl). These models are conceptually simple, decomposing a time
series into level, trend, and seasonal components that are constructed from
weighted averages of past observations. Nonetheless, they produce forecasts
that are competitive with more advanced models and which may be easier to
interpret.

Available models include:

- Simple exponential smoothing
- Holt's method
- Holt-Winters exponential smoothing

Improved time series index support
----------------------------------

Handling of indexes for time series models has been overhauled (#3272) to
take advantage of recent improvements in Pandas and to shift to Pandas much of
the special case handling (espcially for date indexes) that had previously been
done in Statsmodels. Benefits include more consistent behavior, a reduced
number of bugs from corner cases, and a reduction in the maintenance burden.

Although an effort was made to maintain backwards compatibility with this
change, it is possible that some undocumented corner cases that previously
worked will now raise warnings or exceptions.

State space models
------------------

The state space model infrastructure has been rewritten and improved (#2845).
New features include:

- Kalman smoother rewritten in Cython for substantial performance improvements
- Simulation smoother (Durbin and Koopman, 2002)
- Fast simulation of time series for any state space model
- Univariate Kalman filtering and smoothing (Koopman and Durbin, 2000)
- Collapsed Kalman filtering and smoothing (Jungbacker and Koopman, 2014)
- Optional computation of the lag-one state autocovariance
- Use of the Scipy BLAS functions for Cython interface if available
  (`scipy.linalg.cython_blas` for Scipy >= 0.16)

These features yield new features and improve performance for the existing
state space models (`SARIMAX`, `UnobservedComopnents`, `DynamicFactor`, and
`VARMAX), and they also make Bayesian estimation by Gibbs-sampling possible.

**Warning**: this will be the last version that includes the original state
space code and supports Scipy < 0.16. The next release will only include the
new state space code.

Unobserved components models: frequency-domain seasonals
--------------------------------------------------------

Unobserved components models now support modeling seasonal factors from a
frequency-domain perspective with user-specified period and harmonics. This not
only allows for multiple seasonal effects, but also allows the representation
of seasonal components with fewer unobserved states. This can improve
computational performance and, since it allows for a more parsimonious model,
may also improve the out-of-sample performance of the model.

Documentation
-------------



Other important improvements
----------------------------

* MICE (multiple imputation) can use regularized model fitters in the
  imputation step.


Major Bugs fixed
----------------

* see github issues

While most bugs are usability problems, there is now a new label `type-bug-wrong`
for bugs that cause that silently incorrect numbers are returned.
https://github.com/statsmodels/statsmodels/issues?q=label%3Atype-bug-wrong+is%3Aclosed

* Refitting elastic net regularized models using the `refit=True`
  option now returns the unregularized parameters for the coefficients
  selected by the regularized fitter, as documented. #4213

* In MixedLM, a bug that produced exceptions when calling
  `random_effects_cov` on models with variance components has been
  fixed.

Backwards incompatible changes and deprecations
-----------------------------------------------

* In MixedLM, names for the random effects variance and covariance
  parameters have changed from, e.g. G RE to G Var or G x F Cov.  This
  impacts summary output, and also may require modifications to user
  code that extracted these parameters from the fitted results object
  by name.

* In MixedLM, the names for the random effects realizations for
  variance components have been changed.  When using formulas, the
  random effect realizations are named using the column names produced
  by Patsy when parsing the formula.

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
