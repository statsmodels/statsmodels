:orphan:

=============
Release 0.9.0
=============

Release summary
---------------

statsmodels is using github to store the updated documentation which
is available under
https://www.statsmodels.org/stable for the last release, and
https://www.statsmodels.org/devel/ for the development version.


**Warning**

API stability is not guaranteed for new features, although even in
this case changes will be made in a backwards compatible way if
possible. The stability of a new feature depends on how much time it
was already in statsmodels master and how much usage it has already
seen.  If there are specific known problems or limitations, then they
are mentioned in the docstrings.


The list of pull requests for this release can be found on github
https://github.com/statsmodels/statsmodels/pulls?utf8=%E2%9C%93&q=is%3Apr+is%3Amerged+milestone%3A0.9
(The list does not include some pull request that were merged before
the 0.8 release but not included in 0.8.)


The Highlights
--------------

- statespace refactoring, Markov Switching Kim smoother
- 3 Google summer of code (GSOC) projects merged
  - distributed estimation
  - VECM and enhancements to VAR (including cointegration test)
  - new count models: GeneralizedPoisson, zero inflated models
- Bayesian mixed GLM
- Gaussian Imputation
- new multivariate methods: factor analysis, MANOVA, repeated measures
  within ANOVA
- GLM var_weights in addition to freq_weights
- Holt-Winters and Exponential Smoothing


What's new - an overview
------------------------

The following lists the main new features of statsmodels 0.9. In addition,
release 0.9 includes bug fixes, refactorings and improvements in many areas.

**base**
 - distributed estimation #3396  (Leland Bybee GSOC, Kerby Shedden)
 - optimization option scipy minimize #3193 (Roman Ring)
 - Box-Cox #3477 (Niels Wouda)
 - t_test_pairwise #4365 (Josef Perktold)

**discrete**
 - new count models (Evgeny Zhurko GSOC, Josef Perktold)
    - NegativeBinomialP #3832 merged in #3874
    - GeneralizedPoisson #3727 merged in  #3795
    - zero-inflated count models #3755 merged in #3908

 - discrete optimization improvements #3921, #3928 (Josef Perktold)
 - extend discrete margin when extra params, NegativeBinomial #3811
   (Josef Perktold)

**duration**
 - dependent censoring in survival/duration #3090 (Kerby Shedden)
 - entry times for Kaplan-Meier #3126 (Kerby Shedden)

**genmod**
 - Bayesian GLMM #4189, #4540 (Kerby Shedden)
 - GLM add var_weights #3692 (Peter Quackenbush)
 - GLM: EIM in optimization #3646 (Peter Quackenbush)
 - GLM correction to scale handling, loglike #3856 (Peter Quackenbush)

**graphics**
 - graphics HDR functional boxplot #3876 merged in #4049 (Pamphile ROY)
 - graphics Bland-Altman or Tukey mean difference plot
   #4112 merged in #4200 (Joses W. Ho)
 - bandwidth options in violinplots #4510 (Jim Correia)

**imputation**
 - multiple imputation via Gaussian model #4394, #4520 (Kerby Shedden)
 - regularized fitting in MICE #4319 (Kerby Shedden)

**iolib**
 - improvements of summary_coll #3702 merged #4064 (Natasha Watkins,
   Kevin Sheppard)

**multivariate**
 - multivariate: MANOVA, CanCorr #3327 (Yichuan Liu)
 - Factor Analysis #4161, #4156, #4167, #4214 (Yichuan Liu, Kerby Shedden,
   Josef Perktold)
 - statsmodels now includes the rotation code by ....

**regression**
 - fit_regularized for WLS #3581 (Kerby Shedden)

**stats**
 - Knockoff FDR # 3204 (Kerby Shedden)
 - Repeated measures ANOVA #3303 merged in #3663, #3838 (Yichuan Liu, Richard
   Höchenberger)
 - lilliefors test for exponential distribution #3837 merged in #3936 (Jacob
   Kimmel, Josef Perktold)

**tools**
 - quasi-random, Halton sequences #4104 (Pamphile ROY)

**tsa**
 - VECM #3246 (Aleksandar Karakas GSOC, Josef Perktold)
 - exog support in VAR, incomplete for extra results, part of VECM
   #3246, #4538 (Aleksandar Karakas GSOC, Josef Perktold)
 - Markov switching, Kim smoother #3141 (Chad Fulton)
 - Holt-Winters #3817 merged in #4176 (tvanzyl)
 - seasonal_decompose: trend extrapolation and vectorized 2-D #3031
   (kernc, Josef Perktold)
 - add frequency domain seasonal components to UnobservedComponents #4250
   (Jordan Yoder)
 - refactoring of date handling in tsa #3276, #4457 (Chad Fulton)
 - SARIMAX without AR, MA #3383  (Chad Fulton)

**maintenance**
 - switch to pytest #3804 plus several other PRs (Kevin Sheppard)
 - general compatibility fixes for recent versions of numpy, scipy and pandas


`bug-wrong`
~~~~~~~~~~~

A new issue label `type-bug-wrong` indicates bugs that cause that incorrect
numbers are returned without warnings.
(Regular bugs are mostly usability bugs or bugs that raise an exception for
unsupported use cases.)
see https://github.com/statsmodels/statsmodels/issues?q=is%3Aissue+label%3Atype-bug-wrong+is%3Aclosed+milestone%3A0.9

- scale in GLM fit_constrained, #4193 fixed in #4195
  cov_params and bse were incorrect if scale is estimated as in Gaussian.
  (This did not affect families with scale=1 such as Poisson)
- incorrect `pearson_chi2` with binomial counts, #3612 fixed as part of #3692
- null_deviance and llnull in GLMResults were wrong if exposure was used and
  when offset was used with Binomial counts.
- GLM Binomial in the non-binary count case used incorrect endog in recreating
  models which is
  used by fit_regularized and fit_constrained #4599.
- GLM observed hessian was incorrectly computed if non-canonical link is used,
  fixed in #4620
  This fix improves convergence with gradient optimization and removes a usually
  numerically small error in cov_params.
- discrete predict with offset or exposure, #3569 fixed in #3696
  If either offset or exposure are not None but exog is None, then offset and
  exposure arguments in predict were ignored.
- discrete margins had wrong dummy and count effect if constant is prepended,
  #3695 fixed in #3696
- OLS outlier test, wrong index if order is True, #3971 fixed in #4385
- tsa coint ignored the autolag keyword, #3966 fixed in #4492
  This is a backwards incompatible change in default, instead of fixed maxlag
  it defaults now to 'aic' lag selection. The default autolag is now the same
  as the adfuller default.
- wrong confidence interval in contingency table summary, #3822 fixed in #3830
  This only affected the summary and not the corresponding attribute.
- incorrect results in summary_col if regressor_order is used,
  #3767 fixed in #4271


Description of selected new feature
-----------------------------------

The following provides more information about a selected set of new features.

Vector Error Correction Model (VECM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The VECM framework developed during GSOC 2016 by Aleksandar Karakas adds support
for non-stationary cointegrated VAR processes to statsmodels.
Currently, the following topics are implemented

* Parameter estimation for cointegrated VAR
* forecasting
* testing for Granger-causality and instantaneous causality
* testing for cointegrating rank
* lag order selection.

New methods have been added also to the existing VAR model, and VAR has now
limited support for user provided explanatory variables.


New Count Models
----------------

New count models have been added as part of GSOC 2017 by Evgeny Zhurko.
Additional models that are not yet finished will be added for the next release.

The new models are:

* NegativeBinomialP (NBP): This is a generalization of NegativeBinomial that
  allows the variance power parameter to be specified in the range between 1
  and 2. The current NegativeBinomial support NB1 and NB2 which are two special
  cases of NBP.
* GeneralizedPoisson (GPP): Similar to NBP this allows a large range of
  dispersion specification. GPP also allow some amount of under dispersion
* ZeroInflated Models: Based on a generic base class, zeroinflated models
  are now available for Poisson, GeneralizedPoisson and NegativeBinomialP.

Generalized linear mixed models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Limited support for GLIMMIX models is now included in the genmod
module.  Binomial and Poisson models with independent random effects
can be fit using Bayesian methods (Laplace and mean field
approximations to the posterior).

Multiple imputation
~~~~~~~~~~~~~~~~~~~

Multiple imputation using a multivariate Gaussian model is now
included in the imputation module.  The model is fit via Gibbs
sampling from the joint posterior of the mean vector, covariance
matrix, and missing data values.  A convenience function for fitting a
model to the multiply imputed data sets and combining the results is
provided.  This is an alternative to the existing MICE (Multiple
Imputation via Chained Equations) procedures.

Exponential smoothing models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handling of indexes for time series models has been overhauled (#3272) to
take advantage of recent improvements in Pandas and to shift to Pandas much of
the special case handling (especially for date indexes) that had previously been
done in statsmodels. Benefits include more consistent behavior, a reduced
number of bugs from corner cases, and a reduction in the maintenance burden.

Although an effort was made to maintain backwards compatibility with this
change, it is possible that some undocumented corner cases that previously
worked will now raise warnings or exceptions.

State space models
~~~~~~~~~~~~~~~~~~

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
`VARMAX`), and they also make Bayesian estimation by Gibbs-sampling possible.

**Warning**: this will be the last version that includes the original state
space code and supports Scipy < 0.16. The next release will only include the
new state space code.

Unobserved components models: frequency-domain seasonals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unobserved components models now support modeling seasonal factors from a
frequency-domain perspective with user-specified period and harmonics
(introduced in #4250 by Jordan Yoder). This not only allows for multiple
seasonal effects, but also allows the representation of seasonal components
with fewer unobserved states. This can improve computational performance and,
since it allows for a more parsimonious model, may also improve the
out-of-sample performance of the model.


Major Bugs fixed
----------------

* see github issues for a list of bug fixes included in this release
  https://github.com/statsmodels/statsmodels/pulls?utf8=%E2%9C%93&q=is%3Apr+is%3Amerged+milestone%3A0.9+label%3Atype-bug
  https://github.com/statsmodels/statsmodels/pulls?q=is%3Apr+is%3Amerged+milestone%3A0.9+label%3Atype-bug-wrong

* Refitting elastic net regularized models using the `refit=True`
  option now returns the unregularized parameters for the coefficients
  selected by the regularized fitter, as documented. #4213

* In MixedLM, a bug that produced exceptions when calling
  `random_effects_cov` on models with variance components has been
  fixed.


Backwards incompatible changes and deprecations
-----------------------------------------------

* DynamicVAR and DynamicPanelVAR is deprecated and will be removed in
  a future version. It used rolling OLS from pandas which has been
  removed in pandas.

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
important contributions to general maintenance for this release came from

* Kevin Sheppard
* Peter Quackenbush
* Brock Mendel

and the general maintainer and code reviewer

* Josef Perktold

Additionally, many users contributed by participation in github issues and
providing feedback.

Thanks to all of the contributors for the 0.9 release (based on git log):

.. note::

    * Aleksandar Karakas
    * Alex Fortin
    * Alexander Belopolsky
    * Brock Mendel
    * Chad Fulton
    * ChadFulton
    * Christian Lorentzen
    * Dave Willmer
    * Dror Atariah
    * Evgeny Zhurko
    * Gerard Brunick
    * Greg Mosby
    * Jacob Kimmel
    * Jamie Morton
    * Jarvis Miller
    * Jasmine Mou
    * Jeroen Van Goey
    * Jim Correia
    * Joon Ro
    * Jordan Yoder
    * Jorge C. Leitao
    * Josef Perktold
    * Joses W. Ho
    * José Lopez
    * Joshua Engelman
    * Juan Escamilla
    * Justin Bois
    * Kerby Shedden
    * Kernc
    * Kevin Sheppard
    * Leland Bybee
    * Maxim Uvarov
    * Michael Kaminsky
    * Mosky Liu
    * Natasha Watkins
    * Nick DeRobertis
    * Niels Wouda
    * Pamphile ROY
    * Peter Quackenbush
    * Quentin Andre
    * Richard Höchenberger
    * Rob Klooster
    * Roman Ring
    * Scott Tsai
    * Soren Fuglede Jorgensen
    * Tom Augspurger
    * Tommy Odland
    * Tony Jiang
    * Yichuan Liu
    * ftemme
    * hugovk
    * kiwirob
    * malickf
    * tvanzyl
    * weizhongg
    * zveryansky

These lists of names are automatically generated based on git log, and may not
be complete.
