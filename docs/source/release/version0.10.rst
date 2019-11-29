:orphan:

==============
Release 0.10.0
==============

Release summary
===============

statsmodels is using github to store the updated documentation. Two version are available:

* `Stable <https://www.statsmodels.org/stable/>`_, the latest release
* `Development <https://www.statsmodels.org/devel/>`_, the latest build of the master branch

**Warning**

API stability is not guaranteed for new features, although even in
this case changes will be made in a backwards compatible way if
possible. The stability of a new feature depends on how much time it
was already in statsmodels master and how much usage it has already
seen.  If there are specific known problems or limitations, then they
are mentioned in the docstrings.

Stats
-----
**Issues Closed**: 1052
**Pull Requests Merged**: 469

The list of pull requests for this release can be found on `github
<https://github.com/statsmodels/statsmodels/pulls?utf8=%E2%9C%93&q=is%3Apr+is%3Amerged+milestone%3A0.10/>`_
(The list does not include some pull request that were merged before
the 0.9 release but not included in 0.9.)


The Highlights
==============

Generalized Additive Models
---------------------------

:class:`~statsmodels.gam.generalized_additive_model.GLMGam` adds support for Generalized additive models.

.. note::

    **Status: experimental**. This class has full unit test coverage for the core
    results with Gaussian and Poisson (without offset and exposure). Other
    options and additional results might not be correctly supported yet.
    (Binomial with counts, i.e. with n_trials, is most likely wrong in parts.
    User specified var or freq weights are most likely also not correct for
    all results.)

:class:`~statsmodels.gam.generalized_additive_model.LogitGam` adds a Logit version, although this is
unfinished. 

Conditional Models
------------------
Three conditional limited dependent variables models have been added:
:class:`~statsmodels.discrete.conditional_models.ConditionalLogit`,
:class:`~statsmodels.discrete.conditional_models.ConditionalMNLogit` and 
:class:`~statsmodels.discrete.conditional_models.ConditionalPoisson`. These are known
as fixed effect models in Econometrics. 

Dimension Reduction Methods
---------------------------
Three standard methods to perform dimension reduction when modeling data have been added:
:class:`~statsmodels.regression.dimred.SlicedInverseReg`,
:class:`~statsmodels.regression.dimred.PrincipalHessianDirections`, and
:class:`~statsmodels.regression.dimred.SlicedAverageVarianceEstimation`.

Regression using Quadratic Inference Functions (QIF)
----------------------------------------------------
Quadratic Inference Function, :class:`~statsmodels.genmod.qif.QIF`, improve the estimation of GEE models.

Gaussian Process Regression
---------------------------
:class:`~statsmodels.regression.process_regression.GaussianCovariance` implements Gaussian process
regression which is a nonparametric kernel-based method to model data.
:class:`~statsmodels.regression.process_regression.ProcessMLE` is a generic class that can be used
for other types of process regression. The results are returned in a
:class:`~statsmodels.regression.process_regression.ProcessMLEResults`.
:func:`~statsmodels.stats.correlation_tools.kernel_covariance`
provides a method that uses kernel averaging to estimate a multivariate covariance function.

Burg's Method
-------------
Burg's method, :func:`~statsmodels.regression.linear_model.burg`, provides an alternative estimator for the parameters
of AR models that is known to work well in small samples. It minimizes the forward and backward errors.

Time series Tools
-----------------
A number of common helper function for decomposing a time series have been added:
:func:`~statsmodels.tsa.innovations.arma_innovations.arma_innovations`, 
:func:`~statsmodels.tsa.stattools.innovations_algo`, and
:func:`~statsmodels.tsa.stattools.innovations_filter`. Two new PACF estimators have been added:
:func:`~statsmodels.tsa.stattools.levinson_durbin_pacf` and :func:`~statsmodels.tsa.stattools.pacf_burg`.

Other
-----
Knockoff effect estimation has been added for a many models:
:class:`~statsmodels.stats.knockoff_regeffects.RegModelEffects`,
:class:`~statsmodels.stats.knockoff_regeffects.CorrelationEffects`,
:class:`~statsmodels.stats.knockoff_regeffects.OLSEffects`,
:class:`~statsmodels.stats.knockoff_regeffects.ForwardEffects`, and 
:class:`~statsmodels.stats.knockoff_regeffects.OLSEffects`.

Influence functions are available for GLM and generic MLE models:
:class:`~statsmodels.stats.outliers_influence.GLMInfluence` and 
:class:`~statsmodels.stats.outliers_influence.MLEInfluence`.


What's new - an overview
========================

The following lists the main new features of statsmodels 0.10. In addition,
release 0.10 includes bug fixes, refactorings and improvements in many areas.

Submodules
----------

``base``
~~~~~~~~
- Add ``ModelWarning`` base class to avoid warning filter on standard UserWarning (:pr:`4712`)
- Add ultra-high screening with SCAD (:pr:`4683`)
- Add penalized mle scad (:pr:`4576`, :issue:`3677`, :issue:`2374`)
- Add score/LM conditional moment tests (:pr:`2096`)
- Fixed a bug which resulted in weights not being used in penalized models (:pr:`5762`, :issue:`4725`)
- Allow the constant index to be located even when ``hasconst=False`` (:pr:`5680`)
- Ensure ``mle_retvals`` is always set even when ``full_output=False`` (:pr:`5681`, :issue:`2752`)
- Fix a bug in Wald tests when testing a single constraint (:pr:`5684`, :issue:`5475`)
- Improve performance by skipping constant check when ``hasconst=True`` (:pr:`5698`)
- Deprecated ``scale`` parameter in the base model class (:pr:`5614`, :issue:`4598`)
- Fixed a bug that raised an error when a multi-index DataFrame was input into a model (:pr:`5634`, :issue:`5415`, :issue:`5414`)
- Fix bug in use of ``self.score`` in GenericLikelihoodModel (:pr:`5130`, :issue:`4453`)

``discrete``
~~~~~~~~~~~~
- Improve performance by only computing matrix_rank(exog) once in DiscreteModel.initialize (:pr:`4805`)
- Improve performance in discrete models by avoiding repeated calculations (:pr:`4515`)
- Add ``cov_type`` to summary of discrete models (:pr:`5672`, :issue:`4581`)
- Add conditional multinomial logit (:pr:`5510`)
- Add conditional logistic and Poisson regression (:pr:`5304`)

``genmod``
~~~~~~~~~~
- Fix arguments in poisson version of ``BayesMixedLM`` (:pr:`4809`)
- Ensure that column names are properly attached to the model (:pr:`4788`)
- Change ``cov_params`` in ``BayesMixedLM`` to act more like it does in other models (:pr:`4788`)
- Add missing predict and fit methods to ``BayesMixedGLM`` (:pr:`4702`)
- Add influence function support for GLM (:pr:`4732`, :issue:`4268`, :issue:`4257`)
- Fixed a bug in GEE where history was not saved (:pr:`5789`)
- Enable ``missing='drop'`` in GEE (:pr:`5771`)
- Improve score test to allow the submodel to be provided as a GEEResults object instead of as linear constraints (:pr:`5435`)
- Use GLM to get starting values for GEE (:pr:`5440`)
- Added regularized GEE (:pr:`5450`)
- Added Generalized Additive Models (GAM) (:pr:`5481`, :issue:`5370`, :issue:`5296`, :issue:`4575`, :issue:`2744`, :issue:`2435`)
- Added tweedie log-likelihood (:pr:`5521`)
- Added ridge regression by gradient for all GLM (:pr:`5521`)
- Added Tweedie EQL quasi-likelihood (:pr:`5543`)
- Allow ``dep_data`` to be specified using formula or names (:pr:`5345`)
- Fix a bug in stationary cov_struct for GEE (:pr:`5390`)
- Add QIC for GEE (:pr:`4909`)

``graphics``
~~~~~~~~~~~~
- Allow QQ plots using samples with different sizes (:pr:`5673`, :issue:`2896`, :issue:`3169`)
- Added examples of many graphics functions to the documentation (:pr:`5607`, :issue:`5309`)
- Fixed a bug in ``interaction_plot`` which lost information in a ``pd.Series`` index (:pr:`5548`)
- Remove change of global pickle method in functional plots (:pr:`4963`)

``imputation``
~~~~~~~~~~~~~~
- Add formula support to MI multiple imputation (:pr:`4722`)
- Saves the column names from ``pd.DataFrames`` and returns the imputed results as a DataFrame in ``BayesMI`` (:pr:`4722`)
- Fixed warnings in ``MICEData`` related to setting on copy (:pr:`5606`, :issue:`5431`)
- Allow results to be stored for multiple imputation (:pr:`5093`)
- Fixed a bug where MICEData sets initial imputation incorrectly (:pr:`5301`, :issue:`5254`)

``iolib``
~~~~~~~~~
- Deprecate ``StataReader``, ``StataWriter``, and ``genfromdta`` in favor of pandas equivalents (:pr:`5770`)
- Improve string escaping when exporting to LaTeX (:pr:`5683`, :issue:`5297`)
- Fixed a bug in ``summary2`` that ignored user float formatting  (:pr:`5655`, :issue:`1964`, :issue:`1965`)
- Remove ``$$`` from LaTeX output (:pr:`5588`,:issue:`5444`)

``multivariate``
~~~~~~~~~~~~~~~~
- Fixed a bug that only allowed ``MANOVA`` to work correctly when called using the formula interface (:pr:`5646`, :issue:`4903`, :issue:`5578`)
- Fix pickling bug in ``PCA`` (:pr:`4963`)

``nonparametric``
~~~~~~~~~~~~~~~~~
- Added input protection ``lowess` to ensure ``frac`` is always in bounds. (:pr:`5556`)
- Add check of inputs in ``KernelReg`` (:pr:`4968`, :issue:`4873`)

``regression``
~~~~~~~~~~~~~~
- Fix bug in  random effects covariance getter for ``MixedLM`` (:pr:`4704`)
- Add exact diffuse filtering for ``RecursiveLS`` (:pr:`4699`)
- Add Gaussian process regression (:pr:`4691`)
- Add linear restrictions to ``RecursiveLS`` (:pr:`4133`)
- Added regression with quadratic inference functions :class:`~statsmodels.genmod.qif.QIF` (:pr:`5803`)
- Allow mediation to be used with MixedLM as a mediator and/or outcome model (:pr:`5489`)
- Add square root LASSO (:pr:`5516`)
- Add dimension reduction regression methods: ``SlicedInverseReg``, ``PHD`` and ``SAVE`` (:pr:`5518`)
- Increased the number of methods available to optimize ``MixedLM`` models (:pr:`5551`)
- Added label to R2 when model is uncentered (:pr:`5083`, :issue:`5078`)
- Allow several optimizers to be tried in sequence for MixedLM (:pr:`4819`)
- Fix bug in Recursive LS with multiple constraints (:pr:`4826`)
- Fix a typo in ``ColinearityWarning`` (:pr:`4889`, :issue:`4671`)
- Add a finite check for ``_MinimalWLS`` (:pr:`4960`)
- Fix definition of R2 in ``GLS`` (:pr:`4967`, :issue:`1252`, :issue:`1171`)
- Add Burgs algorithm for estimating parameters of AR models (:pr:`5016`)

``sandbox``
~~~~~~~~~~~
- Add copulas (:pr:`5076`)

``stats``
~~~~~~~~~
- Implements a simple method of moments estimator of a spatial covariance in ``kernel_covariance`` (:pr:`4726`)
- Fixed a bug in multiple function in ``~statsmodels.stats.moment_helpers`` which prevents in-place modification of inputs (:pr:`5671`, :issue:`3362`, :issue:`2928`)
- Fixed a bug in contingency tables where shift was not correctly applied (:pr:`5654`, :issue:`3603`, :issue:`3579`)
- Added White's two-moment specification test with null hypothesis of homoskedastic and correctly specified(:pr:`5602`, :issue:`4721`)
- Added adjusted p-values for Tukey's HSD (:issue:`5418`, :pr:`5625`)
- Fixed a bug in ``medcouple`` that produced the incorrect estimate when there are ties in the data (:pr:`5397`, :issue:`5395`)
- Combine the real and knockoff features in init (:pr:`4920`)
- Modifying exog in-place leads to incorrect scaling (:pr:`4920`)
- Add Provide Knockoff+ (guaranteed to control FDR but slightly conservative) as well as Knockoff FDR (:pr:`4920`)
- Add RegModelEffects allows the user to specify which model is used for parameter estimation (:pr:`4920`)

``tools``
~~~~~~~~~
- Fixed a bug in ``group_sums`` that raised ``NameError`` (:pr:`5127`)

``tsa``
~~~~~~~
- Fix k_params in seasonal MAs (:pr:`4790`, :issue:`4789`)
- Fix prediction index in VAR predict (:pr:`4785`, :issue:`4784`)
- Standardized forecast error in state space when using Cholesky methods with partial missing data (:pr:`4770`)
- Add and fix VARMAX trend, exog. timing and polynomial trends (:pr:`4766`)
- Fix bug in exact diffuse filtering in complex data type case (:pr:`4743`)
- SARIMAX warns for non-stationary starting params (:pr:`4739`)
- Make arroots and maroots have consistent return type (:pr:`4559`)
- Add exact diffuse initialization to state space models (:pr:`4418`, :issue:`4042`)
- Allow concentrating scale out of log-likelihood in state space models (:pr:`3480`)
- Fixed a bug in ``coint_johansen`` that prevented it from running with 0 lags (:pr:`5783`)
- Improved performance in ``kpss`` using ``np.sum`` (:pr:`5774`)
- Enforce maximum number of lags in ``kpss`` (:pr:`5707`)
- Add ``arma_innovations`` to compute the innovations from an ARMA process (:pr:`5704`)
- Limit maximum lag length in ``adfuller`` so that model can always be estimated (:pr:`5699`, :issue:`5432`, :issue:`3330`)
- Added automatic data-dependent lag length selection in ``kpss`` (:pr:`5670`, :issue:`2781`, :issue:`5522`)
- Fixed a bug in ``VARMAX`` where the wrong form of the intercept was used when creating starting values (:pr:`5652`, :issue:`5651`)
- Fixed a bug ``sirf_errband_mc`` (:pr:`5641`, :issue:`5280`)
- Clarified error when input to ARMA is not a 1-d array (:pr:`5640`, :issue:`2575`)
- Improved the numerical stability of parameter transformation in ARIMA estimation (:pr:`5569`)
- Fixed a bug in the acf of a ``VAR`` which produced incorrect values (:pr:`5501`)
- Expose additional alternative estimation methods in ``pacf`` (:pr:`5153`, :issue:`3862`)
- Removed original implementation of Kalman Filter in favor of Cythonized version in ``statsmodels.tsa.statespace`` (:pr:`5171`)
- Issue warning when using ``VARResults.cov_params`` that it will become a method in the future (:pr:`5244`)
- Fix a bug in statespace models' ``predict`` that would fail when using row labels (:pr:`5250`)
- Allow ``summary`` even if filter_results=None, which happens after ``save`` and ``load`` (:pr:`5252`)
- Fixed a bug in sequential simulation in models with state_intercept (:pr:`5257`)
- Add an analytic version of ``arma_acovf`` (:pr:`5324`)
- Add a fast ARMA innovation algorithm and loglike computation (:pr:`5360`)
- Fix a bug in the Initialization of simulation smoother with exact diffuse initialization (:pr:`5383`)
- Fix bug in simulation smoothed measurement disturbance with FILTER_COLLAPSED (:pr:`4810`, :issue:`4800`)
- Improve SARIMAX for time series close to non-stationary (:pr:`4815`)
- Use Cython to improve speed of Exponential Smoothing models (:pr:`4845`)
- Fix a bug in ``arma_order_selection`` when data is passed in as a list (:pr:`4890`, :issue:`4727`)
- Add explicit exceptions in ARMA/ARIMA forecast with missing or wrong exog (:pr:`4915`, :issue:`3737`)
- Remove incorrect endog from results if constraints (:pr:`4921`)
- Add ``nlag`` argument to ``acovf`` (:pr:`4937`)
- Set reasonable default lags for acf/pacf plots (:pr:`4949`)
- Add innovations algorithm to convert acov to MA (:pr:`5042`)
- Add and innovations filter to filter for observations in a MA (:pr:`5042`)
- Fix a bug in initialization when simulating in state space models (:pr:`5043`)

``maintenance``
~~~~~~~~~~~~~~~
- Switch to standard setup.py so that ``pip install statsmodels`` can succeed in an empty virtual environment
- General compatibility fixes for recent versions of numpy, scipy and pandas
- Added new CI using Azure Pipelines (:pr:`5617`)
- Enable linting on travis to ensure code is up to standards (:pr:`4820`)
- Add coverage for Cython code (:pr:`4871`)
- Improve import speed (:pr:`5831`)
- Make all version of docs available (:pr:`5879`)

bug-wrong
---------

A new issue label `type-bug-wrong` indicates bugs that cause that incorrect
numbers are returned without warnings.
(Regular bugs are mostly usability bugs or bugs that raise an exception for
unsupported use cases.)
`see tagged issues <https://github.com/statsmodels/statsmodels/issues?q=is%3Aissue+label%3Atype-bug-wrong+is%3Aclosed+milestone%3A0.10/>`_

- :issue:`5475`
- :issue:`5316`


Major Bugs Fixed
================

See github issues for a list of bug fixes included in this release

- `Closed bugs <https://github.com/statsmodels/statsmodels/pulls?utf8=%E2%9C%93&q=is%3Apr+is%3Amerged+milestone%3A0.10+label%3Atype-bug/>`_
- `Closed bugs (wrong result) <https://github.com/statsmodels/statsmodels/pulls?q=is%3Apr+is%3Amerged+milestone%3A0.10+label%3Atype-bug-wrong/>`_


Development summary and credits
===============================

Besides receiving contributions for new and improved features and for bugfixes,
important contributions to general maintenance for this release came from

* Chad Fulton
* Brock Mendel
* Peter Quackenbush
* Kerby Shedden
* Kevin Sheppard

and the general maintainer and code reviewer

* Josef Perktold

Additionally, many users contributed by participation in github issues and
providing feedback.

Thanks to all of the contributors for the 0.10 release (based on git log):


* Amir Masoud Abdol
* Andrew Davis
* Andrew Kittredge
* Andrew Theis
* bertrandhaut
* bksahu
* Brock Mendel
* Chad Fulton
* Chris Snow
* Chris Down
* Daniel Saxton
* donbeo
* Emlyn Price
* equinaut
* Eric Larson
* Evgeny Zhurko
* fourpoints
* Gabriel Reid
* Harry Moreno
* Hauke Jürgen Mönck
* Hugo
* hugovk
* Huize Wang
* JarnoRFB
* Jarrod Millman
* jcdang
* Jefferson Tweed
* Josef Perktold
* jtweeder
* Julian Taylor
* Kerby Shedden
* Kevin Sheppard
* Loknar
* Matthew Brett
* Max Ghenis
* Ming Li
* Mitch Negus
* Michael Handley
* Moritz Lotze
* Nathan Perkins
* Nathaniel J. Smith
* Niklas H
* Peter Quackenbush
* QuentinAndre
* Ralf Gommers
* Rebecca N. Palmer
* Rhys Ulerich
* Richard Barnes
* RonLek
* Stefaan Lippens
* Tad seldovia
* thequackdaddy
* Tom Augspurger
* Torsten Wörtwein
* Varanelli
* xrr
* Yichuan Liu
* zveryansky
* 郭飞

These lists of names are automatically generated based on git log, and may not
be complete.

Merged Pull Requests
--------------------

Thie following Pull Requests were merged since the last release:


* :pr:`2096`: Score/LM conditional moment tests
* :pr:`3480`: ENH: State space: allow concentrating scale out of log-likelihood
* :pr:`4048`: Remove redundant code for dropped Python 2.6
* :pr:`4133`: ENH: Add linear restrictions to RecursiveLS
* :pr:`4316`: ensure MultinomialResults has J, K.  Get rid of unnecessary lmap usage
* :pr:`4322`: Make DiscreteResults Unchanging
* :pr:`4371`: catch the correct exception, make assertions not-pointless
* :pr:`4418`: ENH: State space: Exact diffuse initialization
* :pr:`4458`: De-duplicate a bunch of identical code
* :pr:`4468`: remove unused resetlist
* :pr:`4487`: Get rid of non-standard imports and one-line functions
* :pr:`4494`: Fix imports math.foo -->np.foo in vecm
* :pr:`4501`: xfail test instead of commenting it out
* :pr:`4515`: PERF: Simplify algebra in discrete_model
* :pr:`4559`: REF: make arroots and maroots have consistent return type
* :pr:`4560`: Document and cleanup bits of cython code
* :pr:`4576`: Penalized mle scad rebased2
* :pr:`4593`: DOC:ArmaProcess class documentation typo fix
* :pr:`4594`: TEST/DOC: SMW linalg routines documentation and test
* :pr:`4640`: BF: DataTimeIndex.to_datetime removed in pandas
* :pr:`4648`: BUG/TEST: Make pattern order for multiple imputation deterministic
* :pr:`4650`: DISCUSS/BLD: Update minimum versions.
* :pr:`4653`: REF/MAINT: avoid dict with pandas
* :pr:`4658`: BLD: Use older version of Pandas for docbuild
* :pr:`4683`: ENH: add ultra-high screening with SCAD
* :pr:`4686`: TEST: Docstring edits and variable name changes for clarity
* :pr:`4689`: PERF: Declare temporary output for hessian
* :pr:`4691`: ENH: Gaussian process regression
* :pr:`4692`: DOC: Add GLM varfuncs and weights notebook to documentation
* :pr:`4696`: Configure doctr
* :pr:`4698`: REF: Remove compatibility mode for state space
* :pr:`4699`: ENH: Exact diffuse filtering for RecursiveLS
* :pr:`4702`: BUG: Add missing predict and fit methods to BayesMixedGLM
* :pr:`4704`: BUG Fix random effects covariance getter for MixedLM
* :pr:`4712`: BUG: add ModelWarning base class to avoid warning filter on standard UserWarning.
* :pr:`4717`: TST: allclose instead of exact match for floats and use machine precision
* :pr:`4720`: fix syntax-like error
* :pr:`4722`: ENH: Add formula support to MI multiple imputation
* :pr:`4726`: ENH Kernel covariance
* :pr:`4728`: TST: Openblas appveyor fixes
* :pr:`4732`: ENH: add GLMInfluence
* :pr:`4736`: DOC: Make custom function take effect
* :pr:`4739`: REF: SARIMAX: only warn for non stationary starting params
* :pr:`4743`: BUG: state space: exact diffuse filtering in complex data type case
* :pr:`4750`: DOC: Fix indentation of math formulas
* :pr:`4753`: DOC: Add notebook on concentrated scale in ssm
* :pr:`4758`: DOC: Added missing notebooks to examples
* :pr:`4760`: CLN: Provide better name for pooled risk ratio
* :pr:`4763`: replace copy/pasted code with import
* :pr:`4766`:  BUG/ENH: VARMAX Fix trend / exog. timing. Add polynomial trends.
* :pr:`4767`: MAINT: gitignore univariate_diffuse pyx files.
* :pr:`4770`: BUG: State space: standardized forecast error when using Cholesky methods with partial missing data
* :pr:`4777`: MAINT: conda specify numpy-base
* :pr:`4785`: BUG: Get prediction index in VAR predict.
* :pr:`4786`: CLEAN: fix indentation by four typos
* :pr:`4788`: BUG: bayes mixed GLM maintenance
* :pr:`4790`: BUG: k_params if seasonal MA
* :pr:`4805`: Only compute matrix_rank(exog) once in DiscreteModel.initialize
* :pr:`4809`: BUG: fix arguments in poisson mixed model
* :pr:`4810`: BUG: simulation smoothed measurement disturbance with FILTER_COLLAPSED
* :pr:`4814`: CLEAN: Removed unnecessary and non-informative print
* :pr:`4815`: ENH/BUG: Improve SARIMAX for time series close to non-stationary
* :pr:`4819`: ENH: Allow several optimizers to be tried in sequence for MixedLM
* :pr:`4820`: Implement basic linting for Travis
* :pr:`4823`: Fix deprecation warnings
* :pr:`4826`: BUG/ENH: Recursive LS: fix bug w/ multiple constraints
* :pr:`4834`: Implement full flake8 checking for a subset of files in good condition
* :pr:`4835`: CLEAN: Fix tab indentation, lint for it
* :pr:`4842`: CLN: Flake8 fixups and linting for statespace files (but not tests)
* :pr:`4844`: CLN: Fully lint regime_switching
* :pr:`4845`: ENH: Improve speed in Exponential Smoothing
* :pr:`4853`: CLN/REF: Remove recarrays from datasets
* :pr:`4855`: BUG: Attach vc_names for mixed Poisson models
* :pr:`4858`: MAINT: Delete migrate_issues_gh
* :pr:`4859`: Fix some NameErrors, do not delete unused [...]
* :pr:`4861`: DOC: Fix small doc errors
* :pr:`4864`: CLN: fix and lint for W391 blank line at end of file
* :pr:`4869`: Update setup.cfg
* :pr:`4871`: BLD: Refactor Setup
* :pr:`4872`: MAINT: Remove nose and related references
* :pr:`4879`: CLN: Fix documentation for Levinson-Durbin
* :pr:`4883`: CLN: remove empty __main__ sections
* :pr:`4886`: CLN: Fully lint recursive_ls.py
* :pr:`4889`: REF: Rename ColinearityWarning
* :pr:`4890`: BUG: Add check to ensure array in arma order selection
* :pr:`4891`: BLD: Fix linting and move coverage
* :pr:`4893`: TST: Restore incorrectly disabled test
* :pr:`4895`: CLN: Fix and lint for misleading indentation E125,E129
* :pr:`4896`: CLN: Fix and lint for potential double-negatives E713,E714
* :pr:`4897`: CLN: Fix and lint for multiple spaces after keyword E271
* :pr:`4900`: CLN: Lint for missing whitespace around modulo operator E228,E401
* :pr:`4901`: CLN: Fix and lint for E124 closing bracket does not match visual indentation
* :pr:`4909`: ENH: QIC for GEE
* :pr:`4910`: CLN: Blank Lines E301,E302,E303,E305,E306 in examples, tools, sm.base
* :pr:`4911`: MAINT: Remove future errors and warnings
* :pr:`4912`: BLD: Rebased ci improvements
* :pr:`4913`: TST: Add a fixture to close all plots
* :pr:`4914`: CLN: Blanks E301,E302,E303,E305,E306 in tsa
* :pr:`4915`: ENH: explicit exceptions in ARMA/ARIMA forecast with missing or wrong exog
* :pr:`4920`: BUG/ENH: Two bug fixes and several enhancements to knockoff filter (regression fdr)
* :pr:`4921`: BUG: remove faux endog from results if constraints
* :pr:`4924`: CLN: E242 space after tab, enforce all passing rules
* :pr:`4925`: CLN: Enforce E721, use isinstance
* :pr:`4926`: CLN: Enforce E306, blank lines in nested funcs
* :pr:`4927`: CLN: Enforce E272, multiple spaces
* :pr:`4929`: BLD: Add linting for any new files
* :pr:`4933`: Remove unused patsy import in quantile_regression.ipynb
* :pr:`4937`: ENH: Add nlag argument to acovf
* :pr:`4941`: MAINT: remove exact duplicate file datamlw.py
* :pr:`4943`: TST: Relax tolerance on failing test
* :pr:`4944`: BLD: Add pinned numpy on appveyor
* :pr:`4949`: BUG: Set default lags for acf/pacf plots
* :pr:`4950`: DOC: Fix small typo in unit root testing example
* :pr:`4953`: DOC: Fix nagging issues in docs
* :pr:`4954`: BUG: disallow use_self=False
* :pr:`4959`: DOC: Clean up tsa docs
* :pr:`4960`: BUG: Add finite check for _MinimalWLS
* :pr:`4963`: BUG: Remove change of global pickle method
* :pr:`4967`: BUG: Fix definition of GLS r2
* :pr:`4968`: BUG: Check inputs in KernelReg
* :pr:`4971`: DOC: Switch to https where used
* :pr:`4972`: MAINT/CLN Remove .bzrignore
* :pr:`4977`: [BUG/MAINT] Fix NameErrors caused by missing kwargs
* :pr:`4978`: [MAINT/Test] skip test instead of mangling name in test_generic_methods
* :pr:`4979`: [MAINT/TST] remove np.testing.dec unused imports (nose dependency)
* :pr:`4980`: [MAINT/TST] skip/xfail tests instead of mangling/commenting-out in genmod, regression
* :pr:`4981`: [MAINT] Remove info.py
* :pr:`4982`: DOC Fix typo Parameters-->Parameters
* :pr:`4983`: [TST] xfail/skip instead of commenting-out/mangling discrete tests
* :pr:`4984`: [TST/DOC] make commented-out code in tests/results into readable docs
* :pr:`4985`: [TST/DOC] Make test comments more readable
* :pr:`4986`: [MAINT/TST] turn commented-out code into readable docs in results_arma
* :pr:`4987`: [TST/MAINT] turn commented-out code into readable docs in results_ar,…
* :pr:`4988`: [TST/MAINT] de-duplicate get_correction_factor code
* :pr:`4989`: [MAINT/CLN] Remove code made unusable due to license issue
* :pr:`4990`: [MAINT/CLN] remove numdiff  __main__ section explicitly marked as scratch work
* :pr:`4993`: [TST/CLN] Turn decorators __main__ section into tests
* :pr:`4995`: [TST] make tools.linalg __main__ section into tests
* :pr:`4998`: [CLN/TST] Follow instructions to remove function
* :pr:`4999`: [MAINT] remove wrappers.py
* :pr:`5000`: [MAINT] update compat to remove unusable shims e.g. py26
* :pr:`5002`: [MAINT] add missing import
* :pr:`5003`: MAINT: fix invalid exception messages
* :pr:`5005`: [MAINT] remove unused imports in examples+tools
* :pr:`5007`: MAINT: unused imports in robust
* :pr:`5011`: [MAINT] remove text file relics from scikits/statsmodels
* :pr:`5012`: [MAINT/TST] move misplaced results files in regressions/tests
* :pr:`5013`: [MAINT] fix typo deprecated-->deprecated
* :pr:`5014`: [MAINT] typo in __init__ signature
* :pr:`5015`: [MAINT] move misplaced test_tsa_indexes
* :pr:`5016`: ENH: Burgs algorithm
* :pr:`5020`: MAINT: fix incorrect docstring summary-->summary2
* :pr:`5021`: MAINT: fix typo duplicated References in docstring
* :pr:`5024`: MAINT: silenced as_pandas warnings in documentation
* :pr:`5027`: MAINT: remove functions duplicated from scipy
* :pr:`5029`: MAINT: strict linting for sm.stats files _close_ to already passing
* :pr:`5040`: MAINT: clean up x13.py, delete main
* :pr:`5042`: ENH: Add innovations algorithm
* :pr:`5043`: BUG: Initialization when simulating
* :pr:`5045`: MAINT: strict linting for tsa.statespace.tests.results
* :pr:`5057`: BUG: Correct check for callable
* :pr:`5058`: BUG: Do not use mutable default values
* :pr:`5059`: BLD: Add line displaying CPU info to CI
* :pr:`5065`: TST: Fix incorrect assertion
* :pr:`5070`: MAINT: remove file that just says to remove it
* :pr:`5071`: MAINT: remove example file corresponding to removed module
* :pr:`5074`: MAINT: strict lint test_var.py
* :pr:`5075`: MAINT: strict linting test_univariate.py
* :pr:`5076`: ENH: more work on copula (deriv, classes)
* :pr:`5079`: MAINT: linting statespace tests
* :pr:`5080`: FIX failure caused by #5076
* :pr:`5083`: ENH: Add "(uncentered)" after rsquared label in .summary, .summary2 when appropriate
* :pr:`5086`: TST: parametrize tests instead of using for loops
* :pr:`5088`: DOC: Add javascript to link to other doc versions
* :pr:`5090`: MAINT: Chrome does not like having a secure link with an unsecure image
* :pr:`5093`: Allow results to be stored for multiple imputation
* :pr:`5096`: ENH remove unneeded restriction on QIC (GEE)
* :pr:`5099`: MAINT: fix and lint for W292 newline at end of file
* :pr:`5103`: BUG: fix missing new_branch_dir arg in upload_pdf
* :pr:`5105`: BUG/DOC: Description of k_posdef
* :pr:`5114`: MAINT: many but not all trailing whitespace
* :pr:`5119`: CLN: remove unused imports in tools, sm.tsa
* :pr:`5120`: BUG: Ensure internal tester exits with error if needed
* :pr:`5121`: MAINT: Avoid star imports
* :pr:`5122`: MAINT: Modernize R-->py script, lint output
* :pr:`5123`: CLN: Move results files to a location that is copied to install
* :pr:`5124`: MAINT: fix generated double whitespace
* :pr:`5127`: BUG: Fix NameError in grouputils, make __main__ into tests
* :pr:`5130`: BUG: incorrect self.score in GenericLikelihoodModel; closes #4453
* :pr:`5133`: TST: apply stacklevel to warning in Family.__init__
* :pr:`5135`: MAINT: Fix warnings
* :pr:`5136`: TST: improve testing util functions; de-duplicate
* :pr:`5138`: CLN: Use cache_readonly instead of OneTimeProperty
* :pr:`5141`: MAINT: Delete bspline source files
* :pr:`5143`: ENH/BUG Bootstrap clone rebased
* :pr:`5146`: Clean up the smf namespace
* :pr:`5148`: REF/TST: add seed to hdrboxplot, Use random order in pytest
* :pr:`5149`: TST: Theil test randomseed
* :pr:`5152`: REF: Use iterative cumsum_n
* :pr:`5153`: ENH: Add additional options for pacf ols
* :pr:`5156`: TST: Remove __main__ sections in tests
* :pr:`5162`: TST: Fix incorrect test closes #4325
* :pr:`5164`: BF: drop tolerance of a zero_constrained test
* :pr:`5165`: MAINT: Add decorator for tests that use matplotlib
* :pr:`5166`: DOC: Fix section title in QIC
* :pr:`5167`: TST/BUG: Fix missing SkipTest
* :pr:`5170`: DEPR: Remove items deprecated in previous versions
* :pr:`5171`: MAINT: Remove kalmanf StateSpace code supplanted by tsa.statespace
* :pr:`5176`:  TST: Fix random generation issue
* :pr:`5177`: DOC: Improve Holt Winters documentation
* :pr:`5178`: TST: Fix scale in test
* :pr:`5180`: TST: Change assert_approx_equal to assert_allclose
* :pr:`5184`: TST: parametrize tests in test_lme
* :pr:`5188`: BLD/TST: Add coverage for Cython files
* :pr:`5191`: MAINT: Remove selected __main__ sections
* :pr:`5192`: MAINT: Fix incorrect pass statements
* :pr:`5193`: MAINT: raise specific exceptions instead of just Exception
* :pr:`5194`: MAINT: fix incorrect TypeError --> ValueError
* :pr:`5195`: BLD: Include License in Wheel
* :pr:`5196`: TST: Set seed when using basin hopping
* :pr:`5198`: TST/CLN/BUG: Fix corr nearest factor
* :pr:`5200`: TST: Alter test condition due to parameter scale
* :pr:`5201`: TST/CLN: test_arima_exog_predict, Rescale data to avoid convergence issues
* :pr:`5203`: BUG: raise instead of return ValueError
* :pr:`5204`: MAINT: Avoid/Fix FutureWarnings
* :pr:`5207`: TST: Ensure random numbers are reproducible
* :pr:`5208`: TST/CLN: Tighten tol to reduce spurious test failure
* :pr:`5210`: BLD: Ensure master is available when linting
* :pr:`5211`: MAINT: Import instead of copy/pasting utils
* :pr:`5213`: MAINT: Move misplaced duration results files
* :pr:`5214`: MAINT: remove example-like file that could never run
* :pr:`5217`: MAINT: Remove outdated pandas compat shims
* :pr:`5218`: MAINT: Move misplaced genmod results files
* :pr:`5219`: fixed typo
* :pr:`5222`: MAINT: fully lint formula
* :pr:`5223`: MAINT: fully lint compat
* :pr:`5224`: REF: raise early on invalid method
* :pr:`5227`: MAINT: docstring and whitespace fixups
* :pr:`5228`: DOC: Fix many small errors in examples
* :pr:`5230`: DOC: Fix small doc build errors
* :pr:`5232`: TST: mark smoketests
* :pr:`5237`: TST: Add mac testing
* :pr:`5239`: BLD/TST: Add platform-specific skips to CI testing
* :pr:`5240`: MAINT: remove cythonize.py made unnecessary by #4871
* :pr:`5242`: DOC: Update release instructions [skip ci]
* :pr:`5244`: DEPR: warn that VARResults.cov_params will become method
* :pr:`5246`: DOC: Added documentation of elements in anova_lm
* :pr:`5248`: DOC: Revert incorrect docstring change [skip ci]
* :pr:`5249`: MAINT: Add Script to convert notebooks
* :pr:`5250`: BUG/TST: TSA models: _get_index_label_loc failed when using row labels.
* :pr:`5251`: DOC: Use correct “autoregressive” in docstring
* :pr:`5252`: BUG: Allow `summary` even if filter_results=None (e.g. after `save`, `load`
* :pr:`5257`: BUG: Sequential simulation in models with state_intercept
* :pr:`5260`: MAINT: avoid pandas FutureWarning by checking specific condition
* :pr:`5262`: MAINT: fix typos in pca, wrap long lines
* :pr:`5263`: BLD: Only unshallow when required
* :pr:`5265`: MAINT: Prefer signature over formatargspec
* :pr:`5267`: MAINT: implement _wrap_derivative_exog for de-duplication
* :pr:`5269`: MAINT: De-duplicate code in iolib.summary
* :pr:`5272`: WIP/MAINT: Identify defunct code in summary methods
* :pr:`5273`: Fix incorrect parameter name in docstring
* :pr:`5274`: MAINT: remove self.table pinning
* :pr:`5275`: ENH/BUG Modify GEE indexing to remove numpy warnings
* :pr:`5277`: DOC: Clarify/fix docs on GLM scale estimation for negative binomial
* :pr:`5292`: DOC: Remove only_directive
* :pr:`5295`: TST: Added random seed to test_gee and verified working
* :pr:`5300`: DOC fix docstring in stattools.py
* :pr:`5301`: BUG: MICEData sets initial imputation incorrectly
* :pr:`5304`: ENH: conditional logistic and Poisson regression
* :pr:`5306`: DOC: Workarounds to fix docbuild
* :pr:`5308`: REF: Collect covtype descriptions, de-duplicate normalization func
* :pr:`5314`: DOC: minor fix on documentation on Durbin Watson test
* :pr:`5322`: DOC: Move magic
* :pr:`5324`: ENH: analytic version of arma_acovf
* :pr:`5325`: BUG/TST: Fix innovations_filter, add test vs Kalman filter
* :pr:`5335`: MAINT: eliminate some pytest warnings
* :pr:`5345`: ENH: Allow dep_data to be specified using formula or names
* :pr:`5348`: Set python3 as interpreter for doc tools
* :pr:`5352`: CLN: Fix F901 and E306 mixups
* :pr:`5353`: CLN: W605 fixups in vector_ar
* :pr:`5359`: BUG: raise correct error
* :pr:`5360`: ENH: Fast ARMA innovation algorithm and loglike computation
* :pr:`5369`: MAINT: disable pytest minversion check (broken in pytest 3.10.0)
* :pr:`5383`: BUG: Initialization of simulation smoother with exact diffuse initialization
* :pr:`5390`: BUG/ENH: modify stationary cov_struct for GEE
* :pr:`5397`: BUG: Fix medcouple with ties
* :pr:`5399`: CLN: Fix some invalid escapes
* :pr:`5421`: CLN: informative names for test functions
* :pr:`5424`: MAINT: conda-forge use gcc7
* :pr:`5426`: Misspelling in the documentation proportions_ztest
* :pr:`5435`: ENH Score test enhancements for GEE
* :pr:`5440`: ENH: Use GLM to get starting values for GEE
* :pr:`5449`: ENH/DOC: Added linting instruction in CONTRIBUTING.rst
* :pr:`5450`: ENH: regularized GEE
* :pr:`5462`: Fixed broken link for Guerry Dataset
* :pr:`5471`: Fix broken link
* :pr:`5481`: ENH: Generalized Additive Models and splines (Gam 2744 rebased4)
* :pr:`5484`: DOC: fix gam.rst
* :pr:`5485`: MAINT: Travis fixes
* :pr:`5489`: ENH: Mediation for Mixedlm
* :pr:`5494`: BUG: Bad escapes
* :pr:`5497`: Fix typo in docstring
* :pr:`5501`: BUG: Correct error in VAR ACF
* :pr:`5510`: ENH Conditional multinomial logit
* :pr:`5513`: DOC: Fix spelling
* :pr:`5516`: ENH square root lasso
* :pr:`5518`: ENH dimension reduction regression
* :pr:`5521`: ENH: Tweedie log-likelihood (+ridge regression by gradient for all GLM)
* :pr:`5532`: DOC/ENH Docstring updates for clogit
* :pr:`5541`: DOC: Describe binomial endog formats
* :pr:`5542`: BUG/TEST: py27 needs slacker tolerances
* :pr:`5543`: BUG: Tweedie EQL quasi-likelihood
* :pr:`5548`: keep index of series when recoding a series
* :pr:`5551`: ENH: extend mixedlm optimizer attempts
* :pr:`5556`: Update _smoothers_lowess.pyx
* :pr:`5566`: Add project_urls to setup
* :pr:`5567`: Correct a spell mistake
* :pr:`5569`: ENH: Improve numerical stability of _ar_transparams, _ar_invtransparams
* :pr:`5582`: Jbrockmendel w605b
* :pr:`5583`: MAINT: Set language level for Cython
* :pr:`5584`: MAINT: Remov deprecation issues
* :pr:`5586`: DOC: Add issue and pr templates
* :pr:`5587`: MAINT: Resolve additional deprecations
* :pr:`5588`: BUG: Replace $$ in generated LaTeX
* :pr:`5589`: DOC: Updated the `for all i, j`
* :pr:`5590`: MAINT: Reorder travis so that legacy fails early
* :pr:`5591`: Jbrockmendel manywarns3
* :pr:`5592`: Jbrockmendel pversion
* :pr:`5593`: MAINT: remove never-needed callable and never-used compat functions
* :pr:`5594`: TST: Ensure test is identical in all runs
* :pr:`5595`: MAINT: Remove warnings from tests
* :pr:`5596`: TST: Explicitly set seed in basinhopping
* :pr:`5597`: MAINT: Remove unavailable imports
* :pr:`5599`: DOC: More emphasis and fix reference
* :pr:`5600`: TST: Relax tolerance for OpenBlas issue
* :pr:`5601`: Update mixed_linear.rst
* :pr:`5602`: ENH: White spec test (clean commit for PR 4721)
* :pr:`5604`: MAINT: Update template to encourage master check
* :pr:`5605`: Guofei9987 modify comments proportion confint
* :pr:`5606`: Mattwigway mice setting with copy warning
* :pr:`5607`: Jtweeder graphics addgraphics
* :pr:`5611`: BUG: Stop hardcoding parameters in results
* :pr:`5612`: MAINT: Ensure no warnings are produced by foreign
* :pr:`5613`: DOC: Improve PR template [skip ci]
* :pr:`5614`: MAINT: Deprecate scale in test function
* :pr:`5615`: Thequackdaddy docs
* :pr:`5616`: Bulleted list and minor typos in ttest_ind
* :pr:`5617`: CI: Implement azure-pipelines with multi-platform support
* :pr:`5621`: CLN: simplify lint configuration, fix some invalid escapes
* :pr:`5622`: DOC: Restore import
* :pr:`5625`: Andrew d davis tukey pvals
* :pr:`5626`: MAINT: Improve user-facing error message
* :pr:`5627`: BLD: Remove redundant travis config
* :pr:`5628`: MAINT: Relax tolerance on OSX only
* :pr:`5630`: MAINT: Enable xdist on azure
* :pr:`5631`: MAINT: Allow webuse fail
* :pr:`5633`: TST: change skip to xfail for test_compare_numdiff on OSX
* :pr:`5634`: Gabrielreid pandas multiindex handling bug
* :pr:`5635`: MAINT: Add a codecov config file
* :pr:`5636`: DOC: Update badges [skip ci]
* :pr:`5637`: CLN: strict linting for tools directory
* :pr:`5638`: MAINT: remove file with note to remove in 0.5.0
* :pr:`5640`: ENH: Improve error when ARMA endog is not 1d
* :pr:`5641`: Josef pkt svar irf errband 5280
* :pr:`5642`: TST: Relax tolerance on OSX for OpenBlas issues
* :pr:`5643`: MAINT: Consolidate platform checks
* :pr:`5644`: CLN/DOC: Remove unused module, vbench references
* :pr:`5645`: TST: Allow network failure in web tests
* :pr:`5646`: BUG: Fix MANOVA when not using formulas
* :pr:`5647`: TST: Adjust test_irf atol
* :pr:`5648`: BUG: Replace midrule with hline
* :pr:`5649`: CLN: strict linting for robust/tests directory
* :pr:`5650`: MAINT: Fix error in lint script
* :pr:`5652`: ENH/BUG: Use intercept form of trend / exog in VARMAX start params (not mean form)
* :pr:`5653`: MAINT: Reformat exceptions
* :pr:`5654`: Evgenyzhurko fix contingency table
* :pr:`5655`: BUG: summary2 use float_format when creating `_simple_tables` see #1964
* :pr:`5656`: BLD: Add linting to azure
* :pr:`5657`: TST: Protect multiprocess using test
* :pr:`5658`: BLD: Match requirements in setup and requirements
* :pr:`5659`: TST: Allow corr test to fail on Win32
* :pr:`5660`: MAINT: Fix make.bat [skip ci]
* :pr:`5661`: TST: Relax test tolerance on OSX
* :pr:`5662`: TST: Protect multiprocess on windows
* :pr:`5663`: MAINT: Add test runners
* :pr:`5664`: CLN: Fix and lint for E703 statements ending in semicolon
* :pr:`5666`: TST: Relax tolerance for irf test on windows
* :pr:`5667`: TST: Adjust tol and reset random state
* :pr:`5668`: TST: Adjust tolerance for test on windows
* :pr:`5669`: MAINT: Remove unused code
* :pr:`5670`: Jim varanelli issue2781
* :pr:`5671`: BUG: fix stats.moment_helpers inplace modification
* :pr:`5672`: ENH: Add cov type to summary for discrete models
* :pr:`5673`: ENH: Allow comparing two samples with different sizes
* :pr:`5675`: CLN: strict linting for emplike/tests
* :pr:`5679`: DOC: Clarify that predict expects arrays in dicts [skip ci]
* :pr:`5680`: ENH: Allow const idx to be found
* :pr:`5681`: BUG: Always set mle_retvals
* :pr:`5683`: BUG: Escape strings for latex output
* :pr:`5684`: BUG: fix df in summary for single constraint in wald_test_terms
* :pr:`5685`: Spelling
* :pr:`5686`: DOC: Fix parameter description in weightstats
* :pr:`5691`: MAINT: Near-duplicate example file, remove dominated version
* :pr:`5693`: CLN: Fix invalid escapes where possible
* :pr:`5694`: MAINT: Fix NameErrors in correlation_structures
* :pr:`5695`: MAINT: remove NameError-having version of levinson_durbin, just keep …
* :pr:`5696`: CLN: remove identical functions from garch
* :pr:`5697`: CLN: strict linting for examples/
* :pr:`5698`: PERF: Avoid implicit check when hasconst
* :pr:`5699`: BUG: Limit lag length in adf
* :pr:`5700`: MAINT: Update import of URLError
* :pr:`5701`: MAINT: missing imports, typos, fixes several NameErrors
* :pr:`5702`: MAINT: clean up docstring'd-out failure in __main__ block
* :pr:`5703`: MAINT: confirm that docstring'd-out traceback no longer raises; remove
* :pr:`5704`: ENH: expose innovations computation method to API.
* :pr:`5705`: WIP: TST: Sort dicts in test_multi
* :pr:`5707`: ENH: KPSS - detailed error message when lags > nobs
* :pr:`5709`: TST: Fix bad bash
* :pr:`5710`: CLN: clean up over/under indentation in tsa.tests.results, E12 codes
* :pr:`5712`: CLN: fix invalid escapes in test_stattools introduced in #5707
* :pr:`5713`: CLN/EX: Troubleshoot broken example, clean up now-working scratch paper
* :pr:`5715`: CLN: ellipses-out invalid escapes traceback
* :pr:`5716`: MAINT: Fix incorrect specification of loglike arg
* :pr:`5717`: MAINT: fix non-working example ex_pandas
* :pr:`5720`: CLN: remove impossible commented-out imports, close several
* :pr:`5721`: CLN: strict linting for dimred, processreg, and their tests.
* :pr:`5723`: Spelling fix in ValueError message
* :pr:`5724`: MAINT: close assorted small issues
* :pr:`5726`: DOC: Remove redundant attributes in GLM
* :pr:`5728`: CLN: remove and lint for unused imports
* :pr:`5729`: MAINT: use dummy_sparse func within method, see GH#5687
* :pr:`5730`: CLN: strict linting for discrete.tests.results
* :pr:`5732`: CLN: strict linting for genmod/tests/results
* :pr:`5734`: CLN: codes with only a few violations apiece
* :pr:`5736`: CLN: strict linting for regression/tests/results
* :pr:`5737`: CLN: strict linting for tsa.filters
* :pr:`5738`: CLN: strict linting for stats/tests/results/
* :pr:`5740`: CLN: strict linting for tsa.tests.results
* :pr:`5742`: CLN: strict linting for remaining results directories
* :pr:`5743`: CLN: strict linting for results files in sandbox/regression/tests/
* :pr:`5744`: CLN: Fix/lint for dangerous redefinitions and comparisons
* :pr:`5746`: MAINT: fix missing or redundant imports
* :pr:`5748`: CLN: clean up adfvalues, avoid using `eval`
* :pr:`5750`: CLN: E131 hanging indentation alignment
* :pr:`5758`: CLN: lint for ambiguous variable names
* :pr:`5760`: TST: test for intentionally emitted warnings, avoid some unintentional ones
* :pr:`5762`: BUG: rename wts to weights issue #4725
* :pr:`5765`: BUG/TST: Fix+test pieces of code that would raise NameError
* :pr:`5770`: DEPR: deprecate StataReader, StataWriter, genfromdta
* :pr:`5771`: ENH: improve missing data handling for GEE
* :pr:`5774`: PERF: use np.sum(...) instead of sum(...)
* :pr:`5778`: CLN: strict linting for test_varmax
* :pr:`5780`: TST: Protext against SSLError
* :pr:`5781`: CLN: Replace #5779
* :pr:`5783`: BUG: Ensure coint_johansen runs with 0 lags
* :pr:`5789`: BUG: GEE fit_history
* :pr:`5791`: Holder bunch
* :pr:`5792`: MAINT: matplotlib normed -> density
* :pr:`5793`: MAINT: Adjust tolerance for random fail on OSX
* :pr:`5796`: CLN: test_data.py
* :pr:`5798`: BUG: ignore bugs instead of fixing them
* :pr:`5801`: CI: Consolidate coveragerc spec
* :pr:`5803`: ENH: QIF regression
* :pr:`5805`: REF/CLN: collect imports at top of file, de-duplicate imports
* :pr:`5815`: CLN: test_gee.py
* :pr:`5816`: CLN: genmod/families/
* :pr:`5818`: CLN: qif
* :pr:`5825`: MAINT: use correct key name to check cov params presence
* :pr:`5830`: DOC: Add docstring for base class
* :pr:`5831`: PERF: Import speed
* :pr:`5833`: BUG: ARIMA fit with trend and constant exog
* :pr:`5834`: DOC: Fix small errors in release notes
* :pr:`5839`: MAINT: RangeIndex._start deprecated in pandas 0.25
* :pr:`5836`: CLN: over-indentation E117
* :pr:`5837`: CLN: invalid escapes in linear_model
* :pr:`5843`: MAINT: Catch intentional warnings
* :pr:`5846`: DOC: Update maintainer
* :pr:`5847`: BUG: Allow NumPy ints #
* :pr:`5848`: BUG: Warn rather than print
* :pr:`5850`: MAINT: Improve error message
* :pr:`5851`: BUG: Refactor method used to name variables
* :pr:`5853`: BUG: Add check for xnames length
* :pr:`5854`: BUG: Fix MNLogit summary with float values
* :pr:`5857`: BUG: Allow categorical to accept pandas dtype
* :pr:`5858`: BUG: Fix default alignment for SimpleTable
* :pr:`5859`: DOC: fix incorrect ARResults.predict docstring, closes #4498
* :pr:`5860`: Cdown gofplot typerror
* :pr:`5863`: MAINT: Use pd.Categorical() instead of .astype('categorical')
* :pr:`5868`: BUG: State space univariate smoothing w/ time-varying transition matrix: wrong transition matrix used
* :pr:`5869`: DOC: Improve ExponentialSmoothing docstring
* :pr:`5875`: DOC: Improve bug report template
* :pr:`5876`: BUG: Ensure keywords exist in partial reg plot
* :pr:`5879`: DOC: Update version dropdown javascript
