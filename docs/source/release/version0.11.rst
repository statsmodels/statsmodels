:orphan:

==============
Release 0.11.0
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
**Issues Closed**: 320
**Pull Requests Merged**: 260


The Highlights
==============


What's new - an overview
========================

The following lists the main new features of statsmodels 0.10. In addition,
release 0.10 includes bug fixes, refactorings and improvements in many areas.

Submodules
----------

``base``
~~~~~~~~



``discrete``
~~~~~~~~~~~~



``genmod``
~~~~~~~~~~


``graphics``
~~~~~~~~~~~~



``imputation``
~~~~~~~~~~~~~~



``iolib``
~~~~~~~~~



``multivariate``
~~~~~~~~~~~~~~~~



``nonparametric``
~~~~~~~~~~~~~~~~~



``regression``
~~~~~~~~~~~~~~



``sandbox``
~~~~~~~~~~~



``stats``
~~~~~~~~~



``tools``
~~~~~~~~~



``tsa``
~~~~~~~


New AR model
""""""""""""

- Model class: `sm.tsa.AutoReg`
- Estimates parameters using conditional MLE (OLS)
- Adds the ability to specify exogenous variables, include time trends,
  and add seasonal dummies.
- The function `sm.tsa.ar_model.ar_select_order` performs lag length selection
  for AutoReg models.

New ARIMA model
"""""""""""""""

- Model class: `sm.tsa.arima.ARIMA`
- Incorporates a variety of SARIMA estimators
    - MLE via state space methods (SARIMA models)
    - MLE via innovations algorithm (SARIMA models)
    - Hannan-Rissanen (ARIMA models)
    - Burg's method (AR models)
    - Innovations algorithm (MA models)
    - Yule-Walker (AR models)
- Handles exogenous regressors via GLS or by MLE with state space methods.
- Is part of the class of state space models and so inherits some additional
  functionality.

More robust regime switching models
"""""""""""""""""""""""""""""""""""

- Implementation of the Hamilton filter and Kim smoother in log space avoids
  underflow errors.

``tsa.statespace``
~~~~~~~~~~~~~~~~~~

Linear exponential smoothing models
"""""""""""""""""""""""""""""""""""

- Model class: `sm.tsa.statespace.ExponentialSmoothing`
- Alternative to `sm.tsa.ExponentialSmoothing`
- Only supports linear models
- Is part of the class of state space models and so inherits some additional
  functionality.

Methods to apply parameters fitted on one dataset to another dataset
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- Methods: `extend`, `append`, and `apply`, for state space results classes
- These methods allow applying fitted parameters from a training dataset to a
  test dataset in various ways
- Useful for conveniently performing cross-validation exercises

Method to hold some parameters fixed at known values
""""""""""""""""""""""""""""""""""""""""""""""""""""

- Methods: `fix_params` and `fit_constrained`, for state space model classes
- These methods allow setting some parameters to known values and then
  estimating the remaining parameters

Option for low memory operations
""""""""""""""""""""""""""""""""

- Argument: `low_memory=True`, for `fit`, `filter`, and `smooth`
- Only a subset of results are available when using this option, but it does
  allow for prediction, diagnostics, and forecasting
- Useful to speed up cross-validation exercises

Improved access to state estimates
""""""""""""""""""""""""""""""""""

- Attribute: `states`, for state space results classes
- Wraps estimated states (predicted, filtered, smoothed) as Pandas objects with
  the appropriate index.
- Adds human-readable names for states, where possible.

Improved simulation and impulse responses for time-varying models
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- Argument: `anchor' allows specifying the period after which to begin the
  simulation.
- Example: to simulate data following the sample period, use `anchor='end'`

``maintenance``
~~~~~~~~~~~~~~~




bug-wrong
---------

A new issue label `type-bug-wrong` indicates bugs that cause that incorrect
numbers are returned without warnings.
(Regular bugs are mostly usability bugs or bugs that raise an exception for
unsupported use cases.)
`see tagged issues <https://github.com/statsmodels/statsmodels/issues?q=is%3Aissue+label%3Atype-bug-wrong+is%3Aclosed+milestone%3A0.11/>`_


Major Bugs Fixed
================

See github issues for a list of bug fixes included in this release

- `Closed bugs <https://github.com/statsmodels/statsmodels/pulls?utf8=%E2%9C%93&q=is%3Apr+is%3Amerged+milestone%3A0.11+label%3Atype-bug/>`_
- `Closed bugs (wrong result) <https://github.com/statsmodels/statsmodels/pulls?q=is%3Apr+is%3Amerged+milestone%3A0.11+label%3Atype-bug-wrong/>`_


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

Atticus Yang
Austin Adams
Brock Mendel
Chad Fulton
Christian Clauss
Emil Mirzayev
Graham Inggs
Guglielmo Saggiorato
Hassan Kibirige
Ian Preston
Jefferson Tweed
Josef Perktold
Keller Scholl
Kerby Shedden
Kevin Sheppard
Lucas Roberts
Mandy Gu
Omer Ozen
Padarn Wilson
Peter Quackenbush
Qingqing Mao
Rebecca N. Palmer
Samesh Lakhotia
Sandu Ursu
Tim Staley
Yasine Gangat
comatrion
luxiform
partev
vegcev
郭飞


These lists of names are automatically generated based on git log, and may not
be complete.

Merged Pull Requests
--------------------

The following Pull Requests were merged since the last release:

* :pr:`4421`: implement cached_value, cached_data proof of concept
* :pr:`5235`: STY: use Appender pattern for docstrings
* :pr:`5283`: ENH: Use cython fused types to simplify statespace code
* :pr:`5610`: BUG: Lilliefors min nobs not set
* :pr:`5692`: MAINT: remove sandbox.formula, supplanted by patsy
* :pr:`5735`: ENH: Allow fixing parameters in state space models
* :pr:`5757`: MAINT: Remove docstring'd-out traceback for code that no longer raises
* :pr:`5768`: WIP/TST: enable/mark mangled/commented-out tests
* :pr:`5784`: MAINT: implement parts of #5220, deprecate ancient aliases
* :pr:`5797`: TST: test for anova_nistcertified
* :pr:`5799`: TST: Catch warnings produced during tests
* :pr:`5814`: CLN: parts of iolib
* :pr:`5819`: CLN: robust
* :pr:`5821`: CLN: test_constrained
* :pr:`5826`: ENH/REF: Markov switching in log space: Hamilton filter / Kim smoother
* :pr:`5827`: ENH: Add new version of ARIMA-type estimators (AR, ARIMA, SARIMAX)
* :pr:`5832`: DOC: remove orphaned docs files
* :pr:`5841`: MAINT: remove no-longer-needed HC_se lookups
* :pr:`5842`: CLN: E701 multiple statements on one line (colon)
* :pr:`5852`: ENH: Dimension reduction for covariance matrices
* :pr:`5856`: MAINT: remove ex_pairwise file dominated by try_tukey_hsd
* :pr:`5892`: BUG: fix pandas compat
* :pr:`5893`: BUG: exponential smoothing - damped trend gives incorrect param, predictions
* :pr:`5895`: DOC: improvements to BayesMixedGLM docs, argument checking
* :pr:`5897`: MAINT: Use pytest.raises to check error message
* :pr:`5898`: ENH: use class to define MixedLM variance components structure
* :pr:`5903`: BUG: Fix kwargs update bug in linear model fit_regularized
* :pr:`5910`: MAINT: Bump dependencies
* :pr:`5915`: ENH: state space: add methods to apply fitted parameters to new observations or new dataset
* :pr:`5917`: BUG: TVTP for Markov regression
* :pr:`5921`: BUG: Ensure exponential smoothers has continuous double data
* :pr:`5922`: MAINT: Fix pandas imports
* :pr:`5926`: Add STL decomposition for time series
* :pr:`5927`: MAINT: Remove Python 2.7 from Appveyory
* :pr:`5928`: ENH: Add array_like function to simplify input checking
* :pr:`5929`: DOC: array-like -> array_like
* :pr:`5930`: BUG: Limit lags in KPSS
* :pr:`5931`: MAINT: Relax tol on test that randomly fails
* :pr:`5933`: MAINT: Fix test that fails with positive probability
* :pr:`5935`: CLN: port parts of #5220
* :pr:`5937`: DOC: Change some more links to https
* :pr:`5938`: MAINT: Remove Python 2.7 from travis
* :pr:`5939`: DOC: Fix self-contradictory minimum dependency versions
* :pr:`5940`: MAINT: Fix linting failures
* :pr:`5946`: DOC: Fix formula for log-like in WLS
* :pr:`5947`: PERF: Cythonize innovations algo and filter
* :pr:`5948`: ENH: Normalize eigenvectors from coint_johansen
* :pr:`5949`: DOC: Fix typo
* :pr:`5950`: MAINT: Drop redendant travis configs
* :pr:`5951`: BUG: Fix mosaic plot with missing category
* :pr:`5952`: ENH: Improve RESET test stability
* :pr:`5953`: ENH: Add type checkers/converts for int, float and bool
* :pr:`5954`: MAINT: Mark MPL test as MPL
* :pr:`5956`: BUG: Fix mutidimensional model cov_params when using pandas
* :pr:`5957`: DOC: Clarify xname length and purpose
* :pr:`5958`: MAINT: Deprecate periodogram
* :pr:`5960`: MAINT: Ensure seaborn is available for docbuild
* :pr:`5962`: CLN: cython cleanups
* :pr:`5963`: ENH: Functional SIR
* :pr:`5964`: ENH: Add start_params to RLM
* :pr:`5965`: MAINT: Remove PY3
* :pr:`5966`: ENH: Add JohansenResults class
* :pr:`5967`: BUG/ENH: Improve RLM in the case of perfect fit
* :pr:`5968`: Rlucas7 mad fun empty
* :pr:`5969`: MAINT: Remove future and Python 2.7
* :pr:`5971`: BUG: Fix a future issue in ExpSmooth
* :pr:`5972`: MAINT: Remove string_types in favor of str
* :pr:`5976`: MAINT: Restore ResettableCache
* :pr:`5977`: MAINT: Cleanup legacy imports
* :pr:`5982`: CLN: follow-up to #5956
* :pr:`5983`: BUG: Fix return for RegressionResults
* :pr:`5984`: MAINT: Clarify breusch_pagan is for scalars
* :pr:`5986`: DOC: Add parameters for CountModel predict
* :pr:`5987`: MAINT: add W605 to lint codes
* :pr:`5988`: CLN: follow-up to #5928
* :pr:`5990`: MAINT/DOC: Add spell checking
* :pr:`5991`: MAINT: Remove comment no longer relevant
* :pr:`5992`: DOC: Fix many spelling errors
* :pr:`5994`: DOC: Small fixups after teh spell check
* :pr:`5995`: ENH: Add R-squared and Adj. R_squared to summary_col
* :pr:`5996`: BUG: Limit lags in KPSS
* :pr:`5997`: ENH/BUG: Add check to AR instance to prevent bugs
* :pr:`5998`: Replace alpha=0.05 with alpha=alpha
* :pr:`5999`: ENH: Add summary to AR
* :pr:`6000`: DOC: Clarify that GARCH models are deprecated
* :pr:`6001`: MAINT: Refactor X13 testing
* :pr:`6002`: MAINT: Standardized on nlags for acf/pacf
* :pr:`6003`: BUG: Do not fit when fit=False
* :pr:`6004`: ENH/BUG: Allow ARMA predict to swallow typ
* :pr:`6005`: Ignore warns on 32 bit linux
* :pr:`6006`: BUG/ENH: Check exog in ARMA and ARIMA predict
* :pr:`6007`: MAINT: Rename forecast years to forecast periods
* :pr:`6008`: ENH: Allow GC testing for specific lags
* :pr:`6009`: TST: Verify categorical is supported for MNLogit
* :pr:`6010`: TST: Imrove test that is failing due to precision issues
* :pr:`6011`: MAINT/BUG/TST: Improve testing of seasonal decompose
* :pr:`6012`: BUG: Fix t-test and f-test for multidim params
* :pr:`6014`: ENH: Zivot Andrews test
* :pr:`6015`: CLN: Remove notes about incorrect test
* :pr:`6016`: TST: Add check for dtyppes in Binomial
* :pr:`6017`: ENH: Set limit for number of endog variables when using formulas
* :pr:`6018`: Josef pkt arma startparams2
* :pr:`6019`: BUG: Fix ARMA cov_params
* :pr:`6020`: TST: Correct test to use trend not level
* :pr:`6022`: DOC: added content for two headings in VAR docs
* :pr:`6023`: TST: Verify missing exog raises in ARIMA
* :pr:`6026`: WIP: Added Oaxaca-Blinder Decomposition
* :pr:`6028`: ENH: Add rolling WLS and OLS
* :pr:`6030`: Turn relative import into an absolute import
* :pr:`6031`: DOC: Fix regression doc strings
* :pr:`6036`: BLD/DOC: Add doc string check to doc build
* :pr:`6038`: DOC: Apply documentation standardizations
* :pr:`6039`: MAINT: Change types for future changes in NumPy
* :pr:`6041`: DOC: Fix spelling
* :pr:`6042`: Merge pull request #6041 from bashtage/doc-fixes
* :pr:`6044`: DOC: Fix notebook due to pandas index change
* :pr:`6045`: DOC/MAINT: Remove warning due to deprecated features
* :pr:`6046`: DOC: Remove DynamicVAR
* :pr:`6048`: DOC: Small doc site improvements
* :pr:`6050`: BUG: MLEModel now passes nobs to Representation
* :pr:`6052`: DOC: Small fix ups for modernized size
* :pr:`6053`: DOC: More small doc fixes
* :pr:`6054`: DOC: Small changes to doc building
* :pr:`6055`: DOC: Use the working branch of numpy doc
* :pr:`6056`: Rolling ls prep
* :pr:`6057`: DOC: Fix spelling in notebooks
* :pr:`6058`: DOC: Fix missing spaces around colon
* :pr:`6059`: REF: move garch to archive/
* :pr:`6060`: DOC: Continue fixing docstring formatting
* :pr:`6062`: DOC: Fix web font size
* :pr:`6063`: DOC: Fix web font size
* :pr:`6064`: ENH/PERF: Only perform required predict iterations in state space models
* :pr:`6066`: MAINT: Fix small lint issue
* :pr:`6067`: DOC: Fix doc errrors affecting build
* :pr:`6069`: MAINT: Stop testing on old, bugy SciPy
* :pr:`6070`: BUG: Fix ARMA so that it works with exog when trend=nc
* :pr:`6071`: ENH: state space: Improve low memory usability; allow in fit, loglike
* :pr:`6072`: BUG: state space: cov_params computation in fix_params context
* :pr:`6073`: TST: Add conserve memory tests.
* :pr:`6074`: ENH: Improve cov_params in append, extend, apply
* :pr:`6075`: DOC: Clean tsatools docs
* :pr:`6077`: DOC: Improve regression doc strings
* :pr:`6079`: ENH/DOC: Improve Ljung-Box
* :pr:`6080`: DOC: Improve docs in tools and ar_model
* :pr:`6081`: BUG: Fix error introduced in isestimable
* :pr:`6082`: DOC: Improve filter docstrings
* :pr:`6085`: DOC: Spelling and motebook link
* :pr:`6087`: ENH: Replacement for AR
* :pr:`6088`: MAINT: Small fixes in preperation for larger changes
* :pr:`6089`: DOC: Website fix
* :pr:`6090`: ENH/DOC: Add tools for programatically manipulating docstrings
* :pr:`6091`: MAINT/SEC: Remove unnecessary pickle use
* :pr:`6092`: MAINT: Ensure r download cache works
* :pr:`6093`: MAINT: Fix new cache name
* :pr:`6094`: TST: Fix wrong test
* :pr:`6096`: DOC: Seasonality in SARIMAX Notebook
* :pr:`6102`: ENH: Improve SARIMAX start_params if too few nobs
* :pr:`6104`: BUG: Fix score computation with fixed params
* :pr:`6105`: Update correlation_tools.py
* :pr:`6106`: DOC: Changes summary_col's docstring to match variables
* :pr:`6107`: DOC: Update CONTRIBUTING.rst "relase"-> "release"
* :pr:`6108`: DOC: Update link in CONTRIBUTING.rst
* :pr:`6110`: DOC: Update PR template Numpy guide link
* :pr:`6111`: ENH: Add exact diffuse initialization as an option for SARIMAX, UnobservedComponents
* :pr:`6113`: added interpretations to LogitResults.get_margeff
* :pr:`6116`: DOC: Improve docstrings
* :pr:`6117`: BUG: Remove extra LICENSE.txt and setup.cfg
* :pr:`6118`: Update summary2.py
* :pr:`6119`: DOC: Switch doc theme
* :pr:`6120`: DOC: Add initial API doc
* :pr:`6122`: DOC: Small improvements to docs
* :pr:`6123`: DOC: Switch doc icon
* :pr:`6124`: Plot only unique censored points
* :pr:`6125`: DOC: Fix doc build failure
* :pr:`6126`: DOC: Update templates and add missing API functions
* :pr:`6130`:  BUG: Incorrect TSA index if loc resolves to slice
* :pr:`6131`: ENH: Compute standardized forecast error in diffuse period if possible
* :pr:`6133`: BUG: start_params for VMA model with exog.
* :pr:`6134`: DOC: Add missing functions from the API
* :pr:`6136`: DOC: Restructure the documentation
* :pr:`6142`: DOC: Add a new logo
* :pr:`6143`: DOC: Fix validator so that it works
* :pr:`6144`: BUG: use self.data consistently
* :pr:`6145`: DOC: Add formula API
* :pr:`6152`: BUG: Fix accepting of eval environment for formula
* :pr:`6160`: DOC: fix sidebartoc
* :pr:`6161`: Travis CI: The sudo: tag is deprecated in Travis
* :pr:`6162`: DOC/SEC: Warn that only trusted files should be unpickled
* :pr:`6163`: Improve the cvxopt not found error
* :pr:`6164`: Be compatible with scipy 1.3
* :pr:`6165`: Don't assume that 'python' is Python 3
* :pr:`6166`: DOC: Update pickle warning
* :pr:`6167`: DOC: Fix warning format
* :pr:`6179`: ENH: Adds state space version of linear exponential smoothing models
* :pr:`6181`: ENH: state space: add wrapped states and, where possible, named states
* :pr:`6198`: DOC: Clarify req for cvxopt
* :pr:`6204`: DOC: Spelling and Doc String Fixes
* :pr:`6205`: MAINT: Exclude pytest-xdist 1.30
* :pr:`6208`: ENH: Scale parameter handling in GEE
* :pr:`6214`: fix a typo
* :pr:`6215`: fix typos in install.rst
* :pr:`6216`: fix a typo
* :pr:`6217`: BUG: Fix summary table header for mixedlm
* :pr:`6222`: WIP: Relax precision for ppc64el
* :pr:`6227`: ENH: Add missing keyword argument to plot_acf
* :pr:`6231`: BUG: allow dynamic factor starting parameters computation with NaNs values
* :pr:`6232`: BUG: division by zero in exponential smoothing if damping_slope=0
* :pr:`6233`: BUG: dynamic factor model use AR model for error start params if error_var=False
* :pr:`6235`: DOC: docstring fixes
* :pr:`6239`: BUG: SARIMAX index behavior with simple_differencing=True
* :pr:`6240`: BUG: parameter names in DynamicFactor for unstructured error covariance matrix
* :pr:`6241`: BUG: SARIMAX: basic validation for order, seasonal_order
* :pr:`6242`: BUG: Forecasts now ignore non-monotonic period index
* :pr:`6246`: TST: Add Python 3.8 environment
* :pr:`6250`: ENH: Update SARIMAX to use SARIMAXSpecification for more consistent input handling
* :pr:`6254`: ENH: State space: Add finer-grained memory conserve settings
* :pr:`6255`: ignore vscode
* :pr:`6257`: DOC: Fix spelling in notebooks
* :pr:`6258`: BUG: Hannan-Rissanen third stage is invalid if non-stationary/invertible
* :pr:`6260`: BUG: cloning of arima.ARIMA models.
* :pr:`6261`: BUG: state space: saving fixed params w/ extend, apply, append
* :pr:`6266`: ENH: and vlines option to plot_fit
* :pr:`6275`: MAINT/DOC: Clarify patsy 0.5.1 is required
* :pr:`6279`: DOC: Fix notebook
* :pr:`6280`: ENH: State space: Improve simulate, IRF, prediction
* :pr:`6281`: BUG: Pass arguments through in plot_leverage_resid2
* :pr:`6283`: Close issues
* :pr:`6285`: BUG: Raise in GC test for VAR(0)
* :pr:`6286`: BUG: Correct VAR sumary when model contains exog variables
* :pr:`6288`: MAINT: Update test tolerance
* :pr:`6289`: DOC: doc string changes
* :pr:`6290`: MAINT: Remove open_help method
* :pr:`6291`: MAINT: Remove deprecated code in preperation for release
* :pr:`6292`: Padarn singlepointkde patch
* :pr:`6294`: ENH: better argument checking for StratifiedTable
* :pr:`6297`: BUG: Fix conf interval with MI
* :pr:`6298`: Correct spells
* :pr:`6299`: DOC: Add example notebook for GEE score tests
* :pr:`6303`: DOC/MAINT: Add simple, documented script to get github info
* :pr:`6310`: MAINT: Deprecate recarray support
* :pr:`6311`: TST: Reduce test size to prevent 32-bit crash
* :pr:`6312`: MAINT: Remove chain dot
* :pr:`6313`: MAINT: Catch and fix warnings
* :pr:`6314`: BUG: Check dtype in KDEUnivariate
* :pr:`6315`: MAINT: Use NumPy's linalg when available
* :pr:`6316`: MAINT: Workaround NumPy ptp issue
* :pr:`6317`: DOC: Update test running instructions
* :pr:`6318`: BUG: Ensure inputs are finite in granger causalty test
* :pr:`6319`: DOC: Restore test() autosummary
* :pr:`6320`: BUG: Restore multicomp
* :pr:`6321`: BUG: Fix trend due to recent changes
* :pr:`6322`: DOC: fix alpha description for GLMGam
* :pr:`6324`: Improve ljung box
* :pr:`6327`: Move api docs
* :pr:`6332`: DEPR: state space: deprecate out-of-sample w/ unsupported index
* :pr:`6333`: BUG: state space: integer params can cause imaginary output
* :pr:`6334`: ENH: append, extend check that index matches model
* :pr:`6337`: BUG: fix k_exog, k_trend in arima.ARIMA; raise error when cloning a model with exog if no new exog given
* :pr:`6338`: DOC: Documentation for release v0.11
* :pr:`6340`: BUG: fix _get_index_loc with date strings
