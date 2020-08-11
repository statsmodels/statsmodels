:orphan:

==============
Release 0.12.0
==============

Release summary
===============

statsmodels is using github to store the updated documentation. Two version are available:

- `Stable <https://www.statsmodels.org/>`_, the latest release
- `Development <https://www.statsmodels.org/devel/>`_, the latest build of the master branch

**Warning**

API stability is not guaranteed for new features, although even in
this case changes will be made in a backwards compatible way if
possible. The stability of a new feature depends on how much time it
was already in statsmodels master and how much usage it has already
seen.  If there are specific known problems or limitations, then they
are mentioned in the docstrings.

Stats
-----
**Issues Closed**: 223

**Pull Requests Merged**: 208


The Highlights
==============

Time-Series Analysis
--------------------

New exponential smoothing model: ETS (Error, Trend, Seasonal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Class implementing ETS models :class:`~statsmodels.tsa.exponential_smoothing.ets.ETSModel`.
- Includes linear and non-linear exponential smoothing models
- Supports parameter fitting, in-sample prediction and out-of-sample
  forecasting, prediction intervals, simulation, and more.
- Based on the innovations state space approach.

Statespace Models
-----------------

New dynamic factor model for large datasets and monthly / quarterly mixed frequency models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- New dynamic factor model :class:`~statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ`.
- Allows for hundreds of observed variables, by fitting with the EM algorithm
- Allows specifying factors that load only on a specific group of variables
- Allows for monthly / quarterly mixed frequency models. For example, this
  supports one popular approach to "Nowcasting" GDP

Decomposition of forecast updates based on the "news"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- New :meth:`~statsmodels.tsa.statespace.mlemodel.MLEResults.news` method for state space model results objects
- Links updated data to changes in forecasts
- Supports "nowcasting" exercises that progressively incorporate more and more
  information as time goes on

Sparse Cholesky Simulation Smoother
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- New option for simulation smoothing in state space models via the
  "Cholesky factor algorithm" (CFA) approach in
  :class:`~statsmodels.tsa.statespace.cfa_simulation_smoother.CFASimulationSmoother`
- Takes advantage of algorithms for sparse Cholesky factorization, rather than
  using the typical simulation smoother based on Kalman filtering and smoothing

Option to use Chadrasekhar recursions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- New option for state space models to use Chandrasekhar recursions rather than
  than the typical Kalman filtering recursions by setting ``filter_chandrasekhar=True``.
- Improved performance for some models with large state vectors

Forecasting Methods
~~~~~~~~~~~~~~~~~~~
Two popular method for forecasting time series, forecasting after STL decomposition
(:class:`~statsmodels.tsa.forecasting.stl.STLForecast`)
and the Theta model (:class:`~statsmodels.tsa.forecasting.theta.ThetaModel`) have
been added.

Complex Deterministic Terms
~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`~statsmodels.tsa.deterministic.DeterministicProcess` can be used to generate
deterministic processes containing time trends, seasonal dummies and Fourier components.
A :class:`~statsmodels.tsa.deterministic.DeterministicProcess` can be used to produce
in-sample regressors or out-of-sample values suitable for forecasting.


What's new - an overview
========================

The following lists the main new features of statsmodels 0.12.0. In addition,
release 0.12.0 includes bug fixes, refactorings and improvements in many areas.

Submodules
----------


``Documentation``
~~~~~~~~~~~~~~~~~
- Fix the version that appears in the documentation  (:pr:`6452`)
- Send log to dev/null/  (:pr:`6456`)
- Correct spelling of various  (:pr:`6518`)
- Fix typos  (:pr:`6531`)
- Update interactions_anova.ipynb  (:pr:`6601`)
- Fix `true` type on statespace docs page  (:pr:`6616`)
- Minor fixes for holtwinters simulate  (:pr:`6631`)
- Change OLS example to use datasets  (:pr:`6656`)
- Fix AutoReg docstring  (:pr:`6662`)
- Fix `fdrcorrection` docstring missing `is_sorted` parameter  (:pr:`6680`)
- Add new badges  (:pr:`6704`)
- Fix number if notebook text  (:pr:`6709`)
- Improve Factor and related docstrings  (:pr:`6719`)
- Improve explantion of missing values in ACF and related  (:pr:`6726`)
- Notebook for quasibinomial regression  (:pr:`6732`)
- Improve "conservative" doc  (:pr:`6738`)
- Update broken link  (:pr:`6742`)
- Fix broken links with 404 error  (:pr:`6746`)
- Demonstrate variance components analysis  (:pr:`6758`)
- Make deprecations more visible  (:pr:`6775`)
- Numpydoc signatures  (:pr:`6825`)
- Correct reference in docs  (:pr:`6837`)
- Include dot_plot  (:pr:`6841`)
- Updated durbin_watson Docstring and Tests  (:pr:`6848`)
- Explain low df in cluster  (:pr:`6853`)
- Fix common doc errors  (:pr:`6862`)
- Small doc fixes  (:pr:`6874`)
- Fix issues in docs related to exponential smoothing  (:pr:`6879`)
- Spelling and other doc fixes  (:pr:`6902`)
- Correct spacing around colon in docstrings  (:pr:`6903`)
- Initial 0.12 Release Note  (:pr:`6923`)
- Fix doc errors and silence warning  (:pr:`6931`)
- Clarify deprecations  (:pr:`6932`)
- Document exceptions and warnings  (:pr:`6943`)
- Update pandas function in hp_filter example  (:pr:`6946`)
- Prepare docs  (:pr:`6948`)
- Fix final issues in release note  (:pr:`6951`)

``Performance``
~~~~~~~~~~~~~~~
- State space: add Chandrasekhar recursions  (:pr:`6411`)
- Speed up HC2/HC3 standard error calculation, using less memory  (:pr:`6664`)
- Sparse matrices in MixedLM  (:pr:`6766`)

``backport``
~~~~~~~~~~~~
- `MLEResults.states.predicted` has wrong index  (:pr:`6580`)
- State space: simulate with time-varying covariance matrices.  (:pr:`6607`)
- State space: error with collapsed observations when missing  (:pr:`6613`)
- Dataframe/series concatenation in statespace results append  (:pr:`6768`)
- Pass cov_type, cov_kwargs through ARIMA.fit  (:pr:`6770`)

``base``
~~~~~~~~
- Don't attach patsy constraint instance   (:pr:`6521`)
- Fix constraints and bunds when use scipy.optimize.minimize  (:pr:`6657`)
- Correct shape of fvalue and f_pvalue  (:pr:`6831`)
- Correct dimension when data removed  (:pr:`6888`)

``build``
~~~~~~~~~
- Use pip on Azure  (:pr:`6474`)
- Attempt to cache key docbuild files  (:pr:`6490`)
- Improve doc caching  (:pr:`6491`)
- Azure: Mac OSX 10.13 -> 10.14  (:pr:`6587`)

``discrete``
~~~~~~~~~~~~
- Don't attach patsy constraint instance   (:pr:`6521`)
- Sparse matrices in MixedLM  (:pr:`6766`)
- Catch warnings in discrete  (:pr:`6836`)
- Add improved .cdf() and .ppf() to discrete distributions  (:pr:`6938`)
- Remove k_extra from effects_idx  (:pr:`6939`)
- Improve count model tests  (:pr:`6940`)

``docs``
~~~~~~~~
- Fix doc errors and silence warning  (:pr:`6931`)
- Prepare docs  (:pr:`6948`)

``duration``
~~~~~~~~~~~~
- Allow more than 2 groups for survdiff in statmodels.duration  (:pr:`6626`)

``gam``
~~~~~~~
- Fix GAM for 1-dim exog_linear   (:pr:`6520`)
- Fixed BSplines to match existing docs  (:pr:`6915`)

``genmod``
~~~~~~~~~~
- Change default optimizer for glm/ridge and make it user-settable  (:pr:`6438`)
- Fix exposure/offset handling in GEEResults  (:pr:`6475`)
- Use GLM starting values for QIF  (:pr:`6514`)
- Don't attach patsy constraint instance   (:pr:`6521`)
- Allow GEE weights to vary within clusters  (:pr:`6582`)
- Calculate AR covariance parameters for gridded data  (:pr:`6621`)
- Warn for non-convergence in elastic net  (:pr:`6697`)
- Gh 6627  (:pr:`6852`)
- Change of BIC formula in GLM  (:pr:`6941`)
- Make glm's predict function return numpy array even if exposure is a pandas series  (:pr:`6942`)
- Fix check for offset_exposure in null  (:pr:`6957`)
- Add test for offset exposure null  (:pr:`6959`)

``graphics``
~~~~~~~~~~~~
- Include figsize as parameter for IRF plot  (:pr:`6590`)
- Speed up banddepth calculations  (:pr:`6744`)
- Fix logic in labeling corr plot  (:pr:`6818`)
- Enable qqplot_2sample to handle uneven samples  (:pr:`6906`)
- Support frozen dist in ProbPlots  (:pr:`6910`)

``io``
~~~~~~
- Handle pathlib.Path objects  (:pr:`6654`)
- Added label option to summary.to_latex()  (:pr:`6895`)
- Fixed the shifted column names in summary.to_latex()  (:pr:`6900`)
- Removed additional hline between tabulars  (:pr:`6905`)

``maintenance``
~~~~~~~~~~~~~~~
- Special docbuild  (:pr:`6457`)
- Special docbuild"  (:pr:`6460`)
- Correcting typo  (:pr:`6461`)
- Avoid noise in f-pvalue  (:pr:`6465`)
- Replace Python 3.5 with 3.8 on Azure  (:pr:`6466`)
- Update supported versions  (:pr:`6467`)
- Fix future warnings  (:pr:`6469`)
- Fix issue with ragged array  (:pr:`6471`)
- Avoid future error  (:pr:`6473`)
- Silence expected visible deprecation warning  (:pr:`6477`)
- Remove Python 3.5 references  (:pr:`6492`)
- Avoid calling depr code  (:pr:`6493`)
- Use travis cache and optimize build times  (:pr:`6495`)
- Relax tolerance on test that occasionally fails  (:pr:`6534`)
- Relax tolerance on test that randomly fails  (:pr:`6588`)
- Fix appveyor/conda  (:pr:`6653`)
- Delete empty directory  (:pr:`6671`)
- Flake8 fixes  (:pr:`6710`)
- Remove deprecated keyword  (:pr:`6712`)
- Remove OrderedDict  (:pr:`6715`)
- Remove dtype np.integer for avoid Dep Warning  (:pr:`6728`)
- Update pip-pre links  (:pr:`6733`)
- Spelling and small fixes  (:pr:`6752`)
- Remove error on FutureWarning  (:pr:`6811`)
- Fix failing tests  (:pr:`6817`)
- Replace Warnings with Notes in regression summary  (:pr:`6828`)
- Numpydoc should work now  (:pr:`6842`)
- Deprecate categorical  (:pr:`6843`)
- Remove redundant definition  (:pr:`6845`)
- Relax tolerance on test that fails Win32  (:pr:`6849`)
- Fix error on nightly build  (:pr:`6850`)
- Correct debugging info  (:pr:`6855`)
- Mark VAR from_formula as NotImplemented  (:pr:`6865`)
- Allow skip if rdataset fails  (:pr:`6871`)
- Improve lint  (:pr:`6885`)
- Change default lag in serial correlation tests  (:pr:`6893`)
- Ensure setuptools is imported first  (:pr:`6894`)
- Remove FutureWarnings  (:pr:`6920`)
- Add tool to simplify documenting API in release notes  (:pr:`6922`)
- Relax test tolerance for future compat  (:pr:`6945`)
- Fixes for failures in wheel building  (:pr:`6952`)
- Fixes for wheel building  (:pr:`6954`)

``multivariate``
~~~~~~~~~~~~~~~~
- Multivariate mean tests and confint  (:pr:`4107`)
- Improve missing value handling in PCA  (:pr:`6705`)

``nonparametric``
~~~~~~~~~~~~~~~~~
- Fix #6511  (:pr:`6515`)
- Fix domain check  (:pr:`6547`)
- Ensure sigma estimate is positive in KDE  (:pr:`6713`)
- Fix access to normal_reference_constant  (:pr:`6806`)
- Add xvals param to lowess smoother  (:pr:`6908`)

``regression``
~~~~~~~~~~~~~~
- Statsmodels.regression.linear_model.OLS.fit_regularized fails to generate correct answer (#6604)  (:pr:`6608`)
- Change OLS example to use datasets  (:pr:`6656`)
- Speed up HC2/HC3 standard error calculation, using less memory  (:pr:`6664`)
- Fix summary col R2 ordering  (:pr:`6714`)
- Insufficient input checks in QuantReg  (:pr:`6747`)
- Add expanding initialization to RollingOLS/WLS  (:pr:`6838`)
- Add  a note when R2 is uncentered  (:pr:`6844`)

``stats``
~~~~~~~~~
- Multivariate mean tests and confint  (:pr:`4107`)
- Fix tukey-hsd for 1 pvalue   (:pr:`6470`)
- Add option for original Breusch-Pagan heteroscedasticity test  (:pr:`6508`)
- ENH Allow optional regularization in local fdr  (:pr:`6622`)
- Add meta-analysis (basic methods)  (:pr:`6632`)
- Add two independent proportion inference rebased  (:pr:`6675`)
- Rates, poisson means two-sample comparison  rebased  (:pr:`6677`)
- Stats.base, add HolderTuple, Holder class with indexing  (:pr:`6678`)
- Add covariance structure hypothesis tests  (:pr:`6693`)
- Raise exception when recursive residual is not well defined  (:pr:`6727`)
- Mediation support for PH regression  (:pr:`6782`)
- Stats robust rebased2  (:pr:`6789`)
- Hotelling's Two Sample Mean Test  (:pr:`6810`)
- Stats moment_helpers use random state in unit test  (:pr:`6835`)
- Updated durbin_watson Docstring and Tests  (:pr:`6848`)
- Add recent stats addition to docs  (:pr:`6859`)
- REF/DOC docs and refactor of recent stats  (:pr:`6872`)
- Api cleanup and improve docstrings in stats, round 3  (:pr:`6897`)
- Improve descriptivestats  (:pr:`6944`)
- Catch warning  (:pr:`6964`)

``tools``
~~~~~~~~~
- Return column information in add_constant  (:pr:`6830`)
- Add QR-based matrix rank  (:pr:`6834`)
- Add Root Mean Square Percentage Error  (:pr:`6926`)

``tsa``
~~~~~~~
- Fixes #6553, sliced predicted values according to predicted index  (:pr:`6556`)
- Holt-Winters simulations  (:pr:`6560`)
- Example notebook (r): stationarity and detrending (ADF/KPSS)  (:pr:`6614`)
- Ensure text comparison is lower  (:pr:`6628`)
- Minor fixes for holtwinters simulate  (:pr:`6631`)
- New exponential smoothing implementation  (:pr:`6699`)
- Improve warning message in KPSS  (:pr:`6711`)
- Change trend initialization in STL  (:pr:`6722`)
- Add check in test_whiteness  (:pr:`6723`)
- Raise on incorrectly sized exog  (:pr:`6730`)
- Add deterministic processes  (:pr:`6751`)
- Add Theta forecasting method  (:pr:`6767`)
- Automatic lag selection for Box-Pierce, Ljung-Box #6645  (:pr:`6785`)
- Fix missing str  (:pr:`6827`)
- Add support for PeriodIndex to AutoReg  (:pr:`6829`)
- Error in append for ARIMA model with trend  (:pr:`6832`)
- Add QR-based matrix rank  (:pr:`6834`)
- Rename unbiased to adjusted  (:pr:`6839`)
- Ensure PACF lag length is sensible  (:pr:`6846`)
- Allow Series as exog in predict  (:pr:`6847`)
- Raise on nonstationary parameters when attempting to use GLS  (:pr:`6854`)
- Relax test tolerance  (:pr:`6856`)
- Limit maxlags in VAR  (:pr:`6867`)
- Fix indexing with HoltWinters's forecast  (:pr:`6869`)
- Refactor Holt-Winters  (:pr:`6870`)
- Fix raise exception on granger causality test  (:pr:`6877`)
- Get_prediction method for ETS  (:pr:`6882`)
- Ets: test for simple exponential smoothing convergence  (:pr:`6884`)
- Added diagnostics test to ETS model  (:pr:`6892`)
- Stop transforming ES components  (:pr:`6904`)
- Fix extend in VARMAX with trend  (:pr:`6909`)
- Add STL Forecasting method  (:pr:`6911`)
- Dynamic is incorrect when not an int in statespace get_prediction  (:pr:`6917`)
- Correct IRF nobs with exog  (:pr:`6925`)
- Add get_prediction to AutoReg  (:pr:`6927`)
- Standardize forecast API  (:pr:`6933`)
- Fix small issues post ETS get_prediction merge  (:pr:`6934`)
- Modify failing test on Windows  (:pr:`6949`)
- Improve ETS / statespace documentation and higlights for v0.12   (:pr:`6950`)
- Remove FutureWarnings  (:pr:`6958`)

``tsa.statespace``
~~~~~~~~~~~~~~~~~~
- State space: add Chandrasekhar recursions  (:pr:`6411`)
- Use reset_randomstate  (:pr:`6433`)
- State space: add "Cholesky factor algorithm" simulation smoothing  (:pr:`6501`)
- Bayesian estimation of SARIMAX using PyMC3 NUTS  (:pr:`6528`)
- State space: compute smoothed state autocovariance matrices for arbitrary lags  (:pr:`6579`)
- `MLEResults.states.predicted` has wrong index  (:pr:`6580`)
- State space: simulate with time-varying covariance matrices.  (:pr:`6607`)
- State space: error with collapsed observations when missing  (:pr:`6613`)
- Notebook describing how to create state space custom models  (:pr:`6682`)
- Fix covariance estimation in parameterless models  (:pr:`6688`)
- Fix state space linting errors.  (:pr:`6698`)
- Decomposition of forecast updates in state space models due to the "news"  (:pr:`6765`)
- Dataframe/series concatenation in statespace results append  (:pr:`6768`)
- Pass cov_type, cov_kwargs through ARIMA.fit  (:pr:`6770`)
- Improve univariate smoother performance  (:pr:`6797`)
- Add `news` example notebook image.  (:pr:`6800`)
- Fix extend in VARMAX with trend  (:pr:`6909`)
- Dynamic is incorrect when not an int in statespace get_prediction  (:pr:`6917`)
- Add dynamic factor model with EM algorithm, option for monthly/quarterly mixed frequency model  (:pr:`6937`)
- Improve ETS / statespace documentation and higlights for v0.12   (:pr:`6950`)
- SARIMAX throwing different errors when length of endogenous var is too low  (:pr:`6961`)
- Fix start params computation with few nobs  (:pr:`6962`)
- Relax tolerance on random failure  (:pr:`6963`)

``tsa.vector.ar``
~~~~~~~~~~~~~~~~~
- Include figsize as parameter for IRF plot  (:pr:`6590`)
- Raise on incorrectly sized exog  (:pr:`6730`)
- Correct IRF nobs with exog  (:pr:`6925`)

bug-wrong
---------

A new issue label `type-bug-wrong` indicates bugs that cause that incorrect
numbers are returned without warnings.
(Regular bugs are mostly usability bugs or bugs that raise an exception for
unsupported use cases.)
`see tagged issues <https://github.com/statsmodels/statsmodels/issues?q=is%3Aissue+label%3Atype-bug-wrong+is%3Aclosed+milestone%3A0.12/>`_


Major Bugs Fixed
================

See github issues for a list of bug fixes included in this release

- `Closed bugs <https://github.com/statsmodels/statsmodels/pulls?utf8=%E2%9C%93&q=is%3Apr+is%3Amerged+milestone%3A0.12+label%3Atype-bug/>`_
- `Closed bugs (wrong result) <https://github.com/statsmodels/statsmodels/pulls?q=is%3Apr+is%3Amerged+milestone%3A0.12+label%3Atype-bug-wrong/>`_


Development summary and credits
===============================

Besides receiving contributions for new and improved features and for bugfixes,
important contributions to general maintenance for this release came from

- Chad Fulton
- Brock Mendel
- Peter Quackenbush
- Kerby Shedden
- Kevin Sheppard

and the general maintainer and code reviewer

- Josef Perktold

Additionally, many users contributed by participation in github issues and
providing feedback.

Thanks to all of the contributors for the 0.12 release (based on git log):

- Alex Lyttle
- Amund Vedal
- Baran Karakus
- Batakrishna Sahu
- Chad Fulton
- Cinthia M. Tanaka
- Haoyu Qi
- Hassan Kibirige
- He Yang
- Henning Blunck
- Jimmy2027
- Joon Ro
- Joonsuk Park
- Josef Perktold
- Kerby Shedden
- Kevin Rose
- Kevin Sheppard
- Manmeet Kumar Chaudhuri
- Markus Löning
- Martin Larralde
- Nolan Conaway
- Paulo Galuzio
- Peter Prescott
- Peter Quackenbush
- Samuel Scherrer
- Sean Lane
- Sebastian Pölsterl
- Skipper Seabold
- Thomas Brooks
- Tim Gates
- Victor Ananyev
- Wouter De Coster
- Zhiqing Xiao
- adrienpacifico
- aeturrell
- cd
- eirki
- pag
- partev
- w31ha0


These lists of names are automatically generated based on git log, and may not
be complete.

Merged Pull Requests
--------------------

The following Pull Requests were merged since the last release:

- :pr:`4107`: ENH: multivariate mean tests and confint
- :pr:`6411`: ENH: state space: add Chandrasekhar recursions
- :pr:`6433`: TST/BUG: use reset_randomstate
- :pr:`6438`: BUG: Change default optimizer for glm/ridge and make it user-settable
- :pr:`6452`: DOC: Fix the version that appears in the documentation
- :pr:`6456`: DOC: Send log to dev/null/
- :pr:`6457`: DOC: Special docbuild
- :pr:`6460`: Revert "DOC: Special docbuild"
- :pr:`6461`: MAINT: correcting typo
- :pr:`6465`: MAINT: Avoid noise in f-pvalue
- :pr:`6466`: MAINT: Replace Python 3.5 with 3.8 on Azure
- :pr:`6467`: MAINT: Update supported versions
- :pr:`6469`: MAINT: Fix future warnings
- :pr:`6470`: BUG: fix tukey-hsd for 1 pvalue 
- :pr:`6471`: MAINT: Fix issue with ragged array
- :pr:`6473`: MAINT: Avoid future error
- :pr:`6474`: BLD: Use pip on Azure
- :pr:`6475`: BUG: Fix exposure/offset handling in GEEResults
- :pr:`6477`: BUG: Silence expected visible deprecation warning
- :pr:`6490`: BLD: Attempt to cache key docbuild files
- :pr:`6491`: BLD: Improve doc caching
- :pr:`6492`: MAINT: Remove Python 3.5 references
- :pr:`6493`: MAINT: Avoid calling depr code
- :pr:`6495`: MAINT: Use travis cache and optimize build times
- :pr:`6501`: ENH: state space: add "Cholesky factor algorithm" simulation smoothing
- :pr:`6508`: ENH: Add option for original Breusch-Pagan heteroscedasticity test
- :pr:`6514`: ENH: use GLM starting values for QIF
- :pr:`6515`: BUG: fix #6511
- :pr:`6518`: Fix simple typo: various
- :pr:`6520`: BUG: fix GAM for 1-dim exog_linear 
- :pr:`6521`: REF/BUG: don't attach patsy constraint instance 
- :pr:`6528`: DOC: Bayesian estimation of SARIMAX using PyMC3 NUTS
- :pr:`6531`: DOC: fix typos
- :pr:`6534`: MAINT: Relax tolerance on test that occasionally fails
- :pr:`6547`: BUG: Fix domain check
- :pr:`6556`: BUG: fixes #6553, sliced predicted values according to predicted index
- :pr:`6560`: ENH: Holt-Winters simulations
- :pr:`6579`: ENH: state space: compute smoothed state autocovariance matrices for arbitrary lags
- :pr:`6580`: BUG: `MLEResults.states.predicted` has wrong index
- :pr:`6582`: ENH: Allow GEE weights to vary within clusters
- :pr:`6587`: BLD: Azure: Mac OSX 10.13 -> 10.14
- :pr:`6588`: MAINT: Relax tolerance on test that randomly fails
- :pr:`6590`: ENH: Include figsize as parameter for IRF plot
- :pr:`6601`: DOC: Update interactions_anova.ipynb
- :pr:`6607`: BUG: state space: simulate with time-varying covariance matrices.
- :pr:`6608`: BUG: statsmodels.regression.linear_model.OLS.fit_regularized fails to generate correct answer (#6604)
- :pr:`6613`: BUG: state space: error with collapsed observations when missing
- :pr:`6614`: DOC/ENH: example notebook (r): stationarity and detrending (ADF/KPSS)
- :pr:`6616`: DOC: Fix `true` type on statespace docs page
- :pr:`6621`: ENH: Calculate AR covariance parameters for gridded data
- :pr:`6622`: ENH Allow optional regularization in local fdr
- :pr:`6626`: ENH: allow more than 2 groups for survdiff in statsmodels.duration
- :pr:`6628`: BUG: Ensure text comparison is lower
- :pr:`6631`: DOC/TST: minor fixes for holtwinters simulate
- :pr:`6632`: ENH: add meta-analysis (basic methods)
- :pr:`6653`: MAINT: Fix appveyor/conda
- :pr:`6654`: ENH: Handle pathlib.Path objects
- :pr:`6656`: DOC: change OLS example to use datasets
- :pr:`6657`: BUG: fix constraints and bunds when use scipy.optimize.minimize
- :pr:`6662`: DOC: Fix AutoReg docstring
- :pr:`6664`: PERF: Speed up HC2/HC3 standard error calculation, using less memory
- :pr:`6671`: MAINT: Delete empty directory
- :pr:`6675`: ENH: add two independent proportion inference rebased
- :pr:`6677`: ENH: rates, poisson means two-sample comparison  rebased
- :pr:`6678`: ENH: stats.base, add HolderTuple, Holder class with indexing
- :pr:`6680`: DOC: Fix `fdrcorrection` docstring missing `is_sorted` parameter
- :pr:`6682`: ENH/DOC: Notebook describing how to create state space custom models
- :pr:`6688`: BUG: Fix covariance estimation in parameterless models
- :pr:`6693`: ENH: add covariance structure hypothesis tests
- :pr:`6697`: ENH: Warn for non-convergence in elastic net
- :pr:`6698`: CLN: Fix state space linting errors.
- :pr:`6699`: ENH: New exponential smoothing implementation
- :pr:`6704`: DOC: Add new badges
- :pr:`6705`: BUG\ENH: Improve missing value handling in PCA
- :pr:`6709`: DOC: Fix number if notebook text
- :pr:`6710`: MAINT: Flake8 fixes
- :pr:`6711`: ENH: Improve warning message in KPSS
- :pr:`6712`: MAINT: Remove deprecated keyword
- :pr:`6713`: BUG: Ensure sigma estimate is positive in KDE
- :pr:`6714`: BUG: Fix summary col R2 ordering
- :pr:`6715`: MAINT: Remove OrderedDict
- :pr:`6719`: DOC: Improve Factor and related docstrings
- :pr:`6722`: BUG: Change trend initialization in STL
- :pr:`6723`: ENH: Add check in test_whiteness
- :pr:`6726`: DOC: Improve explantion of missing values in ACF and related
- :pr:`6727`: ENH: Raise exception when recursive residual is not well defined
- :pr:`6728`: MAINT: Remove dtype np.integer for avoid Dep Warning
- :pr:`6730`: BUG: Raise on incorrectly sized exog
- :pr:`6732`: DOC: Notebook for quasibinomial regression
- :pr:`6733`: MAINT: Update pip-pre links
- :pr:`6738`: DOC: Improve "conservative" doc
- :pr:`6742`: Update broken link
- :pr:`6744`: ENH: Speed up banddepth calculations
- :pr:`6746`: DOC: Fix broken links with 404 error
- :pr:`6747`: BUG: Insufficient input checks in QuantReg
- :pr:`6751`: ENH: Add deterministic processes
- :pr:`6752`: MAINT: Spelling and small fixes
- :pr:`6758`: DOC: Demonstrate variance components analysis
- :pr:`6765`: ENH: Decomposition of forecast updates in state space models due to the "news"
- :pr:`6766`: PERF: Sparse matrices in MixedLM
- :pr:`6767`: ENH: Add Theta forecasting method
- :pr:`6768`: BUG: dataframe/series concatenation in statespace results append
- :pr:`6770`: BUG: pass cov_type, cov_kwargs through ARIMA.fit
- :pr:`6775`: DOC: Make deprecations more visible
- :pr:`6782`: ENH: Mediation support for PH regression
- :pr:`6785`: ENH: automatic lag selection for Box-Pierce, Ljung-Box #6645
- :pr:`6789`: ENH: Stats robust rebased2
- :pr:`6797`: ENH: improve univariate smoother performance
- :pr:`6800`: DOC: Add `news` example notebook image.
- :pr:`6806`: BUG: Fix access to normal_reference_constant
- :pr:`6810`: ENH: Hotelling's Two Sample Mean Test
- :pr:`6811`: MAINT: Remove error on FutureWarning
- :pr:`6817`: MAINT: Fix failing tests
- :pr:`6818`: BUG: Fix logic in labeling corr plot
- :pr:`6825`: DOC: Numpydoc signatures
- :pr:`6827`: BUG: Fix missing str
- :pr:`6828`: MAINT: Replace Warnings with Notes in regression summary
- :pr:`6829`: ENH: Add support for PeriodIndex to AutoReg
- :pr:`6830`: ENH: Return column information in add_constant
- :pr:`6831`: BUG: Correct shape of fvalue and f_pvalue
- :pr:`6832`: BUG: error in append for ARIMA model with trend
- :pr:`6834`: ENH: Add QR-based matrix rank
- :pr:`6835`: TST: stats moment_helpers use random state in unit test
- :pr:`6836`: MAINT: Catch warnings in discrete
- :pr:`6837`: DOC: Correct reference in docs
- :pr:`6838`: ENH: Add expanding initialization to RollingOLS/WLS
- :pr:`6839`: REF: Rename unbiased to adjusted
- :pr:`6841`: DOC: Include dot_plot
- :pr:`6842`: MAINT: numpydoc should work now
- :pr:`6843`: MAINT: Deprecate categorical
- :pr:`6844`: ENH: Add  a note when R2 is uncentered
- :pr:`6845`: MAINT: Remove redundant definition
- :pr:`6846`: BUG: Ensure PACF lag length is sensible
- :pr:`6847`: BUG: Allow Series as exog in predict
- :pr:`6848`: Updated durbin_watson Docstring and Tests
- :pr:`6849`: TST: Relax tolerance on test that fails Win32
- :pr:`6850`: MAINT: Fix error on nightly build
- :pr:`6852`: Gh 6627
- :pr:`6853`: DOC: Explain low df in cluster
- :pr:`6854`: BUG: Raise on nonstationary parameters when attempting to use GLS
- :pr:`6855`: MAINT: Correct debugging info
- :pr:`6856`: MAINT: Relax test tolerance
- :pr:`6859`: DOC: add recent stats addition to docs
- :pr:`6862`: DOC: Fix common doc errors
- :pr:`6865`: MAINT: Mark VAR from_formula as NotImplemented
- :pr:`6867`: BUG: Limit maxlags in VAR
- :pr:`6868`: TST: Refactor factor tests again
- :pr:`6869`: BUG: Fix indexing with HoltWinters's forecast
- :pr:`6870`: REF: Refactor Holt-Winters
- :pr:`6871`: MAINT: Allow skip if rdataset fails
- :pr:`6872`: REF/DOC docs and refactor of recent stats
- :pr:`6874`: DOC: Small doc fixes
- :pr:`6877`: BUG: fix raise exception on granger causality test
- :pr:`6879`: DOC: Fix issues in docs related to exponential smoothing
- :pr:`6882`: ENH: get_prediction method for ETS
- :pr:`6884`: TST: ets: test for simple exponential smoothing convergence
- :pr:`6885`: MAINT: Improve lint
- :pr:`6888`: BUG: Correct dimension when data removed
- :pr:`6892`: ENH: added diagnostics test to ETS model
- :pr:`6893`: MAINT: Change default lag in serial correlation tests
- :pr:`6894`: MAINT: Ensure setuptools is imported first
- :pr:`6895`: ENH: Added label option to summary.to_latex()
- :pr:`6897`: REF/DOC: api cleanup and improve docstrings in stats, round 3
- :pr:`6900`: ENH: Fixed the shifted column names in summary.to_latex()
- :pr:`6902`: DOC: Spelling and other doc fixes
- :pr:`6903`: DOC: Correct spacing around colon in docstrings
- :pr:`6904`: BUG: Stop transforming ES components
- :pr:`6905`: ENH: removed additional hline between tabulars
- :pr:`6906`: ENH: Enable qqplot_2sample to handle uneven samples
- :pr:`6908`: ENH: Add xvals param to lowess smoother
- :pr:`6909`: BUG: fix extend in VARMAX with trend
- :pr:`6910`: ENH: Support frozen dist in ProbPlots
- :pr:`6911`: ENH: Add STL Forecasting method
- :pr:`6915`: BUG: Fixed BSplines to match existing docs
- :pr:`6917`: BUG: dynamic is incorrect when not an int in statespace get_prediction
- :pr:`6920`: MAINT: Remove FutureWarnings
- :pr:`6922`: ENH: Add tool to simplify documenting API in release notes
- :pr:`6923`: DOC: Initial 0.12 Release Note
- :pr:`6925`: BUG: Correct IRF nobs with exog
- :pr:`6926`: ENH: Add Root Mean Square Percentage Error
- :pr:`6927`: ENH: Add get_prediction to AutoReg
- :pr:`6931`: DOC/MAINT: Fix doc errors and silence warning
- :pr:`6932`: DOC: Clarify deprecations
- :pr:`6933`: MAINT: Standardize forecast API
- :pr:`6934`: MAINT: Fix small issues post ETS get_prediction merge
- :pr:`6937`: ENH: Add dynamic factor model with EM algorithm, option for monthly/quarterly mixed frequency model
- :pr:`6938`: ENH: Add improved .cdf() and .ppf() to discrete distributions
- :pr:`6939`: BUG: remove k_extra from effects_idx
- :pr:`6940`: TST: Improve count model tests
- :pr:`6941`: REF: Change of BIC formula in GLM
- :pr:`6942`: Make glm's predict function return numpy array even if exposure is a pandas series
- :pr:`6943`: DOC: Document exceptions and warnings
- :pr:`6944`: ENH: improve descriptivestats
- :pr:`6946`: update pandas function in hp_filter example

API Changes
===========

Notable New Classes
-------------------
* :class:`statsmodels.stats.descriptivestats.Description`
* :class:`statsmodels.stats.meta_analysis.CombineResults`
* :class:`statsmodels.stats.robust_compare.TrimmedMean`
* :class:`statsmodels.tools.sm_exceptions.ParseError`
* :class:`statsmodels.tsa.base.prediction.PredictionResults`
* :class:`statsmodels.tsa.deterministic.CalendarDeterministicTerm`
* :class:`statsmodels.tsa.deterministic.CalendarFourier`
* :class:`statsmodels.tsa.deterministic.CalendarSeasonality`
* :class:`statsmodels.tsa.deterministic.CalendarTimeTrend`
* :class:`statsmodels.tsa.deterministic.DeterministicProcess`
* :class:`statsmodels.tsa.deterministic.DeterministicTerm`
* :class:`statsmodels.tsa.deterministic.Fourier`
* :class:`statsmodels.tsa.deterministic.FourierDeterministicTerm`
* :class:`statsmodels.tsa.deterministic.Seasonality`
* :class:`statsmodels.tsa.deterministic.TimeTrend`
* :class:`statsmodels.tsa.deterministic.TimeTrendDeterministicTerm`
* :class:`statsmodels.tsa.exponential_smoothing.ets.ETSModel`
* :class:`statsmodels.tsa.exponential_smoothing.ets.ETSResults`
* :class:`statsmodels.tsa.exponential_smoothing.ets.PredictionResults`
* :class:`statsmodels.tsa.forecasting.stl.STLForecast`
* :class:`statsmodels.tsa.forecasting.stl.STLForecastResults`
* :class:`statsmodels.tsa.forecasting.theta.ThetaModel`
* :class:`statsmodels.tsa.forecasting.theta.ThetaModelResults`
* :class:`statsmodels.tsa.statespace.cfa_simulation_smoother.CFASimulationSmoother`
* :class:`statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ`
* :class:`statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQResults`
* :class:`statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQStates`
* :class:`statsmodels.tsa.statespace.dynamic_factor_mq.FactorBlock`
* :class:`statsmodels.tsa.statespace.news.NewsResults`

New Methods
-----------
* :meth:`statsmodels.tsa.statespace.mlemodel.MLEResults.news`
* :meth:`statsmodels.tsa.statespace.mlemodel.ExponentialSmoothingResults.news`
* :meth:`statsmodels.tsa.statespace.mlemodel.DynamicFactorResults.news`
* :meth:`statsmodels.tsa.statespace.representation.KalmanSmoother.diff_endog`
* :meth:`statsmodels.tsa.statespace.mlemodel.UnobservedComponentsResults.news`
* :meth:`statsmodels.tsa.statespace.mlemodel.RecursiveLSResults.news`
* :meth:`statsmodels.tsa.vector_ar.var_model.VAR.from_formula`
* :meth:`statsmodels.tsa.statespace.mlemodel.SARIMAXResults.news`
* :meth:`statsmodels.genmod.generalized_linear_model.GLMGamResults.bic`
* :meth:`statsmodels.tsa.ar_model.AutoRegResults.forecast`
* :meth:`statsmodels.tsa.ar_model.AutoRegResults.get_prediction`
* :meth:`statsmodels.tsa.statespace.representation.Representation.diff_endog`
* :meth:`statsmodels.genmod.generalized_linear_model.GLMResults.bic`
* :meth:`statsmodels.tsa.statespace.representation.KalmanFilter.diff_endog`
* :meth:`statsmodels.tsa.statespace.kalman_smoother.SmootherResults.smoothed_state_autocovariance`
* :meth:`statsmodels.tsa.statespace.kalman_smoother.SmootherResults.smoothed_state_gain`
* :meth:`statsmodels.tsa.statespace.kalman_smoother.SmootherResults.news`
* :meth:`statsmodels.tsa.statespace.representation.SimulationSmoother.diff_endog`
* :meth:`statsmodels.tsa.statespace.mlemodel.ARIMAResults.news`
* :meth:`statsmodels.tsa.arima.model.ARIMAResults.append`
* :meth:`statsmodels.tsa.statespace.mlemodel.VARMAXResults.news`

Removed Methods
---------------
* ``statsmodels.genmod._prediction.PredictionResults.se_obs``
* ``statsmodels.genmod._prediction.PredictionResults.t_test``
* ``statsmodels.genmod._prediction.PredictionResults.tvalues``
* ``statsmodels.base.model.VAR.from_formula``
* ``statsmodels.tsa.statespace.mlemodel.ARIMAResults.append``

Classes and Methods with New Arguments
--------------------------------------
* :meth:`statsmodels.tsa.statespace.mlemodel.MLEResults`: ``copy_initialization``
* :meth:`statsmodels.tsa.statespace.exponential_smoothing.ExponentialSmoothingResults`: ``truncate_endog_names``
* :meth:`statsmodels.tsa.statespace.dynamic_factor.DynamicFactorResults`: ``copy_initialization``
* :meth:`statsmodels.tsa.statespace.kalman_smoother.KalmanSmoother`: ``update_smoother``, ``update_filter``, ``update_representation``
* :meth:`statsmodels.tsa.statespace.structural.UnobservedComponentsResults`: ``truncate_endog_names``
* :meth:`statsmodels.iolib.summary2.Summary`: ``label``
* :meth:`statsmodels.regression.recursive_ls.RecursiveLSResults`: ``copy_initialization``
* :meth:`statsmodels.discrete.discrete_model.GeneralizedPoisson`: ``check_rank``
* :meth:`statsmodels.discrete.discrete_model.Logit`: ``check_rank``
* :meth:`statsmodels.tsa.statespace.sarimax.SARIMAXResults`: ``copy_initialization``
* :meth:`statsmodels.discrete.discrete_model.DiscreteModel`: ``check_rank``
* :meth:`statsmodels.discrete.discrete_model.CountModel`: ``check_rank``
* :meth:`statsmodels.regression.rolling.RollingWLS`: ``expanding``
* :meth:`statsmodels.genmod.cov_struct.Autoregressive`: ``grid``
* :meth:`statsmodels.discrete.discrete_model.NegativeBinomial`: ``check_rank``
* :meth:`statsmodels.stats.mediation.Mediation`: ``outcome_predict_kwargs``
* :meth:`statsmodels.discrete.discrete_model.MultinomialModel`: ``check_rank``
* :meth:`statsmodels.tsa.ar_model.AutoReg`: ``old_names``, ``deterministic``
* :meth:`statsmodels.discrete.discrete_model.Poisson`: ``check_rank``
* :meth:`statsmodels.discrete.discrete_model.BinaryModel`: ``check_rank``
* :meth:`statsmodels.tsa.statespace.simulation_smoother.SimulationSmoother`: ``update_smoother``, ``update_filter``, ``update_representation``
* :meth:`statsmodels.tsa.arima.model.ARIMAResults`: ``copy_initialization``
* :meth:`statsmodels.tsa.vector_ar.irf.IRAnalysis`: ``figsize``
* :meth:`statsmodels.tsa.statespace.varmax.VARMAXResults`: ``truncate_endog_names``
* :meth:`statsmodels.discrete.discrete_model.NegativeBinomialP`: ``check_rank``
* :meth:`statsmodels.duration.hazard_regression.PHReg`: ``pred_only``
* :meth:`statsmodels.duration.hazard_regression.rv_discrete_float`: ``n``
* :meth:`statsmodels.discrete.discrete_model.MNLogit`: ``check_rank``
* :meth:`statsmodels.regression.rolling.RollingOLS`: ``expanding``
* :meth:`statsmodels.discrete.discrete_model.Probit`: ``check_rank``

Classes with Removed Arguments
------------------------------
* :class:`statsmodels.tsa.statespace.mlemodel.PredictionResults`
   * New: ``PredictionResults(endog, alpha)``
   * Old: ``PredictionResults(endog, what, alpha)``
* :class:`statsmodels.regression._prediction.PredictionResults`
   * New: ``PredictionResults(alpha)``
   * Old: ``PredictionResults(what, alpha)``
* :class:`statsmodels.genmod._prediction.PredictionResults`
   * New: ``PredictionResults(alpha)``
   * Old: ``PredictionResults(what, alpha)``

New Functions
-------------
* :func:`statsmodels.graphics.tukeyplot.tukeyplot`
* :func:`statsmodels.multivariate.plots.plot_loadings`
* :func:`statsmodels.multivariate.plots.plot_scree`
* :func:`statsmodels.stats.contrast.wald_test_noncent`
* :func:`statsmodels.stats.contrast.wald_test_noncent_generic`
* :func:`statsmodels.stats.descriptivestats.describe`
* :func:`statsmodels.stats.meta_analysis.combine_effects`
* :func:`statsmodels.stats.meta_analysis.effectsize_2proportions`
* :func:`statsmodels.stats.meta_analysis.effectsize_smd`
* :func:`statsmodels.stats.multivariate.confint_mvmean`
* :func:`statsmodels.stats.multivariate.confint_mvmean_fromstats`
* :func:`statsmodels.stats.multivariate.test_cov`
* :func:`statsmodels.stats.multivariate.test_cov_blockdiagonal`
* :func:`statsmodels.stats.multivariate.test_cov_diagonal`
* :func:`statsmodels.stats.multivariate.test_cov_oneway`
* :func:`statsmodels.stats.multivariate.test_cov_spherical`
* :func:`statsmodels.stats.multivariate.test_mvmean`
* :func:`statsmodels.stats.multivariate.test_mvmean_2indep`
* :func:`statsmodels.stats.oneway.anova_generic`
* :func:`statsmodels.stats.oneway.anova_oneway`
* :func:`statsmodels.stats.oneway.confint_effectsize_oneway`
* :func:`statsmodels.stats.oneway.confint_noncentrality`
* :func:`statsmodels.stats.oneway.convert_effectsize_fsqu`
* :func:`statsmodels.stats.oneway.effectsize_oneway`
* :func:`statsmodels.stats.oneway.equivalence_oneway`
* :func:`statsmodels.stats.oneway.equivalence_oneway_generic`
* :func:`statsmodels.stats.oneway.equivalence_scale_oneway`
* :func:`statsmodels.stats.oneway.f2_to_wellek`
* :func:`statsmodels.stats.oneway.fstat_to_wellek`
* :func:`statsmodels.stats.oneway.power_equivalence_oneway`
* :func:`statsmodels.stats.oneway.simulate_power_equivalence_oneway`
* :func:`statsmodels.stats.oneway.test_scale_oneway`
* :func:`statsmodels.stats.oneway.wellek_to_f2`
* :func:`statsmodels.stats.power.normal_power_het`
* :func:`statsmodels.stats.power.normal_sample_size_one_tail`
* :func:`statsmodels.stats.proportion.confint_proportions_2indep`
* :func:`statsmodels.stats.proportion.power_proportions_2indep`
* :func:`statsmodels.stats.proportion.samplesize_proportions_2indep_onetail`
* :func:`statsmodels.stats.proportion.score_test_proportions_2indep`
* :func:`statsmodels.stats.proportion.test_proportions_2indep`
* :func:`statsmodels.stats.proportion.tost_proportions_2indep`
* :func:`statsmodels.stats.rates.etest_poisson_2indep`
* :func:`statsmodels.stats.rates.test_poisson_2indep`
* :func:`statsmodels.stats.rates.tost_poisson_2indep`
* :func:`statsmodels.stats.robust_compare.scale_transform`
* :func:`statsmodels.stats.robust_compare.trim_mean`
* :func:`statsmodels.stats.robust_compare.trimboth`
* :func:`statsmodels.tools.eval_measures.rmspe`

Notable Removed Functions
-------------------------
* ``statsmodels.stats.diagnostic.unitroot_adf``
* ``statsmodels.tsa.stattools.periodogram``

Functions with New Arguments
----------------------------
* :func:`statsmodels.tsa.ar_model.ar_select_order`: ``old_names``
* :func:`statsmodels.stats.diagnostic.het_breuschpagan`: ``robust``
* :func:`statsmodels.nonparametric.smoothers_lowess.lowess`: ``xvals``
* :func:`statsmodels.stats.diagnostic.acorr_ljungbox`: ``auto_lag``
* :func:`statsmodels.stats.diagnostic.linear_harvey_collier`: ``skip``
* :func:`statsmodels.stats.multitest.local_fdr`: ``alpha``
* :func:`statsmodels.graphics.gofplots.plotting_pos`: ``b``
* :func:`statsmodels.graphics.gofplots.qqline`: ``lineoptions``

Functions with Keyword Name Changes
-----------------------------------
* :func:`statsmodels.tsa.arima.estimators.durbin_levinson.durbin_levinson`
   * New: ``durbin_levinson(endog, ar_order, demean, adjusted)``
   * Old: ``durbin_levinson(endog, ar_order, demean, unbiased)``
* :func:`statsmodels.tsa.stattools.pacf_ols`
   * New: ``pacf_ols(x, nlags, efficient, adjusted)``
   * Old: ``pacf_ols(x, nlags, efficient, unbiased)``
* :func:`statsmodels.graphics.tsaplots.plot_acf`
   * New: ``plot_acf(x, ax, lags, alpha, use_vlines, adjusted, fft, missing, title, zero, vlines_kwargs, kwargs)``
   * Old: ``plot_acf(x, ax, lags, alpha, use_vlines, unbiased, fft, missing, title, zero, vlines_kwargs, kwargs)``
* :func:`statsmodels.tsa.stattools.ccovf`
   * New: ``ccovf(x, y, adjusted, demean)``
   * Old: ``ccovf(x, y, unbiased, demean)``
* :func:`statsmodels.tools.tools.pinv_extended`
   * New: ``pinv_extended(x, rcond)``
   * Old: ``pinv_extended(X, rcond)``
* :func:`statsmodels.tsa.stattools.ccf`
   * New: ``ccf(x, y, adjusted)``
   * Old: ``ccf(x, y, unbiased)``
* :func:`statsmodels.tsa.stattools.acf`
   * New: ``acf(x, adjusted, nlags, qstat, fft, alpha, missing)``
   * Old: ``acf(x, unbiased, nlags, qstat, fft, alpha, missing)``
* :func:`statsmodels.tsa.stattools.acovf`
   * New: ``acovf(x, adjusted, demean, fft, missing, nlag)``
   * Old: ``acovf(x, unbiased, demean, fft, missing, nlag)``
* :func:`statsmodels.tsa.arima.estimators.yule_walker.yule_walker`
   * New: ``yule_walker(endog, ar_order, demean, adjusted)``
   * Old: ``yule_walker(endog, ar_order, demean, unbiased)``
