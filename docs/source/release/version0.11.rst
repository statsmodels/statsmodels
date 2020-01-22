:orphan:

==============
Release 0.11.0
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
**Issues Closed**: 335

**Pull Requests Merged**: 301


The Highlights
==============

Regression
----------
Rolling OLS and WLS are implemented in :class:`~statsmodels.regression.rolling.RollingOLS`
and :class:`~statsmodels.regression.rolling.RollingWLS`. These function similarly to the estimators
recently removed from pandas.

Statistics
----------
Add the Oaxaca-Blinder decomposition (:class:`~statsmodels.stats.oaxaca.OaxacaBlinder`) that
decomposes the difference in group means into with and between components.

Add the Distance dependence measures statistics
(:func:`~statsmodels.stats.dist_dependence_measures.distance_statistics`) and the Distance Covariance
test (:func:`~statsmodels.stats.dist_dependence_measures.distance_covariance_test`).

Statespace Models
-----------------

Linear exponential smoothing models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Model class: :class:`~statsmodels.tsa.statespace.ExponentialSmoothing`
- Alternative to :class:`~statsmodels.tsa.statespace.exponential_smoothing.ExponentialSmoothing`
- Only supports linear models
- Is part of the class of state space models and so inherits some additional
  functionality.

Methods to apply parameters fitted on one dataset to another dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Methods: ``extend``, ``append``, and ``apply``, for state space results classes
- These methods allow applying fitted parameters from a training dataset to a
  test dataset in various ways
- Useful for conveniently performing cross-validation exercises

Method to hold some parameters fixed at known values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Methods: :func:`statsmodels.tsa.statespace.mlemodel.MLEModel.fix_params` and
  :func:`statsmodels.tsa.statespace.mlemodel.MLEModel.fit_constrained` for state
  space model classes.
- These methods allow setting some parameters to known values and then
  estimating the remaining parameters

Option for low memory operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Argument: ``low_memory=True``, for ``fit``, ``filter``, and ``smooth``
- Only a subset of results are available when using this option, but it does
  allow for prediction, diagnostics, and forecasting
- Useful to speed up cross-validation exercises

Improved access to state estimates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Attribute: ``states``, for state space results classes
- Wraps estimated states (predicted, filtered, smoothed) as Pandas objects with
  the appropriate index.
- Adds human-readable names for states, where possible.

Improved simulation and impulse responses for time-varying models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Argument: ``anchor`` allows specifying the period after which to begin the simulation.
- Example: to simulate data following the sample period, use ``anchor='end'``

Time-Series Analysis
--------------------

STL Decomposition
~~~~~~~~~~~~~~~~~
- Class implementing the STL decomposition :class:`~statsmodels.tsa.seasonal.STL`.

New AR model
~~~~~~~~~~~~

- Model class: :class:`~statsmodels.tsa.ar_model.AutoReg`
- Estimates parameters using conditional MLE (OLS)
- Adds the ability to specify exogenous variables, include time trends,
  and add seasonal dummies.
- The function :class:`~statsmodels.tsa.ar_model.ar_select_order` performs lag length selection
  for AutoReg models.

New ARIMA model
~~~~~~~~~~~~~~~

- Model class: :class:`~statsmodels.tsa.arima.model.ARIMA`
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

Zivot-Andrews Test
~~~~~~~~~~~~~~~~~~
The Zivot-Andrews test for unit roots in the presence of structural breaks has
been added in :func:`~statsmodels.tsa.stattools.zivot_andrews`.

More robust regime switching models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Implementation of the Hamilton filter and Kim smoother in log space avoids
  underflow errors.


What's new - an overview
========================

The following lists the main new features of statsmodels 0.10. In addition,
release 0.10 includes bug fixes, refactorings and improvements in many areas.

Major Feature
-------------
- Allow fixing parameters in state space models  (:pr:`5735`)
- Add new version of ARIMA-type estimators (AR, ARIMA, SARIMAX)  (:pr:`5827`)
- Add STL decomposition for time series  (:pr:`5926`)
- Functional SIR  (:pr:`5963`)
- Zivot Andrews test  (:pr:`6014`)
- Added Oaxaca-Blinder Decomposition  (:pr:`6026`)
- Add rolling WLS and OLS  (:pr:`6028`)
- Replacement for AR  (:pr:`6087`)

Performance Improvements
------------------------
- Cythonize innovations algo and filter  (:pr:`5947`)
- Only perform required predict iterations in state space models  (:pr:`6064`)
- State space: Improve low memory usability; allow in fit, loglike  (:pr:`6071`)

Submodules
----------

``base``
~~~~~~~~
- Clarify xname length and purpose  (:pr:`5957`)
- Remove unnecessary pickle use  (:pr:`6091`)
- Fix accepting of eval environment for formula  (:pr:`6152`)
- Workaround NumPy ptp issue  (:pr:`6316`)


``discrete``
~~~~~~~~~~~~
- Test_constrained  (:pr:`5821`)
- Improve the cvxopt not found error  (:pr:`6163`)


``genmod``
~~~~~~~~~~
- Improvements to BayesMixedGLM docs, argument checking  (:pr:`5895`)
- Scale parameter handling in GEE  (:pr:`6208`)
- Add example notebook for GEE score tests  (:pr:`6299`)
- Fix bug in ridge for vector alpha  (:pr:`6442`)

``graphics``
~~~~~~~~~~~~
- Plot only unique censored points  (:pr:`6124`)
- Add missing keyword argument to plot_acf  (:pr:`6227`)
- And vlines option to plot_fit  (:pr:`6266`)
- Pass arguments through in plot_leverage_resid2  (:pr:`6281`)


``io``
~~~~~~
- Clarify summary2 documentation  (:pr:`6118`)


``nonparametric``
~~~~~~~~~~~~~~~~~
- Ensure BW is not 0  (:pr:`6292`)
- Check dtype in KDEUnivariate  (:pr:`6314`)
- Supporting custom kernel in local linear kernel regression  (:pr:`6375`)



``regression``
~~~~~~~~~~~~~~
- Test for anova_nistcertified  (:pr:`5797`)
- Remove no-longer-needed HC_se lookups  (:pr:`5841`)
- Dimension reduction for covariance matrices  (:pr:`5852`)
- Use class to define MixedLM variance components structure  (:pr:`5898`)
- Add rolling WLS and OLS  (:pr:`6028`)
- Prepare for Rolling Least Squares  (:pr:`6056`)
- Improve regression doc strings  (:pr:`6077`)
- Fix summary table header for mixedlm  (:pr:`6217`)


``robust``
~~~~~~~~~~
- Robust  (:pr:`5819`)
- Make mad function behave correctly when used on empty inputs  (:pr:`5968`)


``stats``
~~~~~~~~~
- Lilliefors min nobs not set  (:pr:`5610`)
- Replace alpha=0.05 with alpha=alpha  (:pr:`5998`)
- Added Oaxaca-Blinder Decomposition  (:pr:`6026`)
- Improve Ljung-Box  (:pr:`6079`)
- Correct thresholding in correlation tools  (:pr:`6105`)
- Use self.data consistently  (:pr:`6144`)
- Better argument checking for StratifiedTable  (:pr:`6294`)
- Restore multicomp  (:pr:`6320`)
- Improve Ljung Box diagnostics  (:pr:`6324`)
- Correct standardization in robust skewness  (:pr:`6374`)
- Distance dependence measures  (:pr:`6401`)
- Improve diagnostics   (:pr:`6410`)



``tools``
~~~~~~~~~
- Fix error introduced in isestimable  (:pr:`6081`)
- Fix axis in irq  (:pr:`6391`)



``tsa``
~~~~~~~
- Use cython fused types to simplify statespace code  (:pr:`5283`)
- Allow fixing parameters in state space models  (:pr:`5735`)
- Markov switching in log space: Hamilton filter / Kim smoother  (:pr:`5826`)
- Add new version of ARIMA-type estimators (AR, ARIMA, SARIMAX)  (:pr:`5827`)
- Exponential smoothing - damped trend gives incorrect param, predictions  (:pr:`5893`)
- State space: add methods to apply fitted parameters to new observations or new dataset  (:pr:`5915`)
- TVTP for Markov regression  (:pr:`5917`)
- Add STL decomposition for time series  (:pr:`5926`)
- Cythonize innovations algo and filter  (:pr:`5947`)
- Zivot Andrews test  (:pr:`6014`)
- Improve ARMA startparams  (:pr:`6018`)
- Fix ARMA so that it works with exog when trend=nc  (:pr:`6070`)
- Clean tsatools docs  (:pr:`6075`)
- Improve Ljung-Box  (:pr:`6079`)
- Replacement for AR  (:pr:`6087`)
- Incorrect TSA index if loc resolves to slice  (:pr:`6130`)
- Division by zero in exponential smoothing if damping_slope=0  (:pr:`6232`)
- Forecasts now ignore non-monotonic period index  (:pr:`6242`)
- Hannan-Rissanen third stage is invalid if non-stationary/invertible  (:pr:`6258`)
- Fix notebook  (:pr:`6279`)
- Correct VAR summary when model contains exog variables  (:pr:`6286`)
- Fix conf interval with MI  (:pr:`6297`)
- Ensure inputs are finite in granger causality test  (:pr:`6318`)
- Fix trend due to recent changes  (:pr:`6321`)
- Improve Ljung Box diagnostics  (:pr:`6324`)
- Documentation for release v0.11  (:pr:`6338`)
- Fix _get_index_loc with date strings  (:pr:`6340`)
- Use correct exog names  (:pr:`6389`)



``tsa.statespace``
~~~~~~~~~~~~~~~~~~
- Use cython fused types to simplify statespace code  (:pr:`5283`)
- Allow fixing parameters in state space models  (:pr:`5735`)
- Add new version of ARIMA-type estimators (AR, ARIMA, SARIMAX)  (:pr:`5827`)
- MLEModel now passes nobs to Representation  (:pr:`6050`)
- Only perform required predict iterations in state space models  (:pr:`6064`)
- State space: Improve low memory usability; allow in fit, loglike  (:pr:`6071`)
- State space: cov_params computation in fix_params context  (:pr:`6072`)
- Add conserve memory tests.  (:pr:`6073`)
- Improve cov_params in append, extend, apply  (:pr:`6074`)
- Seasonality in SARIMAX Notebook  (:pr:`6096`)
- Improve SARIMAX start_params if too few nobs  (:pr:`6102`)
- Fix score computation with fixed params  (:pr:`6104`)
- Add exact diffuse initialization as an option for SARIMAX, UnobservedComponents  (:pr:`6111`)
- Compute standardized forecast error in diffuse period if possible  (:pr:`6131`)
- Start_params for VMA model with exog.  (:pr:`6133`)
- Adds state space version of linear exponential smoothing models  (:pr:`6179`)
- State space: add wrapped states and, where possible, named states  (:pr:`6181`)
- Allow dynamic factor starting parameters computation with NaNs values  (:pr:`6231`)
- Dynamic factor model use AR model for error start params if error_var=False  (:pr:`6233`)
- SARIMAX index behavior with simple_differencing=True  (:pr:`6239`)
- Parameter names in DynamicFactor for unstructured error covariance matrix  (:pr:`6240`)
- SARIMAX: basic validation for order, seasonal_order  (:pr:`6241`)
- Update SARIMAX to use SARIMAXSpecification for more consistent input handling  (:pr:`6250`)
- State space: Add finer-grained memory conserve settings  (:pr:`6254`)
- Cloning of arima.ARIMA models.  (:pr:`6260`)
- State space: saving fixed params w/ extend, apply, append  (:pr:`6261`)
- State space: Improve simulate, IRF, prediction  (:pr:`6280`)
- State space: deprecate out-of-sample w/ unsupported index  (:pr:`6332`)
- State space: integer params can cause imaginary output  (:pr:`6333`)
- Append, extend check that index matches model  (:pr:`6334`)
- Fix k_exog, k_trend in arima.ARIMA; raise error when cloning a model with exog if no new exog given  (:pr:`6337`)
- Documentation for release v0.11  (:pr:`6338`)
- RecursiveLS should not allow `fix_params` method.  (:pr:`6415`)
- More descriptive error message for recursive least squares parameter constraints.  (:pr:`6424`)
- Diffuse multivariate case w/ non-diagonal observation innovation covariance matrix  (:pr:`6425`)



``tsa.vector.ar``
~~~~~~~~~~~~~~~~~
- Raise in GC test for VAR(0)  (:pr:`6285`)
- Correct VAR summary when model contains exog variables  (:pr:`6286`)
- Use correct exog names  (:pr:`6389`)


Build
-----
- Ignore warns on 32 bit linux  (:pr:`6005`)
- Travis CI: The sudo: tag is deprecated in Travis  (:pr:`6161`)
- Relax precision for ppc64el  (:pr:`6222`)

Documentation
-------------
- Remove orphaned docs files  (:pr:`5832`)
- Array-like -> array_like  (:pr:`5929`)
- Change some more links to https  (:pr:`5937`)
- Fix self-contradictory minimum dependency versions  (:pr:`5939`)
- Fix formula for log-like in WLS  (:pr:`5946`)
- Fix typo  (:pr:`5949`)
- Add parameters for CountModel predict  (:pr:`5986`)
- Fix many spelling errors  (:pr:`5992`)
- Small fixups after the spell check  (:pr:`5994`)
- Clarify that GARCH models are deprecated  (:pr:`6000`)
- Added content for two headings in VAR docs  (:pr:`6022`)
- Fix regression doc strings  (:pr:`6031`)
- Add doc string check to doc build  (:pr:`6036`)
- Apply documentation standardizations  (:pr:`6038`)
- Fix spelling  (:pr:`6041`)
- Merge pull request #6041 from bashtage/doc-fixes  (:pr:`6042`)
- Fix notebook due to pandas index change  (:pr:`6044`)
- Remove warning due to deprecated features  (:pr:`6045`)
- Remove DynamicVAR  (:pr:`6046`)
- Small doc site improvements  (:pr:`6048`)
- Small fix ups for modernized size  (:pr:`6052`)
- More small doc fixes  (:pr:`6053`)
- Small changes to doc building  (:pr:`6054`)
- Use the working branch of numpy doc  (:pr:`6055`)
- Fix spelling in notebooks  (:pr:`6057`)
- Fix missing spaces around colon  (:pr:`6058`)
- Continue fixing docstring formatting  (:pr:`6060`)
- Fix web font size  (:pr:`6062`)
- Fix web font size  (:pr:`6063`)
- Fix doc errors affecting build  (:pr:`6067`)
- Improve docs in tools and ar_model  (:pr:`6080`)
- Improve filter docstrings  (:pr:`6082`)
- Spelling and notebook link  (:pr:`6085`)
- Website fix  (:pr:`6089`)
- Changes summary_col's docstring to match variables  (:pr:`6106`)
- Update spelling in CONTRIBUTING.rst  (:pr:`6107`)
- Update link in CONTRIBUTING.rst  (:pr:`6108`)
- Update PR template Numpy guide link  (:pr:`6110`)
- Added interpretations to LogitResults.get_margeff  (:pr:`6113`)
- Improve docstrings  (:pr:`6116`)
- Switch doc theme  (:pr:`6119`)
- Add initial API doc  (:pr:`6120`)
- Small improvements to docs  (:pr:`6122`)
- Switch doc icon  (:pr:`6123`)
- Fix doc build failure  (:pr:`6125`)
- Update templates and add missing API functions  (:pr:`6126`)
- Add missing functions from the API  (:pr:`6134`)
- Restructure the documentation  (:pr:`6136`)
- Add a new logo  (:pr:`6142`)
- Fix validator so that it works  (:pr:`6143`)
- Add formula API  (:pr:`6145`)
- Fix sidebar TOC  (:pr:`6160`)
- Warn that only trusted files should be unpickled  (:pr:`6162`)
- Update pickle warning  (:pr:`6166`)
- Fix warning format  (:pr:`6167`)
- Clarify req for cvxopt  (:pr:`6198`)
- Spelling and Doc String Fixes  (:pr:`6204`)
- Fix a typo  (:pr:`6214`)
- Fix typos in install.rst  (:pr:`6215`)
- Fix a typo  (:pr:`6216`)
- Docstring fixes  (:pr:`6235`)
- Fix spelling in notebooks  (:pr:`6257`)
- Clarify patsy 0.5.1 is required  (:pr:`6275`)
- Fix notebook  (:pr:`6279`)
- Close issues  (:pr:`6283`)
- Doc string changes  (:pr:`6289`)
- Correct spells  (:pr:`6298`)
- Add simple, documented script to get github info  (:pr:`6303`)
- Update test running instructions  (:pr:`6317`)
- Restore test() autosummary  (:pr:`6319`)
- Fix alpha description for GLMGam  (:pr:`6322`)
- Move api docs  (:pr:`6327`)
- Update Release Note  (:pr:`6342`)
- Fix documentation errors  (:pr:`6343`)
- Fixes in preparation for release  (:pr:`6344`)
- Further doc fixes  (:pr:`6345`)
- Fix minor doc errors  (:pr:`6347`)
- Git notes  (:pr:`6348`)
- Finalize release notes for 0.11  (:pr:`6349`)
- Add version dropdown  (:pr:`6350`)
- Finalize release note  (:pr:`6353`)
- Change generated path  (:pr:`6363`)
- Doc updates  (:pr:`6368`)
- Improve doc strings  (:pr:`6369`)
- Clarify demeaning in ljungbox  (:pr:`6390`)
- Fix ridge regression formula in hpfilter  (:pr:`6398`)
- Fix link  (:pr:`6407`)
- Update release note for v0.11.0rc2  (:pr:`6416`)
- Replace array with ndarray (:pr:`6447`)
- Final release note change for 0.11 (:pr:`6450`)


Maintenance
-----------
- Implement cached_value, cached_data proof of concept  (:pr:`4421`)
- Use Appender pattern for docstrings  (:pr:`5235`)
- Remove sandbox.formula, supplanted by patsy  (:pr:`5692`)
- Remove docstring'd-out traceback for code that no longer raises  (:pr:`5757`)
- Enable/mark mangled/commented-out tests  (:pr:`5768`)
- Implement parts of #5220, deprecate ancient aliases  (:pr:`5784`)
- Catch warnings produced during tests  (:pr:`5799`)
- Parts of iolib  (:pr:`5814`)
- E701 multiple statements on one line (colon)  (:pr:`5842`)
- Remove ex_pairwise file dominated by try_tukey_hsd  (:pr:`5856`)
- Fix pandas compat  (:pr:`5892`)
- Use pytest.raises to check error message  (:pr:`5897`)
- Bump dependencies  (:pr:`5910`)
- Fix pandas imports  (:pr:`5922`)
- Remove Python 2.7 from Appveyor  (:pr:`5927`)
- Relax tol on test that randomly fails  (:pr:`5931`)
- Fix test that fails with positive probability  (:pr:`5933`)
- Port parts of #5220  (:pr:`5935`)
- Remove Python 2.7 from travis  (:pr:`5938`)
- Fix linting failures  (:pr:`5940`)
- Drop redundant travis configs  (:pr:`5950`)
- Mark MPL test as MPL  (:pr:`5954`)
- Deprecate periodogram  (:pr:`5958`)
- Ensure seaborn is available for docbuild  (:pr:`5960`)
- Cython cleanups  (:pr:`5962`)
- Remove PY3  (:pr:`5965`)
- Remove future and Python 2.7  (:pr:`5969`)
- Remove string_types in favor of str  (:pr:`5972`)
- Restore ResettableCache  (:pr:`5976`)
- Cleanup legacy imports  (:pr:`5977`)
- Follow-up to #5956  (:pr:`5982`)
- Clarify breusch_pagan is for scalars  (:pr:`5984`)
- Add W605 to lint codes  (:pr:`5987`)
- Follow-up to #5928  (:pr:`5988`)
- Add spell checking  (:pr:`5990`)
- Remove comment no longer relevant  (:pr:`5991`)
- Refactor X13 testing  (:pr:`6001`)
- Standardized on nlags for acf/pacf  (:pr:`6002`)
- Rename forecast years to forecast periods  (:pr:`6007`)
- Improve testing of seasonal decompose  (:pr:`6011`)
- Remove notes about incorrect test  (:pr:`6015`)
- Turn relative import into an absolute import  (:pr:`6030`)
- Change types for future changes in NumPy  (:pr:`6039`)
- Move garch to archive/  (:pr:`6059`)
- Fix small lint issue  (:pr:`6066`)
- Stop testing on old, buggy SciPy  (:pr:`6069`)
- Small fixes in preparation for larger changes  (:pr:`6088`)
- Add tools for programatically manipulating docstrings  (:pr:`6090`)
- Ensure r download cache works  (:pr:`6092`)
- Fix new cache name  (:pr:`6093`)
- Fix wrong test  (:pr:`6094`)
- Remove extra LICENSE.txt and setup.cfg  (:pr:`6117`)
- Be compatible with scipy 1.3  (:pr:`6164`)
- Don't assume that 'python' is Python 3  (:pr:`6165`)
- Exclude pytest-xdist 1.30  (:pr:`6205`)
- Add Python 3.8 environment  (:pr:`6246`)
- Ignore vscode  (:pr:`6255`)
- Update test tolerance  (:pr:`6288`)
- Remove open_help method  (:pr:`6290`)
- Remove deprecated code in preparation for release  (:pr:`6291`)
- Deprecate recarray support  (:pr:`6310`)
- Reduce test size to prevent 32-bit crash  (:pr:`6311`)
- Remove chain dot  (:pr:`6312`)
- Catch and fix warnings  (:pr:`6313`)
- Use NumPy's linalg when available  (:pr:`6315`)
- Pin xdist  (:pr:`6392`)
- Unify pandas testing import  (:pr:`6394`)
- Clarify codecov  (:pr:`6406`)
- Port non-diagnostic changes  (:pr:`6412`)
- Fixes for future SciPY and pandas  (:pr:`6414`)
- Fixes for rc2  (:pr:`6419`)
- Switch to bionic  (:pr:`6423`)
- Improve test that randomly fails  (:pr:`6426`)
- Fix future issues  (:pr:`6440`)
- Disable cvxopt for windows  (:pr:`6445`)
- Reduce tolerance on basin hopping test  (:pr:`6448`)
- Remove unused import  (:pr:`6449`)

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

- Chad Fulton
- Brock Mendel
- Peter Quackenbush
- Kerby Shedden
- Kevin Sheppard

and the general maintainer and code reviewer

- Josef Perktold

Additionally, many users contributed by participation in github issues and
providing feedback.

Thanks to all of the contributors for the 0.10 release (based on git log):

- Atticus Yang
- Austin Adams
- Balazs Varga
- Brock Mendel
- Chad Fulton
- Christian Clauss
- Emil Mirzayev
- Graham Inggs
- Guglielmo Saggiorato
- Hassan Kibirige
- Ian Preston
- Jefferson Tweed
- Josef Perktold
- Keller Scholl
- Kerby Shedden
- Kevin Sheppard
- Lucas Roberts
- Mandy Gu
- Omer Ozen
- Padarn Wilson
- Peter Quackenbush
- Qingqing Mao
- Rebecca N. Palmer
- Ron Itzikovitch
- Samesh Lakhotia
- Sandu Ursu
- Tim Staley
- Varun Sriram
- Yasine Gangat
- comatrion
- luxiform
- partev
- vegcev
- 郭飞


These lists of names are automatically generated based on git log, and may not
be complete.

Merged Pull Requests
--------------------

The following Pull Requests were merged since the last release:

- :pr:`4421`: ENH: Implement cached_value, cached_data proof of concept
- :pr:`5235`: STY: use Appender pattern for docstrings
- :pr:`5283`: ENH: Use cython fused types to simplify statespace code
- :pr:`5610`: BUG: Lilliefors min nobs not set
- :pr:`5692`: MAINT: remove sandbox.formula, supplanted by patsy
- :pr:`5735`: ENH: Allow fixing parameters in state space models
- :pr:`5757`: MAINT: Remove docstring'd-out traceback for code that no longer raises
- :pr:`5768`: WIP/TST: enable/mark mangled/commented-out tests
- :pr:`5784`: MAINT: implement parts of #5220, deprecate ancient aliases
- :pr:`5797`: TST: test for anova_nistcertified
- :pr:`5799`: TST: Catch warnings produced during tests
- :pr:`5814`: CLN: parts of iolib
- :pr:`5819`: CLN: robust
- :pr:`5821`: CLN: test_constrained
- :pr:`5826`: ENH/REF: Markov switching in log space: Hamilton filter / Kim smoother
- :pr:`5827`: ENH: Add new version of ARIMA-type estimators (AR, ARIMA, SARIMAX)
- :pr:`5832`: DOC: remove orphaned docs files
- :pr:`5841`: MAINT: remove no-longer-needed HC_se lookups
- :pr:`5842`: CLN: E701 multiple statements on one line (colon)
- :pr:`5852`: ENH: Dimension reduction for covariance matrices
- :pr:`5856`: MAINT: remove ex_pairwise file dominated by try_tukey_hsd
- :pr:`5892`: BUG: fix pandas compat
- :pr:`5893`: BUG: exponential smoothing - damped trend gives incorrect param, predictions
- :pr:`5895`: DOC: improvements to BayesMixedGLM docs, argument checking
- :pr:`5897`: MAINT: Use pytest.raises to check error message
- :pr:`5898`: ENH: use class to define MixedLM variance components structure
- :pr:`5903`: BUG: Fix kwargs update bug in linear model fit_regularized
- :pr:`5910`: MAINT: Bump dependencies
- :pr:`5915`: ENH: state space: add methods to apply fitted parameters to new observations or new dataset
- :pr:`5917`: BUG: TVTP for Markov regression
- :pr:`5921`: BUG: Ensure exponential smoothers has continuous double data
- :pr:`5922`: MAINT: Fix pandas imports
- :pr:`5926`: ENH: Add STL decomposition for time series
- :pr:`5927`: MAINT: Remove Python 2.7 from Appveyor
- :pr:`5928`: ENH: Add array_like function to simplify input checking
- :pr:`5929`: DOC: array-like -> array_like
- :pr:`5930`: BUG: Limit lags in KPSS
- :pr:`5931`: MAINT: Relax tol on test that randomly fails
- :pr:`5933`: MAINT: Fix test that fails with positive probability
- :pr:`5935`: CLN: port parts of #5220
- :pr:`5937`: DOC: Change some more links to https
- :pr:`5938`: MAINT: Remove Python 2.7 from travis
- :pr:`5939`: DOC: Fix self-contradictory minimum dependency versions
- :pr:`5940`: MAINT: Fix linting failures
- :pr:`5946`: DOC: Fix formula for log-like in WLS
- :pr:`5947`: PERF: Cythonize innovations algo and filter
- :pr:`5948`: ENH: Normalize eigenvectors from coint_johansen
- :pr:`5949`: DOC: Fix typo
- :pr:`5950`: MAINT: Drop redundant travis configs
- :pr:`5951`: BUG: Fix mosaic plot with missing category
- :pr:`5952`: ENH: Improve RESET test stability
- :pr:`5953`: ENH: Add type checkers/converts for int, float and bool
- :pr:`5954`: MAINT: Mark MPL test as MPL
- :pr:`5956`: BUG: Fix multidimensional model cov_params when using pandas
- :pr:`5957`: DOC: Clarify xname length and purpose
- :pr:`5958`: MAINT: Deprecate periodogram
- :pr:`5960`: MAINT: Ensure seaborn is available for docbuild
- :pr:`5962`: CLN: cython cleanups
- :pr:`5963`: ENH: Functional SIR
- :pr:`5964`: ENH: Add start_params to RLM
- :pr:`5965`: MAINT: Remove PY3
- :pr:`5966`: ENH: Add JohansenResults class
- :pr:`5967`: BUG/ENH: Improve RLM in the case of perfect fit
- :pr:`5968`: BUG: Make mad function behave correctly when used on empty inputs
- :pr:`5969`: MAINT: Remove future and Python 2.7
- :pr:`5971`: BUG: Fix a future issue in ExpSmooth
- :pr:`5972`: MAINT: Remove string_types in favor of str
- :pr:`5976`: MAINT: Restore ResettableCache
- :pr:`5977`: MAINT: Cleanup legacy imports
- :pr:`5982`: CLN: follow-up to #5956
- :pr:`5983`: BUG: Fix return for RegressionResults
- :pr:`5984`: MAINT: Clarify breusch_pagan is for scalars
- :pr:`5986`: DOC: Add parameters for CountModel predict
- :pr:`5987`: MAINT: add W605 to lint codes
- :pr:`5988`: CLN: follow-up to #5928
- :pr:`5990`: MAINT/DOC: Add spell checking
- :pr:`5991`: MAINT: Remove comment no longer relevant
- :pr:`5992`: DOC: Fix many spelling errors
- :pr:`5994`: DOC: Small fixups after the spell check
- :pr:`5995`: ENH: Add R-squared and Adj. R_squared to summary_col
- :pr:`5996`: BUG: Limit lags in KPSS
- :pr:`5997`: ENH/BUG: Add check to AR instance to prevent bugs
- :pr:`5998`: BUG: Replace alpha=0.05 with alpha=alpha
- :pr:`5999`: ENH: Add summary to AR
- :pr:`6000`: DOC: Clarify that GARCH models are deprecated
- :pr:`6001`: MAINT: Refactor X13 testing
- :pr:`6002`: MAINT: Standardized on nlags for acf/pacf
- :pr:`6003`: BUG: Do not fit when fit=False
- :pr:`6004`: ENH/BUG: Allow ARMA predict to swallow typ
- :pr:`6005`: MAINT: Ignore warns on 32 bit linux
- :pr:`6006`: BUG/ENH: Check exog in ARMA and ARIMA predict
- :pr:`6007`: MAINT: Rename forecast years to forecast periods
- :pr:`6008`: ENH: Allow GC testing for specific lags
- :pr:`6009`: TST: Verify categorical is supported for MNLogit
- :pr:`6010`: TST: Improve test that is failing due to precision issues
- :pr:`6011`: MAINT/BUG/TST: Improve testing of seasonal decompose
- :pr:`6012`: BUG: Fix t-test and f-test for multidimensional params
- :pr:`6014`: ENH: Zivot Andrews test
- :pr:`6015`: CLN: Remove notes about incorrect test
- :pr:`6016`: TST: Add check for dtypes in Binomial
- :pr:`6017`: ENH: Set limit for number of endog variables when using formulas
- :pr:`6018`: ENH: Improve ARMA startparams
- :pr:`6019`: BUG: Fix ARMA cov_params
- :pr:`6020`: TST: Correct test to use trend not level
- :pr:`6022`: DOC: added content for two headings in VAR docs
- :pr:`6023`: TST: Verify missing exog raises in ARIMA
- :pr:`6026`: WIP: Added Oaxaca-Blinder Decomposition
- :pr:`6028`: ENH: Add rolling WLS and OLS
- :pr:`6030`: MAINT: Turn relative import into an absolute import
- :pr:`6031`: DOC: Fix regression doc strings
- :pr:`6036`: BLD/DOC: Add doc string check to doc build
- :pr:`6038`: DOC: Apply documentation standardizations
- :pr:`6039`: MAINT: Change types for future changes in NumPy
- :pr:`6041`: DOC: Fix spelling
- :pr:`6042`: DOC: Merge pull request #6041 from bashtage/doc-fixes
- :pr:`6044`: DOC: Fix notebook due to pandas index change
- :pr:`6045`: DOC/MAINT: Remove warning due to deprecated features
- :pr:`6046`: DOC: Remove DynamicVAR
- :pr:`6048`: DOC: Small doc site improvements
- :pr:`6050`: BUG: MLEModel now passes nobs to Representation
- :pr:`6052`: DOC: Small fix ups for modernized size
- :pr:`6053`: DOC: More small doc fixes
- :pr:`6054`: DOC: Small changes to doc building
- :pr:`6055`: DOC: Use the working branch of numpy doc
- :pr:`6056`: MAINT: Prepare for Rolling Least Squares
- :pr:`6057`: DOC: Fix spelling in notebooks
- :pr:`6058`: DOC: Fix missing spaces around colon
- :pr:`6059`: REF: move garch to archive/
- :pr:`6060`: DOC: Continue fixing docstring formatting
- :pr:`6062`: DOC: Fix web font size
- :pr:`6063`: DOC: Fix web font size
- :pr:`6064`: ENH/PERF: Only perform required predict iterations in state space models
- :pr:`6066`: MAINT: Fix small lint issue
- :pr:`6067`: DOC: Fix doc errors affecting build
- :pr:`6069`: MAINT: Stop testing on old, buggy SciPy
- :pr:`6070`: BUG: Fix ARMA so that it works with exog when trend=nc
- :pr:`6071`: ENH: state space: Improve low memory usability; allow in fit, loglike
- :pr:`6072`: BUG: state space: cov_params computation in fix_params context
- :pr:`6073`: TST: Add conserve memory tests.
- :pr:`6074`: ENH: Improve cov_params in append, extend, apply
- :pr:`6075`: DOC: Clean tsatools docs
- :pr:`6077`: DOC: Improve regression doc strings
- :pr:`6079`: ENH/DOC: Improve Ljung-Box
- :pr:`6080`: DOC: Improve docs in tools and ar_model
- :pr:`6081`: BUG: Fix error introduced in isestimable
- :pr:`6082`: DOC: Improve filter docstrings
- :pr:`6085`: DOC: Spelling and notebook link
- :pr:`6087`: ENH: Replacement for AR
- :pr:`6088`: MAINT: Small fixes in preparation for larger changes
- :pr:`6089`: DOC: Website fix
- :pr:`6090`: ENH/DOC: Add tools for programatically manipulating docstrings
- :pr:`6091`: MAINT/SEC: Remove unnecessary pickle use
- :pr:`6092`: MAINT: Ensure r download cache works
- :pr:`6093`: MAINT: Fix new cache name
- :pr:`6094`: TST: Fix wrong test
- :pr:`6096`: DOC: Seasonality in SARIMAX Notebook
- :pr:`6102`: ENH: Improve SARIMAX start_params if too few nobs
- :pr:`6104`: BUG: Fix score computation with fixed params
- :pr:`6105`: BUG: Correct thresholding in correlation tools
- :pr:`6106`: DOC: Changes summary_col's docstring to match variables
- :pr:`6107`: DOC: Update spelling in CONTRIBUTING.rst
- :pr:`6108`: DOC: Update link in CONTRIBUTING.rst
- :pr:`6110`: DOC: Update PR template Numpy guide link
- :pr:`6111`: ENH: Add exact diffuse initialization as an option for SARIMAX, UnobservedComponents
- :pr:`6113`: DOC: added interpretations to LogitResults.get_margeff
- :pr:`6116`: DOC: Improve docstrings
- :pr:`6117`: MAINT: Remove extra LICENSE.txt and setup.cfg
- :pr:`6118`: DOC: Clarify summary2 documentation
- :pr:`6119`: DOC: Switch doc theme
- :pr:`6120`: DOC: Add initial API doc
- :pr:`6122`: DOC: Small improvements to docs
- :pr:`6123`: DOC: Switch doc icon
- :pr:`6124`: ENH: Plot only unique censored points
- :pr:`6125`: DOC: Fix doc build failure
- :pr:`6126`: DOC: Update templates and add missing API functions
- :pr:`6130`: BUG: Incorrect TSA index if loc resolves to slice
- :pr:`6131`: ENH: Compute standardized forecast error in diffuse period if possible
- :pr:`6133`: BUG: start_params for VMA model with exog.
- :pr:`6134`: DOC: Add missing functions from the API
- :pr:`6136`: DOC: Restructure the documentation
- :pr:`6142`: DOC: Add a new logo
- :pr:`6143`: DOC: Fix validator so that it works
- :pr:`6144`: BUG: use self.data consistently
- :pr:`6145`: DOC: Add formula API
- :pr:`6152`: BUG: Fix accepting of eval environment for formula
- :pr:`6160`: DOC: fix sidebar TOC
- :pr:`6161`: BLD: Travis CI: The sudo: tag is deprecated in Travis
- :pr:`6162`: DOC/SEC: Warn that only trusted files should be unpickled
- :pr:`6163`: ENH: Improve the cvxopt not found error
- :pr:`6164`: MAINT: Be compatible with scipy 1.3
- :pr:`6165`: MAINT: Don't assume that 'python' is Python 3
- :pr:`6166`: DOC: Update pickle warning
- :pr:`6167`: DOC: Fix warning format
- :pr:`6179`: ENH: Adds state space version of linear exponential smoothing models
- :pr:`6181`: ENH: state space: add wrapped states and, where possible, named states
- :pr:`6198`: DOC: Clarify req for cvxopt
- :pr:`6204`: DOC: Spelling and Doc String Fixes
- :pr:`6205`: MAINT: Exclude pytest-xdist 1.30
- :pr:`6208`: ENH: Scale parameter handling in GEE
- :pr:`6214`: DOC: fix a typo
- :pr:`6215`: DOC: fix typos in install.rst
- :pr:`6216`: DOC: fix a typo
- :pr:`6217`: BUG: Fix summary table header for mixedlm
- :pr:`6222`: MAINT: Relax precision for ppc64el
- :pr:`6227`: ENH: Add missing keyword argument to plot_acf
- :pr:`6231`: BUG: allow dynamic factor starting parameters computation with NaNs values
- :pr:`6232`: BUG: division by zero in exponential smoothing if damping_slope=0
- :pr:`6233`: BUG: dynamic factor model use AR model for error start params if error_var=False
- :pr:`6235`: DOC: docstring fixes
- :pr:`6239`: BUG: SARIMAX index behavior with simple_differencing=True
- :pr:`6240`: BUG: parameter names in DynamicFactor for unstructured error covariance matrix
- :pr:`6241`: BUG: SARIMAX: basic validation for order, seasonal_order
- :pr:`6242`: BUG: Forecasts now ignore non-monotonic period index
- :pr:`6246`: TST: Add Python 3.8 environment
- :pr:`6250`: ENH: Update SARIMAX to use SARIMAXSpecification for more consistent input handling
- :pr:`6254`: ENH: State space: Add finer-grained memory conserve settings
- :pr:`6255`: MAINT: Ignore vscode
- :pr:`6257`: DOC: Fix spelling in notebooks
- :pr:`6258`: BUG: Hannan-Rissanen third stage is invalid if non-stationary/invertible
- :pr:`6260`: BUG: cloning of arima.ARIMA models.
- :pr:`6261`: BUG: state space: saving fixed params w/ extend, apply, append
- :pr:`6266`: ENH: and vlines option to plot_fit
- :pr:`6275`: MAINT/DOC: Clarify patsy 0.5.1 is required
- :pr:`6279`: DOC: Fix notebook
- :pr:`6280`: ENH: State space: Improve simulate, IRF, prediction
- :pr:`6281`: BUG: Pass arguments through in plot_leverage_resid2
- :pr:`6283`: MAINT/DOC: Close issues
- :pr:`6285`: BUG: Raise in GC test for VAR(0)
- :pr:`6286`: BUG: Correct VAR summary when model contains exog variables
- :pr:`6288`: MAINT: Update test tolerance
- :pr:`6289`: DOC: doc string changes
- :pr:`6290`: MAINT: Remove open_help method
- :pr:`6291`: MAINT: Remove deprecated code in preparation for release
- :pr:`6292`: BUG: Ensure BW is not 0
- :pr:`6294`: ENH: better argument checking for StratifiedTable
- :pr:`6297`: BUG: Fix conf interval with MI
- :pr:`6298`: DOC: Correct spells
- :pr:`6299`: DOC: Add example notebook for GEE score tests
- :pr:`6303`: DOC/MAINT: Add simple, documented script to get github info
- :pr:`6310`: MAINT: Deprecate recarray support
- :pr:`6311`: TST: Reduce test size to prevent 32-bit crash
- :pr:`6312`: MAINT: Remove chain dot
- :pr:`6313`: MAINT: Catch and fix warnings
- :pr:`6314`: BUG: Check dtype in KDEUnivariate
- :pr:`6315`: MAINT: Use NumPy's linalg when available
- :pr:`6316`: MAINT: Workaround NumPy ptp issue
- :pr:`6317`: DOC: Update test running instructions
- :pr:`6318`: BUG: Ensure inputs are finite in granger causality test
- :pr:`6319`: DOC: Restore test() autosummary
- :pr:`6320`: BUG: Restore multicomp
- :pr:`6321`: BUG: Fix trend due to recent changes
- :pr:`6322`: DOC: fix alpha description for GLMGam
- :pr:`6324`: ENH: Improve Ljung Box diagnostics
- :pr:`6327`: DOC: Move api docs
- :pr:`6332`: DEPR: state space: deprecate out-of-sample w/ unsupported index
- :pr:`6333`: BUG: state space: integer params can cause imaginary output
- :pr:`6334`: ENH: append, extend check that index matches model
- :pr:`6337`: BUG: fix k_exog, k_trend in arima.ARIMA; raise error when cloning a model with exog if no new exog given
- :pr:`6338`: DOC: Documentation for release v0.11
- :pr:`6340`: BUG: fix _get_index_loc with date strings
- :pr:`6342`: DOC: Update Release Note
- :pr:`6343`: DOC: Fix documentation errors
- :pr:`6344`: DOC: Fixes in preparation for release
- :pr:`6345`: DOC: Further doc fixes
- :pr:`6347`: DOC: Fix minor doc errors
- :pr:`6348`: DOC: git notes
- :pr:`6349`: DOC: Finalize release notes for 0.11
- :pr:`6350`: DOC: Add version dropdown
- :pr:`6353`: DOC: Finalize release note
- :pr:`6363`: DOC: Change generated path
- :pr:`6368`: Doc updates
- :pr:`6369`: DOC: Improve doc strings
- :pr:`6374`: BUG: Correct standardization in robust skewness
- :pr:`6375`: ENH: Supporting custom kernel in local linear kernel regression
- :pr:`6389`: BUG: Use correct exog names
- :pr:`6390`: DOC: Clarify demeaning in ljungbox
- :pr:`6391`: BUG: Fix axis in irq
- :pr:`6392`: MAINT: Pin xdist
- :pr:`6394`: MAINT: Unify pandas testing import
- :pr:`6398`: DOC: fix ridge regression formula in hpfilter
- :pr:`6401`: ENH: Distance dependence measures
- :pr:`6406`: MAINT: Clarify codecov
- :pr:`6407`: DOC: Fix link
- :pr:`6410`: ENH/CLN: Improve diagnostics
- :pr:`6412`: CLN/MAINT: Port non-diagnostic changes
- :pr:`6414`: CLN: Fixes for future SciPY and pandas
- :pr:`6415`: BUG: RecursiveLS should not allow `fix_params` method.
- :pr:`6416`: DOC: Update release note for v0.11.0rc2
- :pr:`6419`: MAINT: Fixes for rc2
- :pr:`6422`: BUG: Improve executable detection
- :pr:`6423`: MAINT: Switch to bionic
- :pr:`6424`: REF: More descriptive error message for recursive least squares parameter constraints.
- :pr:`6425`: BUG/ENH: Diffuse multivariate case w/ non-diagonal observation innovation covariance matrix
- :pr:`6426`: TST: Improve test that randomly fails
- :pr:`6440`: MAINT: Fix future issues
- :pr:`6442`: BUG: Fix bug in ridge for vector alpha
- :pr:`6445`: MAINT: Disable cvxopt for windows
- :pr:`6447`: DOC: Replace array with ndarray
- :pr:`6448`: MAINT: Reduce tolerance on basin hopping test
- :pr:`6449`: MAINT: Remove unused import
- :pr:`6450`: DOC: Final release note change for 0.11
