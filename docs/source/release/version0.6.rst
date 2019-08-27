:orphan:

=============
Release 0.6.1
=============

statsmodels 0.6.1 is a bugfix release. All users are encouraged to upgrade to 0.6.1.

See the :ref:`list of fixed issues <issues_list_06>` for specific backported fixes.

=============
Release 0.6.0
=============

statsmodels 0.6.0 is another large release. It is the result of the work of 37 authors over the last year and includes over 1500 commits. It contains many new features, improvements, and bug fixes detailed below.


See the :ref:`list of fixed issues <issues_list_06>` for specific closed issues.

The following major new features appear in this version.

Generalized Estimating Equations
--------------------------------

Generalized Estimating Equations (GEE) provide an approach to handling
dependent data in a regression analysis.  Dependent data arise
commonly in practice, such as in a longitudinal study where repeated
observations are collected on subjects. GEE can be viewed as an
extension of the generalized linear modeling (GLM) framework to the
dependent data setting.  The familiar GLM families such as the
Gaussian, Poisson, and logistic families can be used to accommodate
dependent variables with various distributions.

Here is an example of GEE Poisson regression in a data set with four
count-type repeated measures per subject, and three explanatory
covariates.

.. code-block:: python

   import numpy as np
   import statsmodels.api as sm
   import statsmodels.formula.api as smf

   data = sm.datasets.get_rdataset("epil", "MASS").data

   md = smf.gee("y ~ age + trt + base", "subject", data,
                cov_struct=sm.cov_struct.Independence(), 
                family=sm.families.Poisson())
   mdf = md.fit()
   print mdf.summary()


The dependence structure in a GEE is treated as a nuisance parameter
and is modeled in terms of a "working dependence structure".  The
statsmodels GEE implementation currently includes five working
dependence structures (independent, exchangeable, autoregressive,
nested, and a global odds ratio for working with categorical data).
Since the GEE estimates are not maximum likelihood estimates,
alternative approaches to some common inference procedures have been
developed.  The statsmodels GEE implementation currently provides
standard errors, Wald tests, score tests for arbitrary parameter
contrasts, and estimates and tests for marginal effects.  Several
forms of standard errors are provided, including robust standard
errors that are approximately correct even if the working dependence
structure is misspecified.

Seasonality Plots
-----------------

Adding functionality to look at seasonality in plots. Two new functions are :func:`sm.graphics.tsa.month_plot` and :func:`sm.graphics.tsa.quarter_plot`. Another function :func:`sm.graphics.tsa.seasonal_plot` is available for power users.

.. code-block:: python

    import statsmodels.api as sm
    import pandas as pd

    dta = sm.datasets.elnino.load_pandas().data
    dta['YEAR'] = dta.YEAR.astype(int).astype(str)
    dta = dta.set_index('YEAR').T.unstack()
    dates = map(lambda x : pd.datetools.parse('1 '+' '.join(x)),
                                           dta.index.values)

    dta.index = pd.DatetimeIndex(dates, freq='M')
    fig = sm.tsa.graphics.month_plot(dta)

.. currentmodule:: statsmodels.tsa

Seasonal Decomposition
----------------------

We added a naive seasonal decomposition tool in the same vein as R's ``decompose``. This function can be found as :func:`sm.tsa.seasonal_decompose <tsa.seasonal.seasonal_decompose>`.


.. plot::
   :include-source:

    import statsmodels.api as sm

    dta = sm.datasets.co2.load_pandas().data
    # deal with missing values. see issue
    dta.co2.interpolate(inplace=True)

    res = sm.tsa.seasonal_decompose(dta.co2)
    res.plot()


Addition of Linear Mixed Effects Models (MixedLM)

Linear Mixed Effects Models
---------------------------

Linear Mixed Effects models are used for regression analyses involving
dependent data.  Such data arise when working with longitudinal and
other study designs in which multiple observations are made on each
subject.  Two specific mixed effects models are "random intercepts
models", where all responses in a single group are additively shifted
by a value that is specific to the group, and "random slopes models",
where the values follow a mean trajectory that is linear in observed
covariates, with both the slopes and intercept being specific to the
group.  The statsmodels MixedLM implementation allows arbitrary random
effects design matrices to be specified for the groups, so these and
other types of random effects models can all be fit.

Here is an example of fitting a random intercepts model to data from a
longitudinal study:

.. code-block:: python

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    data = sm.datasets.get_rdataset('dietox', 'geepack', cache=True).data
    md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])
    mdf = md.fit()
    print mdf.summary()

The statsmodels LME framework currently supports post-estimation
inference via Wald tests and confidence intervals on the coefficients,
profile likelihood analysis, likelihood ratio testing, and AIC.  Some
limitations of the current implementation are that it does not support
structure more complex on the residual errors (they are always
homoscedastic), and it does not support crossed random effects.  We
hope to implement these features for the next release.

Wrapping X-12-ARIMA/X-13-ARIMA
------------------------------

It is now possible to call out to X-12-ARIMA or X-13ARIMA-SEATS from statsmodels. These libraries must be installed separately.

.. code-block:: python

    import statsmodels.api as sm

    dta = sm.datasets.co2.load_pandas().data
    dta.co2.interpolate(inplace=True)
    dta = dta.resample('M').last()

    res = sm.tsa.x13_arima_select_order(dta.co2)
    print(res.order, res.sorder)

    results = sm.tsa.x13_arima_analysis(dta.co2)

    fig = results.plot()
    fig.set_size_inches(12, 5)
    fig.tight_layout()


Other important new features
----------------------------

* The AR(I)MA models now have a :func:`plot_predict <arima_model.ARMAResults.plot_predict>` method to plot forecasts and confidence intervals.
* The Kalman filter Cython code underlying AR(I)MA estimation has been substantially optimized. You can expect speed-ups of one to two orders of magnitude.

* Added :func:`sm.tsa.arma_order_select_ic`. A convenience function to quickly get the information criteria for use in tentative order selection of ARMA processes.

* Plotting functions for timeseries is now imported under the ``sm.tsa.graphics`` namespace in addition to ``sm.graphics.tsa``.

* New `distributions.ExpandedNormal` class implements the Edgeworth expansion for weakly non-normal distributions.

* **New datasets**: Added new :ref:`datasets <datasets>` for examples. ``sm.datasets.co2`` is a univariate time-series dataset of weekly co2 readings. It exhibits a trend and seasonality and has missing values.

* Added robust skewness and kurtosis estimators in :func:`sm.stats.stattools.robust_skewness` and :func:`sm.stats.stattools.robust_kurtosis`, respectively.  An alternative robust measure of skewness has been added in :func:`sm.stats.stattools.medcouple`.

* New functions added to correlation tools: `corr_nearest_factor`
  finds the closest factor-structured correlation matrix to a given
  square matrix in the Frobenius norm; `corr_thresholded` efficiently
  constructs a hard-thresholded correlation matrix using sparse matrix
  operations.

* New `dot_plot` in graphics: A dotplot is a way to visualize a small dataset
  in a way that immediately conveys the identity of every point in the plot.
  Dotplots are commonly seen in meta-analyses, where they are known
  as "forest plots", but can be used in many other settings as well.
  Most tables that appear in research papers can be represented
  graphically as a dotplot.
* statsmodels has added custom warnings to ``statsmodels.tools.sm_exceptions``. By default all of these warnings will be raised whenever appropriate. Use ``warnings.simplefilter`` to turn them off, if desired.
* Allow control over the namespace used to evaluate formulas with patsy via the ``eval_env`` keyword argument. See the :ref:`patsy-namespaces` documentation for more information.


Major Bugs fixed
----------------

* NA-handling with formulas is now correctly handled. :issue:`805`, :issue:`1877`.
* Better error messages when an array with an object dtype is used. :issue:`2013`.
* ARIMA forecasts were hard-coded for order of integration with ``d = 1``. :issue:`1562`.

.. currentmodule:: statsmodels.tsa

Backwards incompatible changes and deprecations
-----------------------------------------------

* RegressionResults.norm_resid is now a readonly property, rather than a function.
* The function ``statsmodels.tsa.filters.arfilter`` has been removed. This did not compute a recursive AR filter but was instead a convolution filter. Two new functions have been added with clearer names :func:`sm.tsa.filters.recursive_filter <tsa.filters.filtertools.recursive_filter>` and :func:`sm.tsa.filters.convolution_filter <tsa.filters.filtertools.convolution_filter>`.

Development summary and credits
-------------------------------

The previous version (0.5.0) was released August 14, 2014. Since then we have closed a total of 528 issues, 276 pull requests, and 252 regular issues. Refer to the :ref:`detailed list<issues_list_06>` for more information.

This release is a result of the work of the following 37 authors who contributed a total of 1531 commits. If for any reason we have failed to list your name in the below, please contact us:

A blurb about the number of changes and the contributors list.

* Alex Griffing <argriffi-at-ncsu.edu>
* Alex Parij <paris.alex-at-gmail.com>
* Ana Martinez Pardo <anamartinezpardo-at-gmail.com>
* Andrew Clegg <andrewclegg-at-users.noreply.github.com>
* Ben Duffield <bduffield-at-palantir.com>
* Chad Fulton <chad-at-chadfulton.com>
* Chris Kerr <cjk34-at-cam.ac.uk>
* Eric Chiang <eric.chiang.m-at-gmail.com>
* Evgeni Burovski <evgeni-at-burovski.me>
* gliptak <gliptak-at-gmail.com>
* Hans-Martin von Gaudecker <hmgaudecker-at-uni-bonn.de>
* Jan Schulz <jasc-at-gmx.net>
* jfoo <jcjf1983-at-gmail.com>
* Joe Hand <joe.a.hand-at-gmail.com>
* Josef Perktold <josef.pktd-at-gmail.com>
* jsphon <jonathanhon-at-hotmail.com>
* Justin Grana <jg3705a-at-student.american.edu>
* Kerby Shedden <kshedden-at-umich.edu>
* Kevin Sheppard <kevin.sheppard-at-economics.ox.ac.uk>
* Kyle Beauchamp <kyleabeauchamp-at-gmail.com>
* Lars Buitinck <l.buitinck-at-esciencecenter.nl>
* Max Linke <max_linke-at-gmx.de>
* Miroslav Batchkarov <mbatchkarov-at-gmail.com>
* m <mngu2382-at-gmail.com>
* Padarn Wilson <padarn-at-gmail.com>
* Paul Hobson <pmhobson-at-gmail.com>
* Pietro Battiston <me-at-pietrobattiston.it>
* Radim Řehůřek <radimrehurek-at-seznam.cz>
* Ralf Gommers <ralf.gommers-at-googlemail.com>
* Richard T. Guy <richardtguy84-at-gmail.com>
* Roy Hyunjin Han <rhh-at-crosscompute.com>
* Skipper Seabold <jsseabold-at-gmail.com>
* Tom Augspurger <thomas-augspurger-at-uiowa.edu>
* Trent Hauck <trent.hauck-at-gmail.com>
* Valentin Haenel <valentin.haenel-at-gmx.de>
* Vincent Arel-Bundock <varel-at-umich.edu>
* Yaroslav Halchenko <debian-at-onerussian.com>

.. note::

   Obtained by running ``git log v0.5.0..HEAD --format='* %aN <%aE>' | sed 's/@/\-at\-/' | sed 's/<>//' | sort -u``.

.. _issues_list_06:

Issues closed in the 0.6.0 development cycle
============================================

Issues closed in 0.6.0
----------------------

GitHub stats for 2013/08/14 - 2014/10/15 (tag: v0.5.0)

We closed a total of 528 issues, 276 pull requests and 252 regular issues;
this is the full list (generated with the script :file:`tools/github_stats.py`):

This list is automatically generated and may be incomplete.

Pull Requests (276):

* :pr:`2044`: ENH: Allow unit interval for binary models. Closes #2040.
* :pr:`1426`: ENH: Import arima_process stuff into tsa.api
* :pr:`2042`: Fix two minor typos in contrast.py
* :pr:`2034`: ENH: Handle missing for extra data with formulas
* :pr:`2035`: MAINT: Remove deprecated code for 0.6
* :pr:`1325`: ENH: add the Edgeworth expansion based on the normal distribution
* :pr:`2032`: DOC: What it is what it is.
* :pr:`2031`: ENH: Expose patsy eval_env to users.
* :pr:`2028`: ENH: Fix numerical issues in links and families.
* :pr:`2029`: DOC: Fix versions to match other docs.
* :pr:`1647`: ENH: Warn on non-convergence.
* :pr:`2014`: BUG: Fix forecasting for ARIMA with d == 2
* :pr:`2013`: ENH: Better error message on object dtype
* :pr:`2012`: BUG: 2d 1 columns -> 1d. Closes #322.
* :pr:`2009`: DOC: Update after refactor. Use code block.
* :pr:`2008`: ENH: Add wrapper for MixedLM
* :pr:`1954`: ENH: PHReg formula improvements
* :pr:`2007`: BLD: Fix build issues
* :pr:`2006`: BLD: Do not generate cython on clean. Closes #1852.
* :pr:`2000`: BLD: Let pip/setuptools handle dependencies that are not installed at all.
* :pr:`1999`: Gee offset exposure 1994 rebased
* :pr:`1998`: BUG/ENH Lasso emptymodel rebased
* :pr:`1989`: BUG/ENH: WLS generic robust cov_type did not use whitened,
* :pr:`1587`: ENH: Wrap X12/X13-ARIMA AUTOMDL. Closes #442.
* :pr:`1563`: ENH: Add plot_predict method to ARIMA models.
* :pr:`1995`: BUG: Fix issue #1993
* :pr:`1981`: ENH: Add api for covstruct. Clear __init__. Closes #1917.
* :pr:`1996`: DEV: Ignore .venv file.
* :pr:`1982`: REF: Rename jac -> score_obs. Closes #1785.
* :pr:`1987`: BUG tsa pacf, base bootstrap
* :pr:`1986`: Bug multicomp 1927 rebased
* :pr:`1984`: Docs add gee.rst
* :pr:`1985`: Bug uncentered latex table 1929 rebased
* :pr:`1983`: BUG: Fix compat asunicode
* :pr:`1574`: DOC: Fix math.
* :pr:`1980`: DOC: Documentation fixes
* :pr:`1974`: REF/Doc beanplot change default color, add notebook
* :pr:`1978`: ENH: Check input to binary models
* :pr:`1979`: BUG: Typo
* :pr:`1976`: ENH: Add _repr_html_ to SimpleTable
* :pr:`1977`: BUG: Fix import refactor victim.
* :pr:`1975`: BUG: Yule walker cast to float
* :pr:`1973`: REF: Move and expose webuse
* :pr:`1972`: TST: Add testing against NumPy 1.9 and matplotlib 1.4
* :pr:`1939`: ENH: Binstar build files
* :pr:`1952`: REF/DOC: Misc
* :pr:`1940`: REF: refactor and speedup of mixed LME
* :pr:`1937`: ENH: Quick access to online documentation
* :pr:`1942`: DOC: Rename Change README type to rst
* :pr:`1938`: ENH: Enable Python 3.4 testing
* :pr:`1924`: Bug gee cov type 1906 rebased
* :pr:`1870`: robust covariance, cov_type in fit
* :pr:`1859`: BUG: Do not use negative indexing with k_ar == 0. Closes #1858.
* :pr:`1914`: BUG: LikelihoodModelResults.pvalues use df_resid_inference
* :pr:`1899`: TST: fix assert_equal for pandas index
* :pr:`1895`: Bug multicomp pandas
* :pr:`1894`: BUG fix more ix indexing cases for pandas compat
* :pr:`1889`: BUG: fix ytick positions closes #1561
* :pr:`1887`: Bug pandas compat asserts
* :pr:`1888`: TST test_corrpsd Test_Factor: add noise to data
* :pr:`1886`: BUG pandas 0.15 compatibility in grouputils labels
* :pr:`1885`: TST: corr_nearest_factor, more informative tests
* :pr:`1884`: Fix: Add compat code for pd.Categorical in pandas>=0.15
* :pr:`1883`: BUG: add _ctor_param to TransfGen distributions
* :pr:`1872`: TST: fix _infer_freq for pandas .14+ compat
* :pr:`1867`: Ref covtype fit
* :pr:`1865`: Disable tst distribution 1864
* :pr:`1856`: _spg_optim returns history of objective function values
* :pr:`1854`: BLD: Do not hard-code path for building notebooks. Closes #1249
* :pr:`1851`: MAINT: Cor nearest factor tests
* :pr:`1847`: Newton regularize
* :pr:`1623`: BUG Negbin fit regularized
* :pr:`1797`: BUG/ENH: fix and improve constant detection
* :pr:`1770`: TST: anova with `-1` noconstant, add tests
* :pr:`1837`: Allow group variable to be passed as variable name when using formula
* :pr:`1839`: BUG: GEE score
* :pr:`1830`: BUG/ENH Use t
* :pr:`1832`: TST error with scipy 0.14 location distribution class
* :pr:`1827`: fit_regularized for linear models   rebase 1674
* :pr:`1825`: Phreg 1312 rebased
* :pr:`1826`: Lme api docs
* :pr:`1824`: Lme profile 1695 rebased
* :pr:`1823`: Gee cat subclass 1694 rebase
* :pr:`1781`: ENH: Glm add score_obs
* :pr:`1821`: Glm maint #1734 rebased
* :pr:`1820`: BUG: revert change to conf_int in PR #1819
* :pr:`1819`: Docwork
* :pr:`1772`: REF: cov_params allow case of only cov_params_default is defined
* :pr:`1771`: REF numpy >1.9 compatibility, indexing into empty slice closes #1754
* :pr:`1769`: Fix ttest 1d
* :pr:`1766`: TST: TestProbitCG increase bound for fcalls closes #1690
* :pr:`1709`: BLD: Made build extensions more flexible
* :pr:`1714`: WIP: fit_constrained
* :pr:`1706`: REF: Use fixed params in test. Closes #910.
* :pr:`1701`: BUG: Fix faulty logic. Do not raise when missing='raise' and no missing data.
* :pr:`1699`: TST/ENH StandardizeTransform, reparameterize TestProbitCG
* :pr:`1697`: Fix for statsmodels/statsmodels#1689
* :pr:`1692`: OSL Example: redundant cell in example removed
* :pr:`1688`: Kshedden mixed rebased of #1398
* :pr:`1629`: Pull request to fix bandwidth bug in issue 597
* :pr:`1666`: Include pyx in sdist but do not install
* :pr:`1683`: TST: GLM shorten random seed closes #1682
* :pr:`1681`: Dotplot kshedden rebased of 1294
* :pr:`1679`: BUG: Fix problems with predict handling offset and exposure
* :pr:`1677`: Update docstring of RegressionModel.predict()
* :pr:`1635`: Allow offset and exposure to be used together with log link; raise except...
* :pr:`1676`: Tests for SVAR
* :pr:`1671`: ENH: avoid hard-listed bandwidths -- use present dictionary (+typos fixed)
* :pr:`1643`: Allow matrix structure in covariance matrices to be exploited
* :pr:`1657`: BUG: Fix refactor victim.
* :pr:`1630`: DOC: typo, "intercept"
* :pr:`1619`: MAINT: Dataset docs cleanup and automatic build of docs
* :pr:`1612`: BUG/ENH Fix negbin exposure #1611
* :pr:`1610`: BUG/ENH fix llnull, extra kwds to recreate model
* :pr:`1582`: BUG: wls_prediction_std fix weight handling, see 987
* :pr:`1613`: BUG: Fix proportions allpairs #1493
* :pr:`1607`: TST: adjust precision, CI Debian, Ubuntu testing
* :pr:`1603`: ENH: Allow start_params in GLM
* :pr:`1600`: CLN: Regression plots fixes
* :pr:`1592`: DOC: Additions and fixes
* :pr:`1520`: CLN: Refactored so that there is no longer a need for 2to3
* :pr:`1585`: Cor nearest 1384 rebased
* :pr:`1553`: Gee maint 1528 rebased
* :pr:`1583`: BUG: For ARMA(0,0) ensure 1d bse and fix summary.
* :pr:`1580`: DOC: Fix links. [skip ci]
* :pr:`1572`: DOC: Fix link title [skip ci]
* :pr:`1566`: BLD: Fix copy paste path error for >= 3.3 Windows builds
* :pr:`1524`: ENH: Optimize Cython code. Use scipy blas function pointers.
* :pr:`1560`: ENH: Allow ARMA(0,0) in order selection
* :pr:`1559`: MAINT: Recover lost commits from vbench PR
* :pr:`1554`: Silenced test output introduced in medcouple
* :pr:`1234`: ENH: Robust skewness, kurtosis and medcouple measures
* :pr:`1484`: ENH: Add naive seasonal decomposition function
* :pr:`1551`: COMPAT: Fix failing test on Python 2.6
* :pr:`1472`: ENH: using human-readable group names instead of integer ids in MultiComparison
* :pr:`1437`: ENH: accept non-int definitions of cluster groups
* :pr:`1550`: Fix test gmm poisson
* :pr:`1549`: TST: Fix locally failing tests.
* :pr:`1121`: WIP: Refactor optimization code.
* :pr:`1547`: COMPAT: Correct bit_length for 2.6
* :pr:`1545`: MAINT: Fix missed usage of deprecated tools.rank
* :pr:`1196`: REF: ensure O(N log N) when using fft for acf
* :pr:`1154`: DOC: Add links for build machines.
* :pr:`1546`: DOC: Fix link to wrong notebook
* :pr:`1383`: MAINT: Deprecate rank in favor of np.linalg.matrix_rank
* :pr:`1432`: COMPAT: Add NumpyVersion from scipy
* :pr:`1438`: ENH: Option to avoid "center" environment.
* :pr:`1544`: BUG: Travis miniconda
* :pr:`1510`: CLN: Improve warnings to avoid generic warnings messages
* :pr:`1543`: TST: Suppress RuntimeWarning for L-BFGS-B
* :pr:`1507`: CLN: Silence test output
* :pr:`1540`: BUG: Correct derivative for exponential transform.
* :pr:`1536`: BUG: Restores coveralls for a single build
* :pr:`1535`: BUG: Fixes for 2.6 test failures, replacing astype(str) with apply(str)
* :pr:`1523`: Travis miniconda
* :pr:`1533`: DOC: Fix link to code on github
* :pr:`1531`: DOC: Fix stale links with linkcheck
* :pr:`1530`: DOC: Fix link
* :pr:`1527`: DOCS: Update docs add FAQ page
* :pr:`1525`: DOC: Update with Python 3.4 build notes
* :pr:`1518`: DOC: Ask for release notes and example.
* :pr:`1516`: DOC: Update examples contributing docs for current practice.
* :pr:`1517`: DOC: Be clear about data attribute of Datasets
* :pr:`1515`: DOC: Fix broken link
* :pr:`1514`: DOC: Fix formula import convention.
* :pr:`1506`: BUG: Format and decode errors in Python 2.6
* :pr:`1505`: TST: Test co2 load_data for Python 3.
* :pr:`1504`: BLD: New R versions require NAMESPACE file. Closes #1497.
* :pr:`1483`: ENH: Some utility functions for working with dates
* :pr:`1482`: REF: Prefer filters.api to __init__
* :pr:`1481`: ENH: Add weekly co2 dataset
* :pr:`1474`: DOC: Add plots for standard filter methods.
* :pr:`1471`: DOC: Fix import
* :pr:`1470`: DOC/BLD: Log code exceptions from nbgenerate
* :pr:`1469`: DOC: Fix bad links
* :pr:`1468`: MAINT: CSS fixes
* :pr:`1463`: DOC: Remove defunct argument. Change default kw. Closes #1462.
* :pr:`1452`: STY: import pandas as pd
* :pr:`1458`: BUG/BLD: exclude sandbox in relative path, not absolute
* :pr:`1447`: DOC: Only build and upload docs if we need to.
* :pr:`1445`: DOCS: Example landing page
* :pr:`1436`: DOC: Fix auto doc builds.
* :pr:`1431`: DOC: Add default for getenv. Fix paths. Add print_info
* :pr:`1429`: MAINT: Use ip_directive shipped with IPython
* :pr:`1427`: TST: Make tests fit quietly
* :pr:`1424`: ENH: Consistent results for transform_slices
* :pr:`1421`: ENH: Add grouping utilities code
* :pr:`1419`: Gee 1314 rebased
* :pr:`1414`: TST temporarily rename tests probplot other to skip them
* :pr:`1403`: Bug norm expan shapes
* :pr:`1417`: REF: Let subclasses keep kwds attached to data.
* :pr:`1416`: ENH: Make handle_data overwritable by subclasses.
* :pr:`1410`: ENH: Handle missing is none
* :pr:`1402`: REF: Expose missing data handling as classmethod
* :pr:`1387`: MAINT: Fix failing tests
* :pr:`1406`: MAINT: Tools improvements
* :pr:`1404`: Tst fix genmod link tests
* :pr:`1396`: REF: Multipletests reduce memory usage
* :pr:`1380`: DOC :Update vector_ar.rst
* :pr:`1381`: BLD: Do not check dependencies on egg_info for pip. Closes #1267.
* :pr:`1302`: BUG: Fix typo.
* :pr:`1375`: STY: Remove unused imports and comment out unused libraries in setup.py
* :pr:`1143`: DOC: Update backport notes for new workflow.
* :pr:`1374`: ENH: Import tsaplots into tsa namespace. Closes #1359.
* :pr:`1369`: STY: Pep-8 cleanup
* :pr:`1370`: ENH: Support ARMA(0,0) models.
* :pr:`1368`: STY: Pep 8 cleanup
* :pr:`1367`: ENH: Make sure mle returns attach to results.
* :pr:`1365`: STY: Import and pep 8 cleanup
* :pr:`1364`: ENH: Get rid of hard-coded lbfgs. Closes #988.
* :pr:`1363`: BUG: Fix typo.
* :pr:`1361`: ENH: Attach mlefit to results not model.
* :pr:`1360`: ENH: Import adfuller into tsa namespace
* :pr:`1346`: STY: PEP-8 Cleanup
* :pr:`1344`: BUG: Use missing keyword given to ARMA.
* :pr:`1340`: ENH: Protect against ARMA convergence failures.
* :pr:`1334`: ENH: ARMA order select convenience function
* :pr:`1339`: Fix typos
* :pr:`1336`: REF: Get rid of plain assert.
* :pr:`1333`: STY: __all__ should be after imports.
* :pr:`1332`: ENH: Add Bunch object to tools.
* :pr:`1331`: ENH: Always use unicode.
* :pr:`1329`: BUG: Decode metadata to utf-8. Closes #1326.
* :pr:`1330`: DOC: Fix typo. Closes #1327.
* :pr:`1185`: Added support for pandas when pandas was installed directly from git trunk
* :pr:`1315`: MAINT: Change back to path for build box
* :pr:`1305`: TST: Update hard-coded path.
* :pr:`1290`: ENH: Add seasonal plotting.
* :pr:`1296`: BUG/TST: Fix ARMA forecast when start == len(endog). Closes #1295
* :pr:`1292`: DOC: cleanup examples folder and webpage
* :pr:`1286`: Make sure PeriodIndex passes through tsa. Closes #1285.
* :pr:`1271`: Silverman enhancement - Issue #1243
* :pr:`1264`: Doc work GEE, GMM, sphinx warnings
* :pr:`1179`: REF/TST: `ProbPlot` now uses `resettable_cache` and added some kwargs to plotting fxns
* :pr:`1225`: Sandwich mle
* :pr:`1258`: Gmm new rebased
* :pr:`1255`: ENH add GEE to genmod
* :pr:`1254`: REF: Results.predict convert to array and adjust shape
* :pr:`1192`: TST: enable tests for llf after change to WLS.loglike see #1170
* :pr:`1253`: Wls llf fix
* :pr:`1233`: sandbox kernels bugs uniform kernel and confint
* :pr:`1240`: Kde weights 1103 823
* :pr:`1228`: Add default value tags to adfuller() docs
* :pr:`1198`: fix typo
* :pr:`1230`: BUG: numerical precision in resid_pearson with perfect fit #1229
* :pr:`1214`: Compare lr test rebased
* :pr:`1200`: BLD: do not install \*.pyx \*.c  MANIFEST.in
* :pr:`1202`: MAINT: Sort backports to make applying easier.
* :pr:`1157`: Tst precision master
* :pr:`1161`: add a fitting interface for simultaneous log likelihood and score, for lbfgs, tested with MNLogit
* :pr:`1160`: DOC: update scipy version from 0.7 to 0.9.0
* :pr:`1147`: ENH: add lbfgs for fitting
* :pr:`1156`: ENH: Raise on 0,0 order models in AR(I)MA. Closes #1123
* :pr:`1149`: BUG: Fix small data issues for ARIMA.
* :pr:`1092`: Fixed duplicate svd in RegressionModel
* :pr:`1139`: TST: Silence tests
* :pr:`1135`: Misc style
* :pr:`1088`: ENH: add predict_prob to poisson
* :pr:`1125`: REF/BUG: Some GLM cleanup. Used trimmed results in NegativeBinomial variance.
* :pr:`1124`: BUG: Fix ARIMA prediction when fit without a trend.
* :pr:`1118`: DOC: Update gettingstarted.rst
* :pr:`1117`: Update ex_arma2.py
* :pr:`1107`: REF: Deprecate stand_mad. Add center keyword to mad. Closes #658.
* :pr:`1089`: ENH: exp(poisson.logpmf()) for poisson better behaved.
* :pr:`1077`: BUG: Allow 1d exog in ARMAX forecasting.
* :pr:`1075`: BLD: Fix build issue on some versions of easy_install.
* :pr:`1071`: Update setup.py to fix broken install on OSX
* :pr:`1052`: DOC: Updating contributing docs
* :pr:`1136`: RLS: Add IPython tools for easier backporting of issues.
* :pr:`1091`: DOC: minor git typo
* :pr:`1082`: coveralls support
* :pr:`1072`: notebook examples title cell
* :pr:`1056`: Example: reg diagnostics
* :pr:`1057`: COMPAT: Fix py3 caching for get_rdatasets.
* :pr:`1045`: DOC/BLD: Update from nbconvert to IPython 1.0.
* :pr:`1026`: DOC/BLD: Add LD_LIBRARY_PATH to env for docs build.

Issues (252):

* :issue:`2040`: enh: fractional Logit, Probit
* :issue:`1220`: missing in extra data (example sandwiches, robust covariances)
* :issue:`1877`: error with GEE on missing data.
* :issue:`805`: nan with categorical in formula
* :issue:`2036`: test in links require exact class so Logit cannot work in place of logit
* :issue:`2010`: Go over deprecations again for 0.6.
* :issue:`1303`: patsy library not automatically installed
* :issue:`2024`: genmod Links numerical improvements
* :issue:`2025`: GEE requires exact import for cov_struct
* :issue:`2017`: Matplotlib warning about too many figures
* :issue:`724`: check warnings
* :issue:`1562`: ARIMA forecasts are hard-coded for d=1
* :issue:`880`: DataFrame with bool type not cast correctly.
* :issue:`1992`: MixedLM style
* :issue:`322`: acf / pacf do not work on pandas objects
* :issue:`1317`: AssertionError: attr is not equal [dtype]: dtype('object') != dtype('datetime64[ns]')
* :issue:`1875`: dtype bug object arrays (raises in clustered standard errors code)
* :issue:`1842`: dtype object, glm.fit() gives AttributeError: sqrt
* :issue:`1300`: Doc errors, missing
* :issue:`1164`: RLM cov_params, t_test, f_test do not use bcov_scaled
* :issue:`1019`: 0.6.0 Roadmap
* :issue:`554`: Prediction Standard Errors
* :issue:`333`: ENH tools: squeeze in R export file
* :issue:`1990`: MixedLM does not have a wrapper
* :issue:`1897`: Consider depending on setuptools in setup.py
* :issue:`2003`: pip install now fails silently
* :issue:`1852`: do not cythonize when cleaning up
* :issue:`1991`: GEE formula interface does not take offset/exposure
* :issue:`442`: Wrap x-12 arima
* :issue:`1993`: MixedLM bug
* :issue:`1917`: API: GEE access to genmod.covariance_structure through api
* :issue:`1785`: REF: rename jac -> score_obs
* :issue:`1969`: pacf has incorrect standard errors for lag 0
* :issue:`1434`: A small bug in GenericLikelihoodModelResults.bootstrap()
* :issue:`1408`: BUG test failure with tsa_plots
* :issue:`1337`: DOC: HCCM are now available for WLS
* :issue:`546`: influence and outlier documentation
* :issue:`1532`: DOC: Related page is out of date
* :issue:`1386`: Add minimum matplotlib to docs
* :issue:`1068`: DOC: keeping documentation of old versions on sourceforge
* :issue:`329`: link to examples and datasets from module pages
* :issue:`1804`: PDF documentation for statsmodels
* :issue:`202`: Extend robust standard errors for WLS/GLS
* :issue:`1519`: Link to user-contributed examples in docs
* :issue:`1053`: inconvenient: logit when endog is (1,2) instead of (0,1)
* :issue:`1555`: SimpleTable: add repr html for ipython notebook
* :issue:`1366`: Change default start_params to .1 in ARMA
* :issue:`1869`: yule_walker (from `statsmodels.regression`) raises exception when given an integer array
* :issue:`1651`: statsmodels.tsa.ar_model.ARResults.predict
* :issue:`1738`: GLM robust sandwich covariance matrices
* :issue:`1779`: Some directories under statsmodels dont have __init_.py
* :issue:`1242`: No support for (0, 1, 0) ARIMA Models
* :issue:`1571`: expose webuse, use cache
* :issue:`1860`: ENH/BUG/DOC: Bean plot should allow for separate widths of bean and violins.
* :issue:`1831`: TestRegressionNM.test_ci_beta2 i386 AssertionError
* :issue:`1079`: bugfix release 0.5.1
* :issue:`1338`: Raise Warning for HCCM use in WLS/GLS
* :issue:`1430`: scipy min version / issue
* :issue:`276`: memoize, last argument wins, how to attach sandwich to Results?
* :issue:`1943`: REF/ENH: LikelihoodModel.fit optimization, make hessian optional
* :issue:`1957`: BUG: Re-create OLS model using _init_keys
* :issue:`1905`: Docs: online docs are missing GEE
* :issue:`1898`: add python 3.4 to continuous integration testing
* :issue:`1684`: BUG: GLM NegativeBinomial: llf ignores offset and exposure
* :issue:`1256`: REF: GEE handling of default covariance matrices
* :issue:`1760`: Changing covariance_type on results
* :issue:`1906`: BUG: GEE default covariance is not used
* :issue:`1931`: BUG: GEE subclasses NominalGEE do not work with pandas exog
* :issue:`1904`: GEE Results does not have a Wrapper
* :issue:`1918`: GEE: required attributes missing, df_resid
* :issue:`1919`: BUG GEE.predict uses link instead of link.inverse
* :issue:`1858`: BUG: arimax forecast should special case k_ar == 0
* :issue:`1903`: BUG: pvalues for cluster robust, with use_t do not use df_resid_inference
* :issue:`1243`: kde silverman bandwidth for non-gaussian kernels
* :issue:`1866`: Pip dependencies
* :issue:`1850`: TST test_corr_nearest_factor fails on Ubuntu
* :issue:`292`: python 3 examples
* :issue:`1868`: ImportError: No module named compat  [ from statsmodels.compat import lmap ]
* :issue:`1890`: BUG tukeyhsd nan in group labels
* :issue:`1891`: TST test_gmm outdated pandas, compat
* :issue:`1561`: BUG plot for tukeyhsd, MultipleComparison
* :issue:`1864`: test failure sandbox distribution transformation with scipy 0.14.0
* :issue:`576`: Add contributing guidelines
* :issue:`1873`: GenericLikelihoodModel is not picklable
* :issue:`1822`: TST failure on Ubuntu pandas 0.14.0 , problems with frequency
* :issue:`1249`: Source directory problem for notebook examples
* :issue:`1855`: anova_lm throws error on models created from api.ols but not formula.api.ols
* :issue:`1853`: a large number of hardcoded paths
* :issue:`1792`: R² adjusted strange after including interaction term
* :issue:`1794`: REF: has_constant, k_constant, include implicit constant detection in base
* :issue:`1454`: NegativeBinomial missing fit_regularized method
* :issue:`1615`: REF DRYing fit methods
* :issue:`1453`: Discrete NegativeBinomialModel regularized_fit ValueError: matrices are not aligned
* :issue:`1836`: BUG Got an TypeError trying to import statsmodels.api
* :issue:`1829`: BUG: GLM summary show "t"  use_t=True for summary
* :issue:`1828`: BUG summary2 does not propagate/use use_t
* :issue:`1812`: BUG/ REF conf_int and use_t
* :issue:`1835`: Problems with installation using easy_install
* :issue:`1801`: BUG 'f_gen' missing in scipy 0.14.0
* :issue:`1803`: Error revealed by numpy 1.9.0r1
* :issue:`1834`: stackloss
* :issue:`1728`: GLM.fit maxiter=0  incorrect
* :issue:`1795`: singular design with offset ?
* :issue:`1730`: ENH/Bug cov_params, generalize, avoid ValueError
* :issue:`1754`: BUG/REF: assignment to slices in numpy >= 1.9 (emplike)
* :issue:`1409`: GEE test errors on Debian Wheezy
* :issue:`1521`: ubuntu failures: tsa_plot and grouputils
* :issue:`1415`: test failure test_arima.test_small_data
* :issue:`1213`: df_diff in anova_lm
* :issue:`1323`: Contrast Results after t_test summary broken for 1 parameter
* :issue:`109`: TestProbitCG failure on Ubuntu
* :issue:`1690`: TestProbitCG: 8 failing tests (Python 3.4 / Ubuntu 12.04)
* :issue:`1763`: Johansen method does not give correct index values
* :issue:`1761`: doc build failures: ipython version ? ipython directive
* :issue:`1762`: Unable to build
* :issue:`1745`: UnicodeDecodeError raised by get_rdataset("Guerry", "HistData")
* :issue:`611`: test failure foreign with pandas 0.7.3
* :issue:`1700`: faulty logic in missing handling
* :issue:`1648`: ProbitCG failures
* :issue:`1689`: test_arima.test_small_data: SVD fails to converge (Python 3.4 / Ubuntu 12.04)
* :issue:`597`: BUG: nonparametric: kernel, efficient=True changes bw even if given
* :issue:`1606`: BUILD from sdist broken if cython available
* :issue:`1246`: test failure test_anova.TestAnova2.test_results
* :issue:`50`: t_test, f_test, model.py for normal instead of t-distribution
* :issue:`1655`: newey-west different than R?
* :issue:`1682`: TST test failure on Ubuntu, random.seed
* :issue:`1614`: docstring for regression.linear_model.RegressionModel.predict() does not match implementation
* :issue:`1318`: GEE and GLM scale parameter
* :issue:`519`: L1 fit_regularized cleanup, comments
* :issue:`651`: add structure to example page
* :issue:`1067`: Kalman Filter convergence. How close is close enough?
* :issue:`1281`: Newton convergence failure prints warnings instead of warning
* :issue:`1628`: Unable to install statsmodels in the same requirements file as numpy, pandas, etc.
* :issue:`617`: Problem in installing statsmodels in Fedora 17 64-bit
* :issue:`935`: ll_null in likelihoodmodels discrete
* :issue:`704`: datasets.sunspot: wrong link in description
* :issue:`1222`: NegativeBinomial ignores exposure
* :issue:`1611`: BUG NegativeBinomial ignores exposure and offset
* :issue:`1608`: BUG: NegativeBinomial, llnul is always default 'nb2'
* :issue:`1221`: llnull with exposure ?
* :issue:`1493`: statsmodels.stats.proportion.proportions_chisquare_allpairs has hardcoded value
* :issue:`1260`: GEE test failure on Debian
* :issue:`1261`: test failure on Debian
* :issue:`443`: GLM.fit does not allow start_params
* :issue:`1602`: Fitting GLM with a pre-assigned starting parameter
* :issue:`1601`: Fitting GLM with a pre-assigned starting parameter
* :issue:`890`: regression_plots problems (pylint) and missing test coverage
* :issue:`1598`: Is "old" string formatting Python 3 compatible?
* :issue:`1589`: AR vs ARMA order specification
* :issue:`1134`: Mark knownfails
* :issue:`1259`: Parameterless models
* :issue:`616`: python 2.6, python 3 in single codebase
* :issue:`1586`: Kalman Filter errors with new pyx
* :issue:`1565`: build_win_bdist*_py3*.bat are using the wrong compiler
* :issue:`843`: UnboundLocalError When trying to install OS X
* :issue:`713`: arima.fit performance
* :issue:`367`: unable to install on RHEL 5.6
* :issue:`1548`: testtransf error
* :issue:`1478`: is sm.tsa.filters.arfilter an AR filter?
* :issue:`1420`: GMM poisson test failures
* :issue:`1145`: test_multi noise
* :issue:`1539`: NegativeBinomial   strange results with bfgs
* :issue:`936`: vbench for statsmodels
* :issue:`1153`: Where are all our testing machines?
* :issue:`1500`: Use Miniconda for test builds
* :issue:`1526`: Out of date docs
* :issue:`1311`: BUG/BLD 3.4 compatibility of cython c files
* :issue:`1513`: build on osx -python-3.4
* :issue:`1497`: r2nparray needs NAMESPACE file
* :issue:`1502`: coveralls coverage report for files is broken
* :issue:`1501`: pandas in/out in predict
* :issue:`1494`: truncated violin plots
* :issue:`1443`: Crash from python.exe using linear regression of statsmodels
* :issue:`1462`: qqplot line kwarg is broken/docstring is wrong
* :issue:`1457`: BUG/BLD: Failed build if "sandbox" anywhere in statsmodels path
* :issue:`1441`: wls function: syntax error "unexpected EOF while parsing" occurs when name of dependent variable starts with digits
* :issue:`1428`: ipython_directive does not work with ipython master
* :issue:`1385`: SimpleTable in Summary (e.g. OLS) is slow for large models
* :issue:`1399`: UnboundLocalError: local variable 'fittedvalues' referenced before assignment
* :issue:`1377`: TestAnova2.test_results fails with pandas 0.13.1
* :issue:`1394`: multipletests: reducing memory consumption
* :issue:`1267`: Packages cannot have both pandas and statsmodels in install_requires
* :issue:`1359`: move graphics.tsa to tsa.graphics
* :issue:`356`: docs take up a lot of space
* :issue:`988`: AR.fit no precision options for fmin_l_bfgs_b
* :issue:`990`: AR fit with bfgs: large score
* :issue:`14`: arma with exog
* :issue:`1348`: reset_index + set_index with drop=False
* :issue:`1343`: ARMA does not pass missing keyword up to TimeSeriesModel
* :issue:`1326`: formula example notebook broken
* :issue:`1327`: typo in docu-code for "Outlier and Influence Diagnostic Measures"
* :issue:`1309`: Box-Cox transform (some code needed: lambda estimator)
* :issue:`1059`: sm.tsa.ARMA making ma invertibility
* :issue:`1295`: Bug in ARIMA forecasting when start is int len(endog) and dates are given
* :issue:`1285`: tsa models fail on PeriodIndex with pandas
* :issue:`1269`: KPSS test for stationary processes
* :issue:`1268`: Feature request: Exponential smoothing
* :issue:`1250`: DOCs error in var_plots
* :issue:`1032`: Poisson predict breaks on list
* :issue:`347`: minimum number of observations - document or check ?
* :issue:`1170`: WLS log likelihood, aic and bic
* :issue:`1187`:  sm.tsa.acovf fails when both unbiased and fft are True
* :issue:`1239`: sandbox kernels, problems with inDomain
* :issue:`1231`: sandbox kernels confint missing alpha
* :issue:`1245`: kernels cosine differs from Stata
* :issue:`823`: KDEUnivariate with weights
* :issue:`1229`: precision problems in degenerate case
* :issue:`1219`: select_order
* :issue:`1206`: REF: RegressionResults cov-HCx into cached attributes
* :issue:`1152`: statsmodels failing tests with pandas master
* :issue:`1195`: pyximport.install() before import api crash
* :issue:`1066`: gmm.IV2SLS has wrong predict signature
* :issue:`1186`: OLS when exog is 1d
* :issue:`1113`: TST: precision too high in test_normality
* :issue:`1159`: scipy version is still >= 0.7?
* :issue:`1108`: SyntaxError: unqualified exec is not allowed in function 'test_EvalEnvironment_capture_flag
* :issue:`1116`: Typo in Example Doc?
* :issue:`1123`: BUG : arima_model._get_predict_out_of_sample, ignores exogenous of there is no trend ?
* :issue:`1155`: ARIMA - The computed initial AR coefficients are not stationary
* :issue:`979`: Win64 binary cannot find Python installation
* :issue:`1046`: TST: test_arima_small_data_bug on current master
* :issue:`1146`: ARIMA fit failing for small set of data due to invalid maxlag
* :issue:`1081`: streamline linear algebra for linear model
* :issue:`1138`: BUG: pacf_yw does not demean
* :issue:`1127`: Allow linear link model with Binomial families
* :issue:`1122`: no data cleaning for statsmodels.genmod.families.varfuncs.NegativeBinomial()
* :issue:`658`: robust.mad is not being computed correctly or is non-standard definition; it returns the median
* :issue:`1076`: Some issues with ARMAX forecasting
* :issue:`1073`: easy_install sandbox violation
* :issue:`1115`: EasyInstall Problem
* :issue:`1106`: bug in robust.scale.mad?
* :issue:`1102`: Installation Problem
* :issue:`1084`: DataFrame.sort_index does not use ascending when then value is a list with a single element
* :issue:`393`: marginal effects in discrete choice do not have standard errors defined
* :issue:`1078`: Use pandas.version.short_version
* :issue:`96`: deepcopy breaks on ResettableCache
* :issue:`1055`: datasets.get_rdataset   string decode error on python 3
* :issue:`46`: tsa.stattools.acf confint needs checking and tests
* :issue:`957`: ARMA start estimate with numpy master
* :issue:`62`: GLSAR incorrect initial condition in whiten
* :issue:`1021`: from_formula() throws error - problem installing
* :issue:`911`: noise in stats.power tests
* :issue:`472`: Update roadmap for 0.5
* :issue:`238`: release 0.5
* :issue:`1006`: update nbconvert to IPython 1.0
* :issue:`1038`: DataFrame with integer names not handled in ARIMA
* :issue:`1036`: Series no longer inherits from ndarray
* :issue:`1028`: Test fail with windows and Anaconda - Low priority
* :issue:`676`: acorr_breush_godfrey  undefined nlags
* :issue:`922`: lowess returns inconsistent with option
* :issue:`425`: no bse in robust with norm=TrimmedMean
* :issue:`1025`: add_constant incorrectly detects constant column
