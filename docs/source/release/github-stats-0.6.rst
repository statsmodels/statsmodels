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

* :ghpull:`2044`: ENH: Allow unit interval for binary models. Closes #2040.
* :ghpull:`1426`: ENH: Import arima_process stuff into tsa.api
* :ghpull:`2042`: Fix two minor typos in contrast.py
* :ghpull:`2034`: ENH: Handle missing for extra data with formulas
* :ghpull:`2035`: MAINT: Remove deprecated code for 0.6
* :ghpull:`1325`: ENH: add the Edgeworth expansion based on the normal distribution
* :ghpull:`2032`: DOC: What it is what it is.
* :ghpull:`2031`: ENH: Expose patsy eval_env to users.
* :ghpull:`2028`: ENH: Fix numerical issues in links and families.
* :ghpull:`2029`: DOC: Fix versions to match other docs.
* :ghpull:`1647`: ENH: Warn on non-convergence.
* :ghpull:`2014`: BUG: Fix forecasting for ARIMA with d == 2
* :ghpull:`2013`: ENH: Better error message on object dtype
* :ghpull:`2012`: BUG: 2d 1 columns -> 1d. Closes #322.
* :ghpull:`2009`: DOC: Update after refactor. Use code block.
* :ghpull:`2008`: ENH: Add wrapper for MixedLM
* :ghpull:`1954`: ENH: PHReg formula improvements 
* :ghpull:`2007`: BLD: Fix build issues
* :ghpull:`2006`: BLD: Do not generate cython on clean. Closes #1852.
* :ghpull:`2000`: BLD: Let pip/setuptools handle dependencies that aren't installed at all.
* :ghpull:`1999`: Gee offset exposure 1994 rebased
* :ghpull:`1998`: BUG/ENH Lasso emptymodel rebased
* :ghpull:`1989`: BUG/ENH: WLS generic robust cov_type didn't use whitened, 
* :ghpull:`1587`: ENH: Wrap X12/X13-ARIMA AUTOMDL. Closes #442.
* :ghpull:`1563`: ENH: Add plot_predict method to ARIMA models.
* :ghpull:`1995`: BUG: Fix issue #1993
* :ghpull:`1981`: ENH: Add api for covstruct. Clear __init__. Closes #1917.
* :ghpull:`1996`: DEV: Ignore .venv file.
* :ghpull:`1982`: REF: Rename jac -> score_obs. Closes #1785.
* :ghpull:`1987`: BUG tsa pacf, base bootstrap
* :ghpull:`1986`: Bug multicomp 1927 rebased
* :ghpull:`1984`: Docs add gee.rst
* :ghpull:`1985`: Bug uncentered latex table 1929 rebased
* :ghpull:`1983`: BUG: Fix compat asunicode
* :ghpull:`1574`: DOC: Fix math.
* :ghpull:`1980`: DOC: Documentation fixes
* :ghpull:`1974`: REF/Doc beanplot change default color, add notebook
* :ghpull:`1978`: ENH: Check input to binary models
* :ghpull:`1979`: BUG: Typo
* :ghpull:`1976`: ENH: Add _repr_html_ to SimpleTable
* :ghpull:`1977`: BUG: Fix import refactor victim.
* :ghpull:`1975`: BUG: Yule walker cast to float
* :ghpull:`1973`: REF: Move and expose webuse
* :ghpull:`1972`: TST: Add testing against NumPy 1.9 and matplotlib 1.4
* :ghpull:`1939`: ENH: Binstar build files
* :ghpull:`1952`: REF/DOC: Misc
* :ghpull:`1940`: REF: refactor and speedup of mixed LME
* :ghpull:`1937`: ENH: Quick access to online documentation
* :ghpull:`1942`: DOC: Rename Change README type to rst
* :ghpull:`1938`: ENH: Enable Python 3.4 testing
* :ghpull:`1924`: Bug gee cov type 1906 rebased
* :ghpull:`1870`: robust covariance, cov_type in fit
* :ghpull:`1859`: BUG: Don't use negative indexing with k_ar == 0. Closes #1858.
* :ghpull:`1914`: BUG: LikelihoodModelResults.pvalues use df_resid_inference
* :ghpull:`1899`: TST: fix assert_equal for pandas index
* :ghpull:`1895`: Bug multicomp pandas
* :ghpull:`1894`: BUG fix more ix indexing cases for pandas compat
* :ghpull:`1889`: BUG: fix ytick positions closes #1561
* :ghpull:`1887`: Bug pandas compat asserts
* :ghpull:`1888`: TST test_corrpsd Test_Factor: add noise to data
* :ghpull:`1886`: BUG pandas 0.15 compatibility in grouputils labels
* :ghpull:`1885`: TST: corr_nearest_factor, more informative tests
* :ghpull:`1884`: Fix: Add compat code for pd.Categorical in pandas>=0.15
* :ghpull:`1883`: BUG: add _ctor_param to TransfGen distributions
* :ghpull:`1872`: TST: fix _infer_freq for pandas .14+ compat
* :ghpull:`1867`: Ref covtype fit
* :ghpull:`1865`: Disable tst distribution 1864
* :ghpull:`1856`: _spg_optim returns history of objective function values
* :ghpull:`1854`: BLD: Don't hard-code path for building notebooks. Closes #1249
* :ghpull:`1851`: MAINT: Cor nearest factor tests
* :ghpull:`1847`: Newton regularize
* :ghpull:`1623`: BUG Negbin fit regularized
* :ghpull:`1797`: BUG/ENH: fix and improve constant detection
* :ghpull:`1770`: TST: anova with `-1` noconstant, add tests
* :ghpull:`1837`: Allow group variable to be passed as variable name when using formula
* :ghpull:`1839`: BUG: GEE score
* :ghpull:`1830`: BUG/ENH Use t 
* :ghpull:`1832`: TST error with scipy 0.14 location distribution class
* :ghpull:`1827`: fit_regularized for linear models   rebase 1674
* :ghpull:`1825`: Phreg 1312 rebased
* :ghpull:`1826`: Lme api docs
* :ghpull:`1824`: Lme profile 1695 rebased
* :ghpull:`1823`: Gee cat subclass 1694 rebase
* :ghpull:`1781`: ENH: Glm add score_obs
* :ghpull:`1821`: Glm maint #1734 rebased
* :ghpull:`1820`: BUG: revert change to conf_int in PR #1819
* :ghpull:`1819`: Docwork
* :ghpull:`1772`: REF: cov_params allow case of only cov_params_default is defined 
* :ghpull:`1771`: REF numpy >1.9 compatibility, indexing into empty slice closes #1754
* :ghpull:`1769`: Fix ttest 1d
* :ghpull:`1766`: TST: TestProbitCG increase bound for fcalls closes #1690
* :ghpull:`1709`: BLD: Made build extensions more flexible
* :ghpull:`1714`: WIP: fit_constrained
* :ghpull:`1706`: REF: Use fixed params in test. Closes #910.
* :ghpull:`1701`: BUG: Fix faulty logic. Do not raise when missing='raise' and no missing data.
* :ghpull:`1699`: TST/ENH StandardizeTransform, reparameterize TestProbitCG
* :ghpull:`1697`: Fix for statsmodels/statsmodels#1689
* :ghpull:`1692`: OSL Example: redundant cell in example removed
* :ghpull:`1688`: Kshedden mixed rebased of #1398
* :ghpull:`1629`: Pull request to fix bandwidth bug in issue 597
* :ghpull:`1666`: Include pyx in sdist but don't install
* :ghpull:`1683`: TST: GLM shorten random seed closes #1682
* :ghpull:`1681`: Dotplot kshedden rebased of 1294
* :ghpull:`1679`: BUG: Fix problems with predict handling offset and exposure
* :ghpull:`1677`: Update docstring of RegressionModel.predict()
* :ghpull:`1635`: Allow offset and exposure to be used together with log link; raise excep...
* :ghpull:`1676`: Tests for SVAR
* :ghpull:`1671`: ENH: avoid hard-listed bandwidths -- use present dictionary (+typos fixed)
* :ghpull:`1643`: Allow matrix structure in covariance matrices to be exploited
* :ghpull:`1657`: BUG: Fix refactor victim.
* :ghpull:`1630`: DOC: typo, "interecept"
* :ghpull:`1619`: MAINT: Dataset docs cleanup and automatic build of docs
* :ghpull:`1612`: BUG/ENH Fix negbin exposure #1611
* :ghpull:`1610`: BUG/ENH fix llnull, extra kwds to recreate model
* :ghpull:`1582`: BUG: wls_prediction_std fix weight handling, see 987
* :ghpull:`1613`: BUG: Fix proportions allpairs #1493
* :ghpull:`1607`: TST: adjust precision, CI Debian, Ubuntu testing
* :ghpull:`1603`: ENH: Allow start_params in GLM
* :ghpull:`1600`: CLN: Regression plots fixes
* :ghpull:`1592`: DOC: Additions and fixes 
* :ghpull:`1520`: CLN: Refactored so that there is no longer a need for 2to3
* :ghpull:`1585`: Cor nearest 1384 rebased
* :ghpull:`1553`: Gee maint 1528 rebased
* :ghpull:`1583`: BUG: For ARMA(0,0) ensure 1d bse and fix summary.
* :ghpull:`1580`: DOC: Fix links. [skip ci]
* :ghpull:`1572`: DOC: Fix link title [skip ci]
* :ghpull:`1566`: BLD: Fix copy paste path error for >= 3.3 Windows builds
* :ghpull:`1524`: ENH: Optimize Cython code. Use scipy blas function pointers.
* :ghpull:`1560`: ENH: Allow ARMA(0,0) in order selection
* :ghpull:`1559`: MAINT: Recover lost commits from vbench PR
* :ghpull:`1554`: Silenced test output introduced in medcouple
* :ghpull:`1234`: ENH: Robust skewness, kurtosis and medcouple measures
* :ghpull:`1484`: ENH: Add naive seasonal decomposition function
* :ghpull:`1551`: COMPAT: Fix failing test on Python 2.6
* :ghpull:`1472`: ENH: using human-readable group names instead of integer ids in MultiComparison
* :ghpull:`1437`: ENH: accept non-int definitions of cluster groups
* :ghpull:`1550`: Fix test gmm poisson
* :ghpull:`1549`: TST: Fix locally failing tests.
* :ghpull:`1121`: WIP: Refactor optimization code.
* :ghpull:`1547`: COMPAT: Correct bit_length for 2.6
* :ghpull:`1545`: MAINT: Fix missed usage of deprecated tools.rank
* :ghpull:`1196`: REF: ensure O(N log N) when using fft for acf
* :ghpull:`1154`: DOC: Add links for build machines.
* :ghpull:`1546`: DOC: Fix link to wrong notebook
* :ghpull:`1383`: MAINT: Deprecate rank in favor of np.linalg.matrix_rank
* :ghpull:`1432`: COMPAT: Add NumpyVersion from scipy
* :ghpull:`1438`: ENH: Option to avoid "center" environment.
* :ghpull:`1544`: BUG: Travis miniconda
* :ghpull:`1510`: CLN: Improve warnings to avoid generic warnings messages
* :ghpull:`1543`: TST: Suppress RuntimeWarning for L-BFGS-B
* :ghpull:`1507`: CLN: Silence test output
* :ghpull:`1540`: BUG: Correct derivative for exponential transform.
* :ghpull:`1536`: BUG: Restores coveralls for a single build
* :ghpull:`1535`: BUG: Fixes for 2.6 test failures, replacing astype(str) with apply(str)
* :ghpull:`1523`: Travis miniconda
* :ghpull:`1533`: DOC: Fix link to code on github
* :ghpull:`1531`: DOC: Fix stale links with linkcheck
* :ghpull:`1530`: DOC: Fix link
* :ghpull:`1527`: DOCS: Update docs add FAQ page
* :ghpull:`1525`: DOC: Update with Python 3.4 build notes
* :ghpull:`1518`: DOC: Ask for release notes and example.
* :ghpull:`1516`: DOC: Update examples contributing docs for current practice.
* :ghpull:`1517`: DOC: Be clear about data attribute of Datasets
* :ghpull:`1515`: DOC: Fix broken link
* :ghpull:`1514`: DOC: Fix formula import convention.
* :ghpull:`1506`: BUG: Format and decode errors in Python 2.6
* :ghpull:`1505`: TST: Test co2 load_data for Python 3.
* :ghpull:`1504`: BLD: New R versions require NAMESPACE file. Closes #1497.
* :ghpull:`1483`: ENH: Some utility functions for working with dates
* :ghpull:`1482`: REF: Prefer filters.api to __init__
* :ghpull:`1481`: ENH: Add weekly co2 dataset
* :ghpull:`1474`: DOC: Add plots for standard filter methods.
* :ghpull:`1471`: DOC: Fix import
* :ghpull:`1470`: DOC/BLD: Log code exceptions from nbgenerate
* :ghpull:`1469`: DOC: Fix bad links
* :ghpull:`1468`: MAINT: CSS fixes
* :ghpull:`1463`: DOC: Remove defunct argument. Change default kw. Closes #1462.
* :ghpull:`1452`: STY: import pandas as pd
* :ghpull:`1458`: BUG/BLD: exclude sandbox in relative path, not absolute
* :ghpull:`1447`: DOC: Only build and upload docs if we need to.
* :ghpull:`1445`: DOCS: Example landing page
* :ghpull:`1436`: DOC: Fix auto doc builds.
* :ghpull:`1431`: DOC: Add default for getenv. Fix paths. Add print_info
* :ghpull:`1429`: MAINT: Use ip_directive shipped with IPython
* :ghpull:`1427`: TST: Make tests fit quietly
* :ghpull:`1424`: ENH: Consistent results for transform_slices
* :ghpull:`1421`: ENH: Add grouping utilities code
* :ghpull:`1419`: Gee 1314 rebased
* :ghpull:`1414`: TST temporarily rename tests probplot other to skip them
* :ghpull:`1403`: Bug norm expan shapes
* :ghpull:`1417`: REF: Let subclasses keep kwds attached to data.
* :ghpull:`1416`: ENH: Make handle_data overwritable by subclasses.
* :ghpull:`1410`: ENH: Handle missing is none
* :ghpull:`1402`: REF: Expose missing data handling as classmethod
* :ghpull:`1387`: MAINT: Fix failing tests
* :ghpull:`1406`: MAINT: Tools improvements
* :ghpull:`1404`: Tst fix genmod link tests
* :ghpull:`1396`: REF: Multipletests reduce memory usage
* :ghpull:`1380`: DOC :Update vector_ar.rst
* :ghpull:`1381`: BLD: Don't check dependencies on egg_info for pip. Closes #1267.
* :ghpull:`1302`: BUG: Fix typo.
* :ghpull:`1375`: STY: Remove unused imports and comment out unused libraries in setup.py
* :ghpull:`1143`: DOC: Update backport notes for new workflow.
* :ghpull:`1374`: ENH: Import tsaplots into tsa namespace. Closes #1359.
* :ghpull:`1369`: STY: Pep-8 cleanup
* :ghpull:`1370`: ENH: Support ARMA(0,0) models.
* :ghpull:`1368`: STY: Pep 8 cleanup
* :ghpull:`1367`: ENH: Make sure mle returns attach to results.
* :ghpull:`1365`: STY: Import and pep 8 cleanup
* :ghpull:`1364`: ENH: Get rid of hard-coded lbfgs. Closes #988.
* :ghpull:`1363`: BUG: Fix typo.
* :ghpull:`1361`: ENH: Attach mlefit to results not model.
* :ghpull:`1360`: ENH: Import adfuller into tsa namespace
* :ghpull:`1346`: STY: PEP-8 Cleanup
* :ghpull:`1344`: BUG: Use missing keyword given to ARMA.
* :ghpull:`1340`: ENH: Protect against ARMA convergence failures.
* :ghpull:`1334`: ENH: ARMA order select convenience function
* :ghpull:`1339`: Fix typos
* :ghpull:`1336`: REF: Get rid of plain assert.
* :ghpull:`1333`: STY: __all__ should be after imports.
* :ghpull:`1332`: ENH: Add Bunch object to tools.
* :ghpull:`1331`: ENH: Always use unicode.
* :ghpull:`1329`: BUG: Decode metadata to utf-8. Closes #1326.
* :ghpull:`1330`: DOC: Fix typo. Closes #1327.
* :ghpull:`1185`: Added support for pandas when pandas was installed directly from git trunk
* :ghpull:`1315`: MAINT: Change back to path for build box
* :ghpull:`1305`: TST: Update hard-coded path.
* :ghpull:`1290`: ENH: Add seasonal plotting.
* :ghpull:`1296`: BUG/TST: Fix ARMA forecast when start == len(endog). Closes #1295
* :ghpull:`1292`: DOC: cleanup examples folder and webpage
* :ghpull:`1286`: Make sure PeriodIndex passes through tsa. Closes #1285.
* :ghpull:`1271`: Silverman enhancement - Issue #1243 
* :ghpull:`1264`: Doc work GEE, GMM, sphinx warnings
* :ghpull:`1179`: REF/TST: `ProbPlot` now uses `resettable_cache` and added some kwargs to plotting fxns
* :ghpull:`1225`: Sandwich mle
* :ghpull:`1258`: Gmm new rebased
* :ghpull:`1255`: ENH add GEE to genmod
* :ghpull:`1254`: REF: Results.predict convert to array and adjust shape
* :ghpull:`1192`: TST: enable tests for llf after change to WLS.loglike see #1170
* :ghpull:`1253`: Wls llf fix
* :ghpull:`1233`: sandbox kernels bugs uniform kernel and confint
* :ghpull:`1240`: Kde weights 1103 823
* :ghpull:`1228`: Add default value tags to adfuller() docs
* :ghpull:`1198`: fix typo
* :ghpull:`1230`: BUG: numerical precision in resid_pearson with perfect fit #1229
* :ghpull:`1214`: Compare lr test rebased
* :ghpull:`1200`: BLD: do not install \*.pyx \*.c  MANIFEST.in
* :ghpull:`1202`: MAINT: Sort backports to make applying easier.
* :ghpull:`1157`: Tst precision master
* :ghpull:`1161`: add a fitting interface for simultaneous log likelihood and score, for lbfgs, tested with MNLogit
* :ghpull:`1160`: DOC: update scipy version from 0.7 to 0.9.0
* :ghpull:`1147`: ENH: add lbfgs for fitting
* :ghpull:`1156`: ENH: Raise on 0,0 order models in AR(I)MA. Closes #1123
* :ghpull:`1149`: BUG: Fix small data issues for ARIMA.
* :ghpull:`1092`: Fixed duplicate svd in RegressionModel
* :ghpull:`1139`: TST: Silence tests
* :ghpull:`1135`: Misc style
* :ghpull:`1088`: ENH: add predict_prob to poisson
* :ghpull:`1125`: REF/BUG: Some GLM cleanup. Used trimmed results in NegativeBinomial variance.
* :ghpull:`1124`: BUG: Fix ARIMA prediction when fit without a trend.
* :ghpull:`1118`: DOC: Update gettingstarted.rst
* :ghpull:`1117`: Update ex_arma2.py
* :ghpull:`1107`: REF: Deprecate stand_mad. Add center keyword to mad. Closes #658.
* :ghpull:`1089`: ENH: exp(poisson.logpmf()) for poisson better behaved.
* :ghpull:`1077`: BUG: Allow 1d exog in ARMAX forecasting.
* :ghpull:`1075`: BLD: Fix build issue on some versions of easy_install.
* :ghpull:`1071`: Update setup.py to fix broken install on OSX
* :ghpull:`1052`: DOC: Updating contributing docs
* :ghpull:`1136`: RLS: Add IPython tools for easier backporting of issues.
* :ghpull:`1091`: DOC: minor git typo
* :ghpull:`1082`: coveralls support
* :ghpull:`1072`: notebook examples title cell
* :ghpull:`1056`: Example: reg diagnostics
* :ghpull:`1057`: COMPAT: Fix py3 caching for get_rdatasets.
* :ghpull:`1045`: DOC/BLD: Update from nbconvert to IPython 1.0.
* :ghpull:`1026`: DOC/BLD: Add LD_LIBRARY_PATH to env for docs build.

Issues (252):

* :ghissue:`2040`: enh: fractional Logit, Probit
* :ghissue:`1220`: missing in extra data (example sandwiches, robust covariances)
* :ghissue:`1877`: error with GEE on missing data.
* :ghissue:`805`: nan with categorical in formula
* :ghissue:`2036`: test in links require exact class so Logit can't work in place of logit
* :ghissue:`2010`: Go over deprecations again for 0.6.
* :ghissue:`1303`: patsy library not automatically installed
* :ghissue:`2024`: genmod Links numerical improvements
* :ghissue:`2025`: GEE requires exact import for cov_struct
* :ghissue:`2017`: Matplotlib warning about too many figures
* :ghissue:`724`: check warnings
* :ghissue:`1562`: ARIMA forecasts are hard-coded for d=1
* :ghissue:`880`: DataFrame with bool type not cast correctly.
* :ghissue:`1992`: MixedLM style
* :ghissue:`322`: acf / pacf do not work on pandas objects
* :ghissue:`1317`: AssertionError: attr is not equal [dtype]: dtype('object') != dtype('datetime64[ns]')
* :ghissue:`1875`: dtype bug object arrays (raises in clustered standard errors code)
* :ghissue:`1842`: dtype object, glm.fit() gives AttributeError: sqrt
* :ghissue:`1300`: Doc errors, missing 
* :ghissue:`1164`: RLM cov_params, t_test, f_test don't use bcov_scaled
* :ghissue:`1019`: 0.6.0 Roadmap
* :ghissue:`554`: Prediction Standard Errors
* :ghissue:`333`: ENH tools: squeeze in R export file
* :ghissue:`1990`: MixedLM does not have a wrapper
* :ghissue:`1897`: Consider depending on setuptools in setup.py
* :ghissue:`2003`: pip install now fails silently
* :ghissue:`1852`: do not cythonize when cleaning up
* :ghissue:`1991`: GEE formula interface does not take offset/exposure
* :ghissue:`442`: Wrap x-12 arima
* :ghissue:`1993`: MixedLM bug
* :ghissue:`1917`: API: GEE access to genmod.covariance_structure through api
* :ghissue:`1785`: REF: rename jac -> score_obs
* :ghissue:`1969`: pacf has incorrect standard errors for lag 0
* :ghissue:`1434`: A small bug in GenericLikelihoodModelResults.bootstrap()
* :ghissue:`1408`: BUG test failure with tsa_plots
* :ghissue:`1337`: DOC: HCCM are now available for WLS
* :ghissue:`546`: influence and outlier documentation
* :ghissue:`1532`: DOC: Related page is out of date
* :ghissue:`1386`: Add minimum matplotlib to docs
* :ghissue:`1068`: DOC: keeping documentation of old versions on sourceforge
* :ghissue:`329`: link to examples and datasets from module pages
* :ghissue:`1804`: PDF documentation for statsmodels
* :ghissue:`202`: Extend robust standard errors for WLS/GLS
* :ghissue:`1519`: Link to user-contributed examples in docs
* :ghissue:`1053`: inconvenient: logit when endog is (1,2) instead of (0,1)
* :ghissue:`1555`: SimpleTable: add repr html for ipython notebook
* :ghissue:`1366`: Change default start_params to .1 in ARMA 
* :ghissue:`1869`: yule_walker (from `statsmodels.regression`) raises exception when given an integer array
* :ghissue:`1651`: statsmodels.tsa.ar_model.ARResults.predict
* :ghissue:`1738`: GLM robust sandwich covariance matrices
* :ghissue:`1779`: Some directories under statsmodels dont have __init_.py
* :ghissue:`1242`: No support for (0, 1, 0) ARIMA Models
* :ghissue:`1571`: expose webuse, use cache
* :ghissue:`1860`: ENH/BUG/DOC: Bean plot should allow for separate widths of bean and violins.
* :ghissue:`1831`: TestRegressionNM.test_ci_beta2 i386 AssertionError
* :ghissue:`1079`: bugfix release 0.5.1
* :ghissue:`1338`: Raise Warning for HCCM use in WLS/GLS
* :ghissue:`1430`: scipy min version / issue
* :ghissue:`276`: memoize, last argument wins, how to attach sandwich to Results?
* :ghissue:`1943`: REF/ENH: LikelihoodModel.fit optimization, make hessian optional
* :ghissue:`1957`: BUG: Re-create OLS model using _init_keys
* :ghissue:`1905`: Docs: online docs are missing GEE
* :ghissue:`1898`: add python 3.4 to continuous integration testing
* :ghissue:`1684`: BUG: GLM NegativeBinomial: llf ignores offset and exposure
* :ghissue:`1256`: REF: GEE handling of default covariance matrices
* :ghissue:`1760`: Changing covariance_type on results
* :ghissue:`1906`: BUG: GEE default covariance is not used
* :ghissue:`1931`: BUG: GEE subclasses NominalGEE don't work with pandas exog 
* :ghissue:`1904`: GEE Results doesn't have a Wrapper
* :ghissue:`1918`: GEE: required attributes missing, df_resid
* :ghissue:`1919`: BUG GEE.predict uses link instead of link.inverse
* :ghissue:`1858`: BUG: arimax forecast should special case k_ar == 0 
* :ghissue:`1903`: BUG: pvalues for cluster robust, with use_t don't use df_resid_inference
* :ghissue:`1243`: kde silverman bandwidth for non-gaussian kernels
* :ghissue:`1866`: Pip dependencies
* :ghissue:`1850`: TST test_corr_nearest_factor fails on Ubuntu
* :ghissue:`292`: python 3 examples
* :ghissue:`1868`: ImportError: No module named compat  [ from statsmodels.compat import lmap ]
* :ghissue:`1890`: BUG tukeyhsd nan in group labels
* :ghissue:`1891`: TST test_gmm outdated pandas, compat
* :ghissue:`1561`: BUG plot for tukeyhsd, MultipleComparison
* :ghissue:`1864`: test failure sandbox distribution transformation with scipy 0.14.0
* :ghissue:`576`: Add contributing guidelines
* :ghissue:`1873`: GenericLikelihoodModel is not picklable
* :ghissue:`1822`: TST failure on Ubuntu pandas 0.14.0 , problems with frequency
* :ghissue:`1249`: Source directory problem for notebook examples
* :ghissue:`1855`: anova_lm throws error on models created from api.ols but not formula.api.ols 
* :ghissue:`1853`: a large number of hardcoded paths
* :ghissue:`1792`: R² adjusted strange after including interaction term
* :ghissue:`1794`: REF: has_constant, k_constant, include implicit constant detection in base
* :ghissue:`1454`: NegativeBinomial missing fit_regularized method
* :ghissue:`1615`: REF DRYing fit methods
* :ghissue:`1453`: Discrete NegativeBinomialModel regularized_fit ValueError: matrices are not aligned
* :ghissue:`1836`: BUG Got an TypeError trying to import statsmodels.api
* :ghissue:`1829`: BUG: GLM summary show "t"  use_t=True for summary
* :ghissue:`1828`: BUG summary2 doesn't propagate/use use_t
* :ghissue:`1812`: BUG/ REF conf_int and use_t
* :ghissue:`1835`: Problems with installation using easy_install
* :ghissue:`1801`: BUG 'f_gen' missing in scipy 0.14.0
* :ghissue:`1803`: Error revealed by numpy 1.9.0r1
* :ghissue:`1834`: stackloss
* :ghissue:`1728`: GLM.fit maxiter=0  incorrect
* :ghissue:`1795`: singular design with offset ?
* :ghissue:`1730`: ENH/Bug cov_params, generalize, avoid ValueError
* :ghissue:`1754`: BUG/REF: assignment to slices in numpy >= 1.9 (emplike)
* :ghissue:`1409`: GEE test errors on Debian Wheezy
* :ghissue:`1521`: ubuntu failues: tsa_plot and grouputils
* :ghissue:`1415`: test failure test_arima.test_small_data 
* :ghissue:`1213`: df_diff in anova_lm
* :ghissue:`1323`: Contrast Results after t_test summary broken for 1 parameter
* :ghissue:`109`: TestProbitCG failure on Ubuntu
* :ghissue:`1690`: TestProbitCG: 8 failing tests (Python 3.4 / Ubuntu 12.04)
* :ghissue:`1763`: Johansen method doesn't give correct index values
* :ghissue:`1761`: doc build failures: ipython version ? ipython directive
* :ghissue:`1762`: Unable to build
* :ghissue:`1745`: UnicodeDecodeError raised by get_rdataset("Guerry", "HistData")
* :ghissue:`611`: test failure foreign with pandas 0.7.3
* :ghissue:`1700`: faulty logic in missing handling
* :ghissue:`1648`: ProbitCG failures
* :ghissue:`1689`: test_arima.test_small_data: SVD fails to converge (Python 3.4 / Ubuntu 12.04)
* :ghissue:`597`: BUG: nonparametric: kernel, efficient=True changes bw even if given
* :ghissue:`1606`: BUILD from sdist broken if cython available
* :ghissue:`1246`: test failure test_anova.TestAnova2.test_results
* :ghissue:`50`: t_test, f_test, model.py for normal instead of t-distribution
* :ghissue:`1655`: newey-west different than R?
* :ghissue:`1682`: TST test failure on Ubuntu, random.seed
* :ghissue:`1614`: docstring for regression.linear_model.RegressionModel.predict() does not match implementation
* :ghissue:`1318`: GEE and GLM scale parameter 
* :ghissue:`519`: L1 fit_regularized cleanup, comments
* :ghissue:`651`: add structure to example page
* :ghissue:`1067`: Kalman Filter convergence. How close is close enough?
* :ghissue:`1281`: Newton convergence failure prints warnings instead of warning
* :ghissue:`1628`: Unable to install statsmodels in the same requirements file as numpy, pandas, etc.
* :ghissue:`617`: Problem in installing statsmodel in Fedora 17 64-bit
* :ghissue:`935`: ll_null in likelihoodmodels discrete
* :ghissue:`704`: datasets.sunspot: wrong link in description
* :ghissue:`1222`: NegativeBinomial ignores exposure
* :ghissue:`1611`: BUG NegativeBinomial ignores exposure and offset
* :ghissue:`1608`: BUG: NegativeBinomial, llnul is always default 'nb2'
* :ghissue:`1221`: llnull with exposure ?
* :ghissue:`1493`: statsmodels.stats.proportion.proportions_chisquare_allpairs has hardcoded value
* :ghissue:`1260`: GEE test failure on Debian
* :ghissue:`1261`: test failure on Debian
* :ghissue:`443`: GLM.fit does not allow start_params
* :ghissue:`1602`: Fitting GLM with a pre-assigned starting parameter
* :ghissue:`1601`: Fitting GLM with a pre-assigned starting parameter
* :ghissue:`890`: regression_plots problems (pylint) and missing test coverage
* :ghissue:`1598`: Is "old" string formatting Python 3 compatible?
* :ghissue:`1589`: AR vs ARMA order specification
* :ghissue:`1134`: Mark knownfails
* :ghissue:`1259`: Parameterless models
* :ghissue:`616`: python 2.6, python 3 in single codebase
* :ghissue:`1586`: Kalman Filter errors with new pyx
* :ghissue:`1565`: build_win_bdist*_py3*.bat are using the wrong compiler
* :ghissue:`843`: UnboundLocalError When trying to install OS X
* :ghissue:`713`: arima.fit performance
* :ghissue:`367`: unable to install on RHEL 5.6
* :ghissue:`1548`: testtransf error
* :ghissue:`1478`: is sm.tsa.filters.arfilter an AR filter?
* :ghissue:`1420`: GMM poisson test failures
* :ghissue:`1145`: test_multi noise
* :ghissue:`1539`: NegativeBinomial   strange results with bfgs
* :ghissue:`936`: vbench for statsmodels
* :ghissue:`1153`: Where are all our testing machines?
* :ghissue:`1500`: Use Miniconda for test builds
* :ghissue:`1526`: Out of date docs
* :ghissue:`1311`: BUG/BLD 3.4 compatibility of cython c files
* :ghissue:`1513`: build on osx -python-3.4
* :ghissue:`1497`: r2nparray needs NAMESPACE file
* :ghissue:`1502`: coveralls coverage report for files is broken
* :ghissue:`1501`: pandas in/out in predict
* :ghissue:`1494`: truncated violin plots
* :ghissue:`1443`: Crash from python.exe using linear regression of statsmodels 
* :ghissue:`1462`: qqplot line kwarg is broken/docstring is wrong
* :ghissue:`1457`: BUG/BLD: Failed build if "sandbox" anywhere in statsmodels path
* :ghissue:`1441`: wls function: syntax error "unexpected EOF while parsing" occurs when name of dependent variable starts with digits
* :ghissue:`1428`: ipython_directive doesn't work with ipython master
* :ghissue:`1385`: SimpleTable in Summary (e.g. OLS) is slow for large models
* :ghissue:`1399`: UnboundLocalError: local variable 'fittedvalues' referenced before assignment
* :ghissue:`1377`: TestAnova2.test_results fails with pandas 0.13.1
* :ghissue:`1394`: multipletests: reducing memory consumption
* :ghissue:`1267`: Packages cannot have both pandas and statsmodels in install_requires
* :ghissue:`1359`: move graphics.tsa to tsa.graphics
* :ghissue:`356`: docs take up a lot of space
* :ghissue:`988`: AR.fit no precision options for fmin_l_bfgs_b
* :ghissue:`990`: AR fit with bfgs: large score
* :ghissue:`14`: arma with exog
* :ghissue:`1348`: reset_index + set_index with drop=False
* :ghissue:`1343`: ARMA doesn't pass missing keyword up to TimeSeriesModel
* :ghissue:`1326`: formula example notebook broken
* :ghissue:`1327`: typo in docu-code for "Outlier and Influence Diagnostic Measures"
* :ghissue:`1309`: Box-Cox transform (some code needed: lambda estimator)
* :ghissue:`1059`: sm.tsa.ARMA making ma invertibility
* :ghissue:`1295`: Bug in ARIMA forecasting when start is int len(endog) and dates are given
* :ghissue:`1285`: tsa models fail on PeriodIndex with pandas 
* :ghissue:`1269`: KPSS test for stationary processes
* :ghissue:`1268`: Feature request: Exponential smoothing
* :ghissue:`1250`: DOCs error in var_plots
* :ghissue:`1032`: Poisson predict breaks on list
* :ghissue:`347`: minimum number of observations - document or check ?
* :ghissue:`1170`: WLS log likelihood, aic and bic
* :ghissue:`1187`:  sm.tsa.acovf fails when both unbiased and fft are True
* :ghissue:`1239`: sandbox kernels, problems with inDomain
* :ghissue:`1231`: sandbox kernels confint missing alpha
* :ghissue:`1245`: kernels cosine differs from Stata
* :ghissue:`823`: KDEUnivariate with weights
* :ghissue:`1229`: precision problems in degenerate case
* :ghissue:`1219`: select_order
* :ghissue:`1206`: REF: RegressionResults cov-HCx into cached attributes
* :ghissue:`1152`: statsmodels failing tests with pandas master
* :ghissue:`1195`: pyximport.install() before import api crash
* :ghissue:`1066`: gmm.IV2SLS has wrong predict signature
* :ghissue:`1186`: OLS when exog is 1d
* :ghissue:`1113`: TST: precision too high in test_normality
* :ghissue:`1159`: scipy version is still >= 0.7?
* :ghissue:`1108`: SyntaxError: unqualified exec is not allowed in function 'test_EvalEnvironment_capture_flag
* :ghissue:`1116`: Typo in Example Doc?
* :ghissue:`1123`: BUG : arima_model._get_predict_out_of_sample, ignores exogenous of there is no trend ?
* :ghissue:`1155`: ARIMA - The computed initial AR coefficients are not stationary
* :ghissue:`979`: Win64 binary can't find Python installation
* :ghissue:`1046`: TST: test_arima_small_data_bug on current master 
* :ghissue:`1146`: ARIMA fit failing for small set of data due to invalid maxlag
* :ghissue:`1081`: streamline linear algebra for linear model
* :ghissue:`1138`: BUG: pacf_yw doesn't demean
* :ghissue:`1127`: Allow linear link model with Binomial families
* :ghissue:`1122`: no data cleaning for statsmodels.genmod.families.varfuncs.NegativeBinomial()
* :ghissue:`658`: robust.mad is not being computed correctly or is non-standard definition; it returns the median
* :ghissue:`1076`: Some issues with ARMAX forecasting
* :ghissue:`1073`: easy_install sandbox violation
* :ghissue:`1115`: EasyInstall Problem
* :ghissue:`1106`: bug in robust.scale.mad?
* :ghissue:`1102`: Installation Problem
* :ghissue:`1084`: DataFrame.sort_index does not use ascending when then value is a list with a single element
* :ghissue:`393`: marginal effects in discrete choice do not have standard errors defined
* :ghissue:`1078`: Use pandas.version.short_version
* :ghissue:`96`: deepcopy breaks on ResettableCache
* :ghissue:`1055`: datasets.get_rdataset   string decode error on python 3
* :ghissue:`46`: tsa.stattools.acf confint needs checking and tests
* :ghissue:`957`: ARMA start estimate with numpy master
* :ghissue:`62`: GLSAR incorrect initial condition in whiten
* :ghissue:`1021`: from_formula() throws error - problem installing
* :ghissue:`911`: noise in stats.power tests
* :ghissue:`472`: Update roadmap for 0.5
* :ghissue:`238`: release 0.5
* :ghissue:`1006`: update nbconvert to IPython 1.0
* :ghissue:`1038`: DataFrame with integer names not handled in ARIMA
* :ghissue:`1036`: Series no longer inherits from ndarray
* :ghissue:`1028`: Test fail with windows and Anaconda - Low priority
* :ghissue:`676`: acorr_breush_godfrey  undefined nlags
* :ghissue:`922`: lowess returns inconsistent with option
* :ghissue:`425`: no bse in robust with norm=TrimmedMean
* :ghissue:`1025`: add_constant incorrectly detects constant column
