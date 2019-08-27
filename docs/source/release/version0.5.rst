=============
Release 0.5.0
=============

statsmodels 0.5 is a large and very exciting release that brings together a year of work done by 38 authors, including over 2000 commits. It contains many new features and a large amount of bug fixes detailed below.

See the :ref:`list of fixed issues <issues_list_05>` for specific closed issues.

The following major new features appear in this version.

Support for Model Formulas via Patsy
====================================

statsmodels now supports fitting models with a formula. This functionality is provided by `patsy <https://patsy.readthedocs.org/en/latest/>`_. Patsy is now a dependency for statsmodels. Models can be individually imported from the ``statsmodels.formula.api`` namespace or you can import them all as::

    import statsmodels.formula.api as smf

Alternatively, each model in the usual ``statsmodels.api`` namespace has a ``from_formula`` classmethod that will create a model using a formula. Formulas are also available for specifying linear hypothesis tests using the ``t_test`` and ``f_test`` methods after model fitting. A typical workflow can now look something like this.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/csv/HistData/Guerry.csv'
    data = pd.read_csv(url)

    # Fit regression model (using the natural log of one of the regressors)
    results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=data).fit()

See :ref:`here for some more documentation of using formulas in statsmodels <formula_examples>`

Empirical Likelihood (Google Summer of Code 2012 project)
---------------------------------------------------------

Empirical Likelihood-Based Inference for moments of univariate and multivariate variables is available as well as EL-based ANOVA tests. EL-based linear regression, including the regression through the origin model. In addition, the accelerated failure time model for inference on a linear regression model with a randomly right censored endogenous variable is available.

Analysis of Variance (ANOVA) Modeling
-------------------------------------

Support for ANOVA is now available including type I, II, and III sums of squares. See :ref:`anova`.

.. currentmodule:: statsmodels.nonparametric

Multivariate Kernel Density Estimators (GSoC 2012 project)
----------------------------------------------------------

Kernel density estimation has been extended to handle multivariate estimation as well via product kernels. It is available as :class:`sm.nonparametric.KDEMultivariate <kernel_density.KDEMultivariate>`. It supports least squares and maximum likelihood cross-validation for bandwidth estimation, as well as mixed continuous, ordered, and unordered categorical data. Conditional density estimation is also available via :class:`sm.nonparametric.KDEMUltivariateConditional <kernel_density.KDEMultivariateConditional>`.

Nonparameteric Regression (GSoC 2012 project)
---------------------------------------------
Kernel regression models are now available via :class:`sm.nonparametric.KernelReg <kernel_regression.KernelReg>`. It is based on the product kernel mentioned above, so it also has the same set of features including support for cross-validation as well as support for estimation mixed continuous and categorical variables. Censored kernel regression is also provided by `kernel_regression.KernelCensoredReg`.

Quantile Regression Model
-------------------------

.. currentmodule:: statsmodels.regression.quantile_regression

Quantile regression is supported via the :class:`sm.QuantReg <QuantReg>` class. Kernel and bandwidth selection options are available for estimating the asymptotic covariance matrix using a kernel density estimator.

Negative Binomial Regression Model
----------------------------------

.. currentmodule:: statsmodels.discrete.discrete_model

It is now possible to fit negative binomial models for count data via maximum-likelihood using the :class:`sm.NegativeBinomial <NegativeBinomial>` class. ``NB1``, ``NB2``, and ``geometric`` variance specifications are available.

l1-penalized Discrete Choice Models
-----------------------------------

A new optimization method has been added to the discrete models, which includes Logit, Probit, MNLogit and Poisson, that makes it possible to estimate the models with an l1, linear, penalization. This shrinks parameters towards zero and can set parameters that are not very different from zero to zero. This is especially useful if there are a large number of explanatory variables and a large associated number of parameters. `CVXOPT <https://cvxopt.org/>`_ is now an optional dependency that can be used for fitting these models.

New and Improved Graphics
-------------------------

.. currentmodule:: statsmodels.graphics

* **ProbPlot**: A new `ProbPlot` object has been added to provide a simple interface to create P-P, Q-Q, and probability plots with options to fit a distribution and show various reference lines. In the case of Q-Q and P-P plots, two different samples can be compared with the `other` keyword argument. :func:`sm.graphics.ProbPlot <gofplots.ProbPlot>`

.. code-block:: python
   
   import numpy as np
   import statsmodels.api as sm
   x = np.random.normal(loc=1.12, scale=0.25, size=37)
   y = np.random.normal(loc=0.75, scale=0.45, size=37)
   ppx = sm.ProbPlot(x)
   ppy =  sm.ProbPlot(y)
   fig1 = ppx.qqplot()
   fig2 = ppx.qqplot(other=ppy)

* **Mosaic Plot**: Create a mosaic plot from a contingency table. This allows you to visualize multivariate categorical data in a rigorous and informative way. Available with :func:`sm.graphics.mosaic <mosaicplot.mosaic>`.

* **Interaction Plot**: Interaction plots now handle categorical factors as well as other improvements. :func:`sm.graphics.interaction_plot <factorplots.interaction_plot>`.

* **Regression Plots**: The regression plots have been refactored and improved. They can now handle pandas objects and regression results instances appropriately. See :func:`sm.graphics.plot_fit <regressionplots.plot_fit>`, :func:`sm.graphics.plot_regress_exog <regressionplots.plot_regress_exog>`, :func:`sm.graphics.plot_partregress <regressionplots.plot_partregress>`, :func:`sm.graphics.plot_ccpr   <regressionplots.plot_ccpr>`, :func:`sm.graphics.abline_plot <regressionplots.abline_plot>`, :func:`sm.graphics.influence_plot <regressionplots.influence_plot>`, and :func:`sm.graphics.plot_leverage_resid2 <regressionplots.plot_leverage_resid2>`.

.. currentmodule:: statsmodels.stats.power

Power and Sample Size Calculations
----------------------------------

The power module (``statsmodels.stats.power``) currently implements power and sample size calculations for the t-tests (:class:`sm.stats.TTestPower <TTestPower>`, :class:`sm.stats.TTestIndPower <TTestIndPower>`), normal based test (:class:`sm.stats.NormIndPower <NormIndPower>`), F-tests (:class:`sm.stats.FTestPower <FTestPower>`, `:class:sm.stats.FTestAnovaPower <FTestAnovaPower>`) and Chisquare goodness of fit (:class:`sm.stats.GofChisquarePower <GofChisquarePower>`) test. The implementation is class based, but the module also provides three shortcut functions, :func:`sm.stats.tt_solve_power <tt_solve_power>`, :func:`sm.stats.tt_ind_solve_power <tt_ind_solve_power>` and :func:`sm.stats.zt_ind_solve_power <zt_ind_solve_power>` to solve for any one of the parameters of the power equations. See this `blog post <http://jpktd.blogspot.fr/2013/03/statistical-power-in-statsmodels.html>`_ for a more in-depth description of the additions.


Other important new features
----------------------------
* **IPython notebook examples**: Many of our examples have been converted or added as IPython notebooks now. They are available `here <https://www.statsmodels.org/devel/examples/index.html#notebook-examples>`_.

* **Improved marginal effects for discrete choice models**: Expanded options for obtaining marginal effects after the estimation of nonlinear discrete choice models are available. See :py:meth:`get_margeff <statsmodels.discrete.discrete_model.DiscreteResuls.get_margeff>`.

* **OLS influence outlier measures**: After the estimation of a model with OLS, the common set of influence and outlier measures and a outlier test are now available attached as methods ``get_influnce`` and ``outlier_test`` to the Results instance. See :py:class:`OLSInfluence <statsmodels.stats.outliers_influence.OLSInfluence>` and :func:`outlier_test <statsmodels.stats.outliers_influence.outlier_test>`.

* **New datasets**: New :ref:`datasets <datasets>` are available for examples.

* **Access to R datasets**: We now have access to many of the same datasets available to R users through the `Rdatasets project <https://vincentarelbundock.github.io/Rdatasets/>`_. You can access these using the :func:`sm.datasets.get_rdataset <statsmodels.datasets.get_rdataset>` function. This function also includes caching of these datasets.

* **Improved numerical differentiation tools**: Numerical differentiation routines have been greatly improved and expanded to cover all the routines discussed in::

    Ridout, M.S. (2009) Statistical applications of the complex-step method
        of numerical differentiation. The American Statistician, 63, 66-74

  See the :ref:`sm.tools.numdiff <numdiff>` module.

* **Consistent constant handling across models**: Result statistics no longer rely on the assumption that a constant is present in the model.

* **Missing value handling across models**: Users can now control what models do in the presence of missing values via the ``missing`` keyword available in the instantiation of every model. The options are ``'none'``, ``'drop'``, and ``'raise'``. The default is ``'none'``, which does no missing value checks. To drop missing values use ``'drop'``. And ``'raise'`` will raise an error in the presence of any missing data.

.. currentmodule:: statsmodels.iolib

* **Ability to write Stata datasets**: Added the ability to write Stata ``.dta`` files. See :class:`sm.iolib.StataWriter <foreign.StataWriter>`.

.. currentmodule:: statsmodels.tsa.arima_model

* **ARIMA modeling**: statsmodels now has support for fitting Autoregressive Integrated Moving Average (ARIMA) models. See :class:`ARIMA` and :class:`ARIMAResults` for more information.

* **Support for dynamic prediction in AR(I)MA models**: It is now possible to obtain dynamic in-sample forecast values in :class:`ARMA` and :class:`ARIMA` models.

* **Improved Pandas integration**: statsmodels now supports all frequencies available in pandas for time-series modeling. These are used for intelligent dates handling for prediction. These features are available, if you pass a pandas Series or DataFrame with a DatetimeIndex to a time-series model.

.. currentmodule:: statsmodels

* **New statistical hypothesis tests**: Added statistics for calculating interrater agreement including Cohen's kappa and Fleiss' kappa (See :ref:`interrater`), statistics and hypothesis tests for proportions (See :ref:`proportion stats <proportion_stats>`), Tukey HSD (with plot) was added as an enhancement to the multiple comparison tests (:class:`sm.stats.multicomp.MultiComparison <sandbox.stats.multicomp.MultiComparison>`, :func:`sm.stats.multicomp.pairwise_tukeyhsd <stats.multicomp.pairwise_tukeyhsd>`). Weighted statistics and t tests were enhanced with new options. Tests of equivalence for one sample and two independent or paired samples were added based on t tests and z tests (See :ref:`tost`).  


Major Bugs fixed
----------------

* Post-estimation statistics for weighted least squares that depended on the centered total sum of squares were not correct. These are now correct and tested. See :issue:`501`.

* Regression through the origin models now correctly use uncentered total sum of squares in post-estimation statistics. This affected the :math:`R^2` value in linear models without a constant. See :issue:`27`.

Backwards incompatible changes and deprecations
-----------------------------------------------

* Cython code is now non-optional. You will need a C compiler to build from source. If building from github and not a source release, you will also need Cython installed. See the :ref:`installation documentation <install>`.

* The ``q_matrix`` keyword to `t_test` and `f_test` for linear models is deprecated. You can now specify linear hypotheses using formulas.

.. currentmodule:: statsmodels.tsa

* The ``conf_int`` keyword to :func:`sm.tsa.acf <stattools.acf>` is deprecated.

* The ``names`` argument is deprecated in :class:`sm.tsa.VAR <vector_ar.var_model.VAR>` and `sm.tsa.SVAR <vector_ar.svar_model.SVAR>`. This is now automatically detected and handled.

.. currentmodule:: statsmodels.tsa

* The ``order`` keyword to :py:meth:`sm.tsa.ARMA.fit <ARMA.fit>` is deprecated. It is now passed in during model instantiation.

.. currentmodule:: statsmodels.distributions

* The empirical distribution function (:class:`sm.distributions.ECDF <ECDF>`) and supporting functions have been moved to ``statsmodels.distributions``. Their old paths have been deprecated.

* The ``margeff`` method of the discrete choice models has been deprecated. Use ``get_margeff`` instead. See above. Also, the vague ``resid`` attribute of the discrete choice models has been deprecated in favor of the more descriptive ``resid_dev`` to indicate that they are deviance residuals.

.. currentmodule:: statsmodels.nonparametric.kde

* The class ``KDE`` has been deprecated and renamed to :class:`KDEUnivariate` to distinguish it from the new ``KDEMultivariate``. See above.

Development summary and credits
-------------------------------

The previous version (statsmodels 0.4.3) was released on July 2, 2012. Since then we have closed a total of 380 issues, 172 pull requests and 208 regular issues. The :ref:`detailed list<issues_list_05>` can be viewed.

This release is a result of the work of the following 38 authors who contributed total of 2032 commits. If for any reason, we have failed to list your name in the below, please contact us:

* Ana Martinez Pardo <anamartinezpardo-at-gmail.com>
* anov <novikova.go.zoom-at-gmail.com>
* avishaylivne <avishay.livne-at-gmail.com>
* Bruno Rodrigues <rodrigues.bruno-at-aquitania.org>
* Carl Vogel <carljv-at-gmail.com>
* Chad Fulton <chad-at-chadfulton.com>
* Christian Prinoth <christian-at-prinoth.name>
* Daniel B. Smith <neuromathdan-at-gmail.com>
* dengemann <denis.engemann-at-gmail.com>
* Dieter Vandenbussche <dvandenbussche-at-axioma.com>
* Dougal Sutherland <dougal-at-gmail.com>
* Enrico Giampieri <enrico.giampieri-at-unibo.it>
* evelynmitchell <efm-github-at-linsomniac.com>
* George Panterov <econgpanterov-at-gmail.com>
* Grayson <graysonbadgley-at-gmail.com>
* Jan Schulz <jasc-at-gmx.net>
* Josef Perktold <josef.pktd-at-gmail.com>
* Jeff Reback <jeff-at-reback.net>
* Justin Grana <jg3705a-at-student.american.edu>
* langmore <ianlangmore-at-gmail.com>
* Matthew Brett <matthew.brett-at-gmail.com>
* Nathaniel J. Smith <njs-at-pobox.com>
* otterb <itoi-at-live.com>
* padarn <padarn-at-wilsonp.anu.edu.au>
* Paul Hobson <pmhobson-at-gmail.com>
* Pietro Battiston <me-at-pietrobattiston.it>
* Ralf Gommers <ralf.gommers-at-googlemail.com>
* Richard T. Guy <richardtguy84-at-gmail.com>
* Robert Cimrman <cimrman3-at-ntc.zcu.cz>
* Skipper Seabold <jsseabold-at-gmail.com>
* Thomas Haslwanter <thomas.haslwanter-at-fh-linz.at>
* timmie <timmichelsen-at-gmx-topmail.de>
* Tom Augspurger <thomas-augspurger-at-uiowa.edu>
* Trent Hauck <trent.hauck-at-gmail.com>
* tylerhartley <tyleha-at-gmail.com>
* Vincent Arel-Bundock <varel-at-umich.edu>
* VirgileFritsch <virgile.fritsch-at-gmail.com>
* Zhenya <evgeni-at-burovski.me>

.. note:: 

   Obtained by running ``git log v0.4.3..HEAD --format='* %aN <%aE>' | sed 's/@/\-at\-/' | sed 's/<>//' | sort -u``.

.. _issues_list_05:

Issues closed in the 0.5.0 development cycle
============================================

Issued closed in 0.5.0
-----------------------

GitHub stats for release 0.5.0 (07/02/2012/ - 08/14/2013/).

We closed a total of 380 issues, 172 pull requests and 208 regular issues. This is the full list (generated with the script  :file:`tools/github_stats.py`):

This list is automatically generated, and may be incomplete:

Pull Requests (172):

* :pr:`1015`: DOC: Bump version. Remove done tasks.
* :pr:`1010`: DOC/RLS: Update release notes workflow. Help Needed!
* :pr:`1014`: DOC: nbgenerate does not like the comment at end of line.
* :pr:`1012`: DOC: Add link to notebook and crosslink ref. Closes #924.
* :pr:`997`: misc, tests, diagnostic
* :pr:`1009`: MAINT: Add .mailmap file.
* :pr:`817`: Add 3 new unit tests for arima_process
* :pr:`1001`: BUG include_package_data for install closes #907
* :pr:`1005`: GITHUB: Contributing guidelines
* :pr:`1007`: Cleanup docs for release
* :pr:`1003`: BUG: Workaround for bug in sphinx 1.1.3. See #1002.
* :pr:`1004`: DOC: Update maintainer notes with branching instructions.
* :pr:`1000`: BUG: Support pandas 0.8.0.
* :pr:`996`: BUG: Handle combo of pandas 0.8.0 and dateutils 1.5.0
* :pr:`995`: ENH: Print dateutil version.
* :pr:`994`: ENH: Fail gracefully for version not found.
* :pr:`993`: More conservative error catching in TimeSeriesModel
* :pr:`992`: Misc fixes 12: adjustments to unit test
* :pr:`985`: MAINT: Print versions script.
* :pr:`986`: ENH: Prefer to_offset to get_offset. Closes #964.
* :pr:`984`: COMPAT: Pandas 0.8.1 compatibility. Closes #983.
* :pr:`982`: Misc fixes 11
* :pr:`978`: TST: generic mle pareto disable bsejac tests with estimated loc
* :pr:`977`: BUG python 3.3 fix for numpy str TypeError, see #633
* :pr:`975`: Misc fixes 10 numdiff
* :pr:`970`: BUG: array too long, raises exception with newer numpy closes #967
* :pr:`965`: Vincent summary2 rebased
* :pr:`933`: Update and improve GenericlikelihoodModel and miscmodels
* :pr:`950`: BUG/REF mcnemar fix exact pvalue, allow table as input
* :pr:`951`: Pylint emplike formula genmod
* :pr:`956`: Fix a docstring in KDEMultivariateConditional.
* :pr:`949`: BUG fix lowess sort when nans closes #946
* :pr:`932`: ENH: support basinhopping solver in LikelihoodModel.fit()
* :pr:`927`: DOC: clearer minimal example
* :pr:`919`: OLS summary crash
* :pr:`918`: Fixes10 emplike lowess
* :pr:`909`: Bugs in GLM pvalues, more tests, pylint
* :pr:`906`: ENH: No fmax with Windows SDK so define inline.
* :pr:`905`: MAINT more fixes
* :pr:`898`: Misc fixes 7
* :pr:`896`: Quantreg rebase2
* :pr:`895`: Fixes issue #832
* :pr:`893`: ENH: Remove unneeded restriction on low. Closes #867.
* :pr:`894`: MAINT: Remove broken function. Keep deprecation. Closes #781.
* :pr:`856`: Carljv improved lowess rebased2
* :pr:`884`: Pyflakes cleanup
* :pr:`887`: BUG: Fix kde caching
* :pr:`883`: Fixed pyflakes issue in discrete module
* :pr:`882`: Update predstd.py
* :pr:`871`: Update of sandbox doc
* :pr:`631`: WIP: Correlation positive semi definite
* :pr:`857`: BLD: apt get dependencies from Neurodebian, whitespace cleanup
* :pr:`855`: AnaMP issue 783 mixture rvs tests rebased
* :pr:`854`: Enrico multinear rebased
* :pr:`849`: Tyler tukeyhsd rebased
* :pr:`848`: BLD TravisCI use python-dateutil package
* :pr:`784`: Misc07 cleanup multipletesting and proportions
* :pr:`841`: ENH: Add load function to main API. Closes #840.
* :pr:`820`: Ensure that tuples are not considered as data, not as data containers
* :pr:`822`: DOC: Update for Cython changes.
* :pr:`765`: Fix build issues
* :pr:`800`: Automatically generate output from notebooks
* :pr:`802`: BUG: Use two- not one-sided t-test in t_test. Closes #740.
* :pr:`806`: ENH: Import formula.api in statsmodels.api namespace.
* :pr:`803`: ENH: Fix arima error message for bad start_params
* :pr:`801`: DOC: Fix ANOVA section titles
* :pr:`795`: Negative Binomial Rebased
* :pr:`787`: Origintests
* :pr:`794`: ENH: Allow pandas-in/pandas-out in tsa.filters
* :pr:`791`: Github stats for release notes
* :pr:`779`: added np.asarray call to durbin_watson in stattools
* :pr:`772`: Anova docs
* :pr:`776`: BUG: Fix dates_from_range with length. Closes #775.
* :pr:`774`: BUG: Attach prediction start date in AR. Closes #773.
* :pr:`767`: MAINT: Remove use of deprecated from examples and docs.
* :pr:`762`: ENH: Add new residuals to wrapper
* :pr:`754`: Fix arima predict
* :pr:`760`: ENH: Adjust for k_trend in information criteria. Closes #324.
* :pr:`761`: ENH: Fixes and tests sign_test. Closes #642.
* :pr:`759`: Fix 236
* :pr:`758`: DOC: Update VAR docs. Closes #537.
* :pr:`752`: Discrete cleanup
* :pr:`750`: VAR with 1d array
* :pr:`748`: Remove reference to new_t_test and new_f_test.
* :pr:`739`: DOC: Remove outdated note in docstring
* :pr:`732`: BLD: Check for patsy dependency at build time + docs
* :pr:`731`: Handle wrapped
* :pr:`730`: Fix opt fulloutput
* :pr:`729`: Get rid of warnings in docs build
* :pr:`698`: update url for hsb2 dataset
* :pr:`727`: DOC: Fix indent and add missing params to linear models. Closes #709.
* :pr:`726`: CLN: Remove unused method. Closes #694
* :pr:`725`: BUG: Should call anova_single. Closes #702.
* :pr:`723`: Rootfinding for Power
* :pr:`722`: Handle pandas.Series with names in make_lags
* :pr:`714`: Fix 712
* :pr:`668`: Allow for any pandas frequency to be used in TimeSeriesModel.
* :pr:`711`: Misc06 - bug fixes
* :pr:`708`: BUG: Fix one regressor case for conf_int. Closes #706.
* :pr:`700`: Bugs rebased
* :pr:`680`: BUG: Swap arguments in fftconvolve for scipy >= 0.12.0
* :pr:`640`: Misc fixes 05
* :pr:`663`: a typo in runs.py doc string for mcnemar test
* :pr:`652`: WIP: fixing pyflakes / pep8, trying to improve readability
* :pr:`619`: DOC: intro to formulas
* :pr:`648`: BF: Make RLM stick to Huber's description
* :pr:`649`: Bug Fix
* :pr:`637`: Pyflakes cleanup
* :pr:`634`: VAR DOC typo
* :pr:`623`: Slowtests
* :pr:`621`: MAINT: in setup.py, only catch ImportError for pandas.
* :pr:`590`: Cleanup test output
* :pr:`591`: Interrater agreement and reliability measures
* :pr:`618`: Docs fix the main warnings and errors during sphinx build
* :pr:`610`: nonparametric examples and some fixes
* :pr:`578`: Fix 577
* :pr:`575`: MNT: Remove deprecated scikits namespace
* :pr:`499`: WIP: Handle constant
* :pr:`567`: Remove deprecated
* :pr:`571`: Dataset docs
* :pr:`561`: Grab rdatasets
* :pr:`570`: DOC: Fixed links to Rdatasets
* :pr:`524`: DOC: Clean up discrete model documentation.
* :pr:`506`: ENH: Re-use effects if model fit with QR
* :pr:`556`: WIP:  L1 doc fix
* :pr:`564`: TST: Use native integer to avoid issues in dtype asserts
* :pr:`543`: Travis CI using M.Brett nipy hack
* :pr:`558`: Plot cleanup
* :pr:`541`: Replace pandas DataMatrix with DataFrame
* :pr:`534`: Stata test fixes
* :pr:`532`: Compat 323
* :pr:`531`: DOC: Add ECDF to distributions docs
* :pr:`526`: ENH: Add class to write Stata binary dta files
* :pr:`521`: DOC: Add abline plot to docs
* :pr:`518`: Small fixes: interaction_plot
* :pr:`508`: ENH: Avoid taking cholesky decomposition of diagonal matrix
* :pr:`509`: DOC: Add ARIMA to docs
* :pr:`510`: DOC: realdpi is disposable personal income. Closes #394.
* :pr:`507`: ENH: Protect numdifftools import. Closes #45
* :pr:`504`: Fix weights
* :pr:`498`: DOC: Add patys requirement to install docs
* :pr:`491`: Make _data a public attribute.
* :pr:`494`: DOC: Fix pandas links
* :pr:`492`: added intersphinx for pandas
* :pr:`422`: Handle missing data
* :pr:`485`: ENH: Improve error message for pandas objects without dates in index
* :pr:`428`: Remove other data
* :pr:`483`: Arima predict bug
* :pr:`482`: TST: Do array-array comparison when using numpy.testing
* :pr:`471`: Formula rename df -> data
* :pr:`473`: Vincent docs tweak rebased
* :pr:`468`: Docs 050
* :pr:`462`: El aft rebased
* :pr:`461`: TST: numpy 1.5.1 compatibility
* :pr:`460`: Emplike desc reg rebase
* :pr:`410`: Discrete model marginal effects
* :pr:`417`: Numdiff cleanup
* :pr:`398`: Improved plot_corr and plot_corr_grid functions.
* :pr:`401`: BUG: Finish refactoring margeff for dummy. Closes #399.
* :pr:`400`: MAINT: remove lowess.py, which was kept in 0.4.x for backwards compatibi...
* :pr:`371`: BF+TEST: fixes, checks and tests for isestimable
* :pr:`351`: ENH: Copy diagonal before write for upcoming numpy changes
* :pr:`384`: REF: Move mixture_rvs out of sandbox.
* :pr:`368`: ENH: Add polished version of acf/pacf plots with confidence intervals
* :pr:`378`: Infer freq
* :pr:`374`: ENH: Add Fair's extramarital affair dataset. From tobit-model branch.
* :pr:`358`: ENH: Add method to OLSResults for outlier detection
* :pr:`369`: ENH: allow predict to pass through patsy for transforms
* :pr:`352`: Formula integration rebased
* :pr:`360`: REF: Deprecate order in fit and move to ARMA init
* :pr:`366`: Version fixes
* :pr:`359`: DOC: Fix sphinx warnings

Issues (208):

* :issue:`1036`: Series no longer inherits from ndarray
* :issue:`1038`: DataFrame with integer names not handled in ARIMA
* :issue:`1028`: Test fail with windows and Anaconda - Low priority
* :issue:`676`: acorr_breush_godfrey  undefined nlags
* :issue:`922`: lowess returns inconsistent with option
* :issue:`425`: no bse in robust with norm=TrimmedMean
* :issue:`1025`: add_constant incorrectly detects constant column
* :issue:`533`: py3 compatibility ``pandas.read_csv(urlopen(...))``
* :issue:`662`: doc: install instruction: explicit about removing scikits.statsmodels
* :issue:`910`: test failure Ubuntu TestARMLEConstant.test_dynamic_predict
* :issue:`80`: t_model: f_test, t_test do not work
* :issue:`432`: GenericLikelihoodModel change default for score and hessian
* :issue:`454`: BUG/ENH: HuberScale instance is not used, allow user defined scale estimator
* :issue:`98`: check connection or connect summary to variable names in wrappers
* :issue:`418`: BUG: MNLogit loglikeobs, jac
* :issue:`1017`: nosetests warnings
* :issue:`924`: DOCS link in notebooks to notebook for download
* :issue:`1011`: power ttest endless loop possible
* :issue:`907`: BLD data_files for stats.libqsturng
* :issue:`328`: consider moving example scripts into IPython notebooks
* :issue:`1002`: Docs will not build with Sphinx 1.1.3
* :issue:`69`: Make methods like compare_ftest work with wrappers
* :issue:`503`: summary_old in RegressionResults
* :issue:`991`: TST precision of normal_power
* :issue:`945`: Installing statsmodels from github?
* :issue:`964`: Prefer to_offset not get_offset in tsa stuff
* :issue:`983`: bug: pandas 0.8.1 incompatibility
* :issue:`899`: build_ext inplace does not cythonize
* :issue:`923`: location of initialization code
* :issue:`980`: auto lag selection in  S_hac_simple
* :issue:`968`: genericMLE Ubuntu test failure
* :issue:`633`: python 3.3 compatibility
* :issue:`728`: test failure for solve_power with fsolve
* :issue:`971`: numdiff test cases
* :issue:`976`: VAR Model does not work in 1D
* :issue:`972`: numdiff: epsilon has no minimum value
* :issue:`967`: lowes test failure Ubuntu
* :issue:`948`: nonparametric tests: mcnemar, cochranq unit test
* :issue:`963`: BUG in runstest_2sample
* :issue:`946`: Issue with lowess() smoother in statsmodels
* :issue:`868`: k_vars > nobs
* :issue:`917`: emplike emplikeAFT stray dimensions
* :issue:`264`: version comparisons need to be made more robust (may be just use LooseVersion)
* :issue:`674`: failure in test_foreign, pandas testing
* :issue:`828`: GLMResults inconsistent distribution in pvalues
* :issue:`908`: RLM missing test for tvalues, pvalues
* :issue:`463`: formulas missing in docs
* :issue:`256`: discrete Nbin has zero test coverage
* :issue:`831`: test errors running bdist
* :issue:`733`: Docs: interrater cohens_kappa is missing
* :issue:`897`: lowess failure - sometimes
* :issue:`902`: test failure tsa.filters  precision too high
* :issue:`901`: test failure stata_writer_pandas, newer versions of pandas
* :issue:`900`: ARIMA.__new__   errors on python 3.3
* :issue:`832`: notebook errors
* :issue:`867`: Baxter King has unneeded limit on value for low?
* :issue:`781`: discreteResults margeff method not tests, obsolete
* :issue:`870`: discrete unit tests duplicates
* :issue:`630`: problems in regression plots
* :issue:`885`: Caching behavior for KDEUnivariate icdf
* :issue:`869`: sm.tsa.ARMA(..., order=(p,q)) gives "__init__() got an unexpected keyword argument 'order'" error
* :issue:`783`: statsmodels.distributions.mixture_rvs.py    no unit tests
* :issue:`824`: Multicomparison w/Pandas Series
* :issue:`789`: presentation of multiple comparison results
* :issue:`764`: BUG: multipletests incorrect reject for Holm-Sidak
* :issue:`766`: multipletests - status and tests of 2step FDR procedures
* :issue:`763`: Bug: multipletests raises exception with empty array
* :issue:`840`: sm.load should be in the main API namespace
* :issue:`830`: invalid version number
* :issue:`821`: Fail gracefully when extensions are not built
* :issue:`204`: Cython extensions built twice?
* :issue:`689`: tutorial notebooks
* :issue:`740`: why does t_test return one-sided p-value
* :issue:`804`: What goes in statsmodels.formula.api?
* :issue:`675`: Improve error message for ARMA SVD convergence failure.
* :issue:`15`: arma singular matrix
* :issue:`559`: Add Rdatasets to optional dependencies list
* :issue:`796`: Prediction Standard Errors
* :issue:`793`: filters are not pandas aware
* :issue:`785`: Negative R-squared
* :issue:`777`: OLS residuals returned as Pandas series when endog and exog are Pandas series
* :issue:`770`: Add ANOVA to docs
* :issue:`775`: Bug in dates_from_range
* :issue:`773`: AR model pvalues error with Pandas
* :issue:`768`: multipletests: numerical problems at threshold
* :issue:`355`: add draw if interactive to plotting functions
* :issue:`625`: Exog is not correctly handled in ARIMA predict
* :issue:`626`: ARIMA summary does not print exogenous variable coefficients
* :issue:`657`: order (0,1) breaks ARMA forecast
* :issue:`736`: ARIMA predict problem for ARMA model
* :issue:`324`: ic in ARResults, aic, bic, hqic, fpe inconsistent definition?
* :issue:`642`: sign_test   check
* :issue:`236`: AR start_params broken
* :issue:`235`: tests hang on Windows
* :issue:`156`: matplotlib deprecated legend ? var plots
* :issue:`331`: Remove stale tests
* :issue:`592`: test failures in datetools
* :issue:`537`: Var Models
* :issue:`755`: Unable to access AR fit parameters when model is estimated with pandas.DataFrame
* :issue:`670`: discrete: numerically useless clipping
* :issue:`515`: MNLogit residuals raise a TypeError
* :issue:`225`: discrete models only define deviance residuals
* :issue:`594`: remove skiptest in TestProbitCG
* :issue:`681`: Dimension Error in discrete_model.py When Running test_dummy_*
* :issue:`744`: DOC: new_f_test
* :issue:`549`: Ship released patsy source in statsmodels
* :issue:`588`: patsy is a hard dependency?
* :issue:`716`: Tests missing for functions if pandas is used
* :issue:`715`: statsmodels regression plots not working with pandas datatypes
* :issue:`450`: BUG: full_output in optimizers Likelihood model
* :issue:`709`: DOCstrings linear models do not have missing params
* :issue:`370`: BUG weightstats has wrong cov
* :issue:`694`: DiscreteMargins duplicate method
* :issue:`702`: bug, pylint stats.anova
* :issue:`423`: Handling of constant across models
* :issue:`456`: BUG: ARMA date handling incompatibility with recent pandas
* :issue:`514`: NaNs in Multinomial
* :issue:`405`: Check for existing old version of scikits.statsmodels?
* :issue:`586`: Segmentation fault with OLS
* :issue:`721`: Unable to run AR on named time series objects
* :issue:`125`: caching pinv_wexog broke iterative fit - GLSAR
* :issue:`712`: TSA bug with frequency inference
* :issue:`319`: Timeseries Frequencies
* :issue:`707`: .summary with alpha ignores parsed value
* :issue:`673`: nonparametric: bug in _kernel_base
* :issue:`710`: test_power failures
* :issue:`706`: .conf_int() fails on linear regression without intercept
* :issue:`679`: Test Baxter King band-pass filter fails with scipy 0.12 beta1
* :issue:`552`: influence outliers breaks when regressing on constant
* :issue:`639`: test folders not on python path
* :issue:`565`: omni_normtest does not propagate the axis argument
* :issue:`563`: error in doc generation for AR.fit
* :issue:`109`: TestProbitCG failure on Ubuntu
* :issue:`661`: from scipy import comb fails on the latest scipy 0.11.0
* :issue:`413`: DOC: example_discrete.py missing from 0.5 documentation
* :issue:`644`: FIX: factor plot + examples broken
* :issue:`645`: STY: pep8 violations in many examples
* :issue:`173`: doc sphinx warnings
* :issue:`601`: bspline.py dependency on old scipy.stats.models
* :issue:`103`: ecdf and step function conventions
* :issue:`18`: Newey-West sandwich covariance is missing
* :issue:`279`: cov_nw_panel not tests, example broken
* :issue:`150`: precision in test_discrete.TestPoissonNewton.test_jac ?
* :issue:`480`: rescale loglike for optimization
* :issue:`627`: Travis-CI support for scipy
* :issue:`622`: mark tests as slow in emplike
* :issue:`589`: OLS F-statistic error
* :issue:`572`: statsmodels/tools/data.py Stuck looking for la.py
* :issue:`580`: test errors in graphics
* :issue:`577`: PatsyData detection buglet
* :issue:`470`: remove deprecated features
* :issue:`573`: lazy imports are (possibly) very slow
* :issue:`438`: New results instances are not in online documentation
* :issue:`542`: Regression plots fail when Series objects passed to sm.OLS
* :issue:`239`: release 0.4.x
* :issue:`530`: l1 docs issues
* :issue:`539`: test for statawriter (failure)
* :issue:`490`: Travis CI on PRs
* :issue:`252`: doc: distributions.rst refers to sandbox only
* :issue:`85`: release 0.4
* :issue:`65`: MLE fit of AR model has no tests
* :issue:`522`: ``test`` does not propagate arguments to nose
* :issue:`517`: missing array conversion or shape in linear model
* :issue:`523`: test failure with ubuntu decimals too large
* :issue:`520`: web site documentation, source not updated
* :issue:`488`: Avoid cholesky decomposition of diagonal matrices in linear regression models
* :issue:`394`: Definition in macrodata NOTE
* :issue:`45`: numdifftools dependency
* :issue:`501`: WLS/GLS post estimation results
* :issue:`500`: WLS fails if weights is a pandas.Series
* :issue:`27`: add hasconstant indicator for R-squared and df calculations
* :issue:`497`: DOC: add patsy?
* :issue:`495`: ENH: add footer SimpleTable
* :issue:`402`: model._data -> model.data?
* :issue:`477`: VAR NaN Bug
* :issue:`421`: Enhancement: Handle Missing Data
* :issue:`489`: Expose model._data as model.data
* :issue:`315`: tsa models assume pandas object indices are dates
* :issue:`440`: arima predict is broken for steps > q and q != 1
* :issue:`458`: TST BUG?   comparing pandas and array in tests, formula
* :issue:`464`: from_formula signature
* :issue:`245`: examples in docs: make nicer
* :issue:`466`: broken example, pandas
* :issue:`57`: Unhelpful error from bad exog matrix in model.py
* :issue:`271`: ARMA.geterrors requires model to be fit
* :issue:`350`: Writing to array returned np.diag
* :issue:`354`: example_rst does not copy unchanged files over
* :issue:`467`: Install issues with Pandas
* :issue:`444`: ARMA example on stable release website not working
* :issue:`377`: marginal effects count and discrete adjustments
* :issue:`426`: "svd" method not supported for OLS.fit()
* :issue:`409`: Move numdiff out of the sandbox
* :issue:`416`: Switch to complex-step Hessian for AR(I)MA
* :issue:`415`: bug in kalman_loglike_complex
* :issue:`397`: plot_corr axis text labeling not working (with fix)
* :issue:`399`: discrete errors due to incorrect in-place operation
* :issue:`389`: VAR test_normality is broken with KeyError
* :issue:`388`: Add tsaplots to graphics.api as graphics.tsa
* :issue:`387`: predict date was not getting set with start = None
* :issue:`386`: p-values not returned from acf
* :issue:`385`: Allow AR.select_order to work without model being fit
* :issue:`383`: Move mixture_rvs out of sandbox.
* :issue:`248`: ARMA breaks with a 1d exog
* :issue:`273`: When to give order for AR/AR(I)MA
* :issue:`363`: examples folder -> tutorials folder
* :issue:`346`: docs in sitepackages
* :issue:`353`: PACF docs raise a sphinx warning
* :issue:`348`: python 3.2.3 test failure zip_longest
