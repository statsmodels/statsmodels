:orphan:

==============
Release 0.14.0
==============

Release summary
===============

statsmodels is using github to store the updated documentation. Two version are available:

- `Stable <https://www.statsmodels.org/>`_, the latest release
- `Development <https://www.statsmodels.org/devel/>`_, the latest build of the main branch

**Warning**

API stability is not guaranteed for new features, although even in
this case changes will be made in a backwards compatible way if
possible. The stability of a new feature depends on how much time it
was already in statsmodels main and how much usage it has already
seen.  If there are specific known problems or limitations, then they
are mentioned in the docstrings.

Release Statistics
------------------
**Issues Closed**: 255

**Pull Requests Merged**: 345


The Highlights
==============

New cross-sectional models and extensions to models
---------------------------------------------------

Treatment Effect
~~~~~~~~~~~~~~~~
:class:`~statsmodels.treatment.TreatmentEffect` estimates treatment effect
for a binary treatment and potential outcome for a continuous outcome variable
using 5 different methods, ipw, ra, aipw, aipw-wls, ipw-ra.
Standard errors and inference are based on the joint GMM representation of
selection or treatment model, outcome model and effect functions.

Hurdle and Truncated Count Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`statsmodels.discrete.truncated_model.HurdleCountModel` implements
hurdle models for count data with either Poisson or NegativeBinomialP as
submodels.
Three left truncated models used for zero truncation are available,
:class:`statsmodels.discrete.truncated_model.TruncatedLFPoisson`,
:class:`statsmodels.discrete.truncated_model.TruncatedLFNegativeBinomialP`
and
:class:`statsmodels.discrete.truncated_model.TruncatedLFGeneralizedPoisson`.
Models for right censoring at one are implemented but only as support for
the hurdle models.

Extended postestimation methods for models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Results methods for post-estimation have been added or extended.

``get_distribution`` returns a scipy or scipy compatible distribution instance
with parameters based on the estimated model. This is available for
GLM, discrete models and BetaModel.

``get_prediction`` returns predicted statistics including inferential
statistics, standard errors and confidence intervals. The ``which`` keyword
selects which statistic is predicted. Inference for statistics that are
nonlinear in the estimated parameters are based on the delta-method for
standard errors.

``get_diagnostic`` returns a Diagnostic class with additional specification
statistics, tests and plots. Currently only available for count models.

``get_influence`` returns a class with outlier and influence diagnostics.
(This was mostly added in previous releases.)

``score_test`` makes score (LM) test available as alternative to Wald tests.
This is currently available for GLM and some discrete models. The score tests
can optionally be robust to misspecification similar to ``cov_type`` for wald
tests.


Stats
~~~~~

Hypothesis tests, confidence intervals and other inferential statistics are
now available for one and two sample Poisson rates.

Distributions
~~~~~~~~~~~~~

Methods of Archimedean copulas have been extended to multivariate copulas with
dimension larger than 2. The ``pdf`` method of Frank and Gumbel has been
extended only to dimensions 3 and 4.

New class ECDFDiscrete for empirical distribution function when observations
are not unique as in discrete distributions.

Multiseason STL decomposition (MSTL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The existing :class:`~statsmodels.tsa.seasonal.STL` class has been extended to handle multiple seasonal
components in :class:`~statsmodels.tsa.seasonal.MSTL`. See :pr:`8160`.


Other Notable Enhancments
-------------------------

- burg option in pacf :pr:`8113`
- new link for GLM: Logc :pr:`8155`
- rename class names for links for GLM, lower case names are deprecated :pr:`8569`
- allow singular covariance in gaussian copula :pr:`8504`
- GLM: Tweedie full loglikelihood :pr:`8560`
- x13: option for location of temporary files :pr:`8564`
- Added an information set argument to ``get_prediction`` and ``predict`` methods of statespace models
  that lets the user decide on which information set to use when making forecasts.  :pr:`8002`

What's new - an overview
========================

The following lists the main new features of statsmodels 0.14.0. In addition,
release 0.14.0 includes bug fixes, refactorings and improvements in many areas.

Submodules
----------

``Documentation``
~~~~~~~~~~~~~~~~~
- Fix ZivotAndrewsUnitRoot.run() docstring  (:pr:`7812`)
- Fixes typo "Welsh ttest" to "Welch ttest"  (:pr:`7839`)
- Update maxlag to maxlags  (:pr:`7916`)
- Add prediction results to docs  (:pr:`7932`)
- Add tests for pandas compat  (:pr:`7939`)
- Fix heading level  (:pr:`7954`)
- Fix prediction docstrings  (:pr:`7970`)
- Remove DataFrame.append usage  (:pr:`7986`)
- ETS model loglike doc typo fix  (:pr:`8003`)
- Fix doc errors in MLEResults predict  (:pr:`8005`)
- Grammar  (:pr:`8023`)
- Fix missing reference  (:pr:`8038`)
- Apply small docstring corrections  (:pr:`8042`)
- Clarify difference between q_stat and acorr_ljungbox  (:pr:`8191`)
- Fix a typo in the documentation  (:pr:`8275`)
- Fix `histogram`  (:pr:`8299`)
- Add notebook for Poisson post-estimation overview  (:pr:`8420`)
- Add version  (:pr:`8863`)



``base``
~~~~~~~~
- REF/ENH  delta method and nonlinear wald test rebased  (:pr:`7758`)
- Discrete scorefactor offset rebased3  (:pr:`7825`)
- Deprecate cols in conf_int  (:pr:`7842`)
- Add start_params to TestPenalizedPoissonOraclePenalized2  (:pr:`7868`)
- ENH/REF generic get_prediction  (:pr:`7870`)
- Start move to scalar test statistics  (:pr:`7874`)
- Get_prediction for more models and cases  (:pr:`7900`)
- Scoretest betareg  (:pr:`7907`)
- Discrete add get_distribution, add which="var" for NBP, GPP  (:pr:`7929`)
- Add notebook for Poisson post-estimation overview  (:pr:`8420`)
- GenericLikelihood Results hasattr for df_resid is always true, s…  (:pr:`8476`)
- Nelder-Mead and Powell has bounds in scipy  (:pr:`8545`)



``discrete``
~~~~~~~~~~~~
- Diagnostic class rebased  (:pr:`7597`)
- Discrete scorefactor offset rebased3  (:pr:`7825`)
- Add start_params to TestPenalizedPoissonOraclePenalized2  (:pr:`7868`)
- ENH/REF generic get_prediction  (:pr:`7870`)
- Add CountResults.get_diagnostic  (:pr:`7895`)
- Get_prediction for more models and cases  (:pr:`7900`)
- Discrete add get_distribution, add which="var" for NBP, GPP  (:pr:`7929`)
- Add get_influence to DiscreteResults  (:pr:`7951`)
- Truncated, hurdle count model rebased  (:pr:`7973`)
- ENH/REF/DOC  improve hurdle and truncated count models  (:pr:`8031`)
- Add method and converged attributes to DiscreteModel.  (:pr:`8305`)
- Add notebook for Poisson post-estimation overview  (:pr:`8420`)
- Add notebook for hurdle count model  (:pr:`8424`)
- REF/DOC Poisson diagnostic  (:pr:`8502`)
- PerfectSeparation, warn by default instead of raise, GLM, discrete  (:pr:`8552`)
- Fixes, discrete perfect prediction check, Multinomial fit  (:pr:`8669`)
- MNLogit if endog is series with no name   (:pr:`8674`)
- Get_distribution, return 1-d instead of column frozen distribution  (:pr:`8780`)
- Numpy compat, indexed assignment shape in NegativeBinomial  (:pr:`8822`)
- Support offset in truncated count models  (:pr:`8845`)



``distributions``
~~~~~~~~~~~~~~~~~
- Denominator needs to be a vector  (:pr:`8086`)
- Adding weighted empirical CDF  (:pr:`8192`)
- Add parameter allow_singular for gaussian copula  (:pr:`8504`)
- Lint, pep-8 of empirical distribution, remove `__main__`  (:pr:`8546`)
- Remove extradoc from distribution, scipy deprecation  (:pr:`8598`)
- Archimedean k_dim > 2, deriv inverse in generator transform  (:pr:`8633`)
- Archimedean rvs for k_dim>2, test/gof tools  (:pr:`8642`)
- Correct tau for small theta in FrankCopula   (:pr:`8662`)



``docs``
~~~~~~~~
- Release 0.13.1 documentation  (:pr:`7881`)
- Issue #7889  (:pr:`7890`)
- Fix heading level  (:pr:`7954`)
- DEV Guide modify redundant text  (:pr:`8104`)
- Fix spelling in ARDL  (:pr:`8127`)
- Fix typos in docstring  (:pr:`8169`)
- Improve docs for using fleiss_kappa  (:pr:`8203`)
- Fix docs std_null twice instead of std_alternative  (:pr:`8228`)
- Missing `f` prefix on f-strings fix  (:pr:`8245`)
- Updated duration.rst to display output  (:pr:`8259`)
- Small doc fixes  (:pr:`8264`)
- Update book reference in ETS example  (:pr:`8282`)
- Easy PR! Fix minor typos  (:pr:`8316`)
- Added detailed ValueError to prepare_trend_spec()  (:pr:`8365`)
- Fix typo in documentation  (:pr:`8386`)
- Improvements to linear regression diagnostics example  (:pr:`8402`)
- Use pandas loc in contrasts notebook  (:pr:`8433`)
- Fix warnings  (:pr:`8483`)
- Add release note for 0.13.3  (:pr:`8485`)
- Final 0.13.3 docs  (:pr:`8493`)
- Add release notes for .4 and .5  (:pr:`8501`)
- Fix typo in gmm.py  (:pr:`8527`)
- Orthographic fix  (:pr:`8555`)
- Changes made in the documentation on endogeneity  (:pr:`8557`)
- Add to rst docs, fix docstrings  (:pr:`8559`)
- Add Statsmodels logo to Readme  (:pr:`8571`)
- Added notebook links to TSA documentation and doc strings  (:pr:`8585`)
- Fix docstring typo in rank_compare_2indep  (:pr:`8593`)
- Fix doc build  (:pr:`8608`)
- Fix indent  (:pr:`8613`)
- Remove dupe section  (:pr:`8618`)
- Fix extlinks  (:pr:`8621`)
- Various doc fixes and improvements  (:pr:`8648`)
- Fix typo in examples/notebooks/mixed_lm_example.ipynb  (:pr:`8684`)
- Fix developer page linting requirements  (:pr:`8744`)
- Add a better description of the plot generated by plot_fit  (:pr:`8760`)
- Add old release notes and draft of 0.14  (:pr:`8798`)
- Merge existing highlights  (:pr:`8799`)
- Update PRs in release note  (:pr:`8805`)
- Improve release notes highlights  (:pr:`8806`)
- Fix more deprecations and restore doc build  (:pr:`8826`)
- Final changes for 0.14.0rc0 notes  (:pr:`8839`)
- Fix internet address of dataset  (:pr:`8861`)
- Small additional fixes  (:pr:`8862`)



``gam``
~~~~~~~
- Use sorted residual to calcualte _cpr  (:pr:`7875`)



``genmod``
~~~~~~~~~~
- Genmod's loglog Formula Fixes  (:pr:`7787`)
- Allow all appropriate links in a Family  (:pr:`7816`)
- Discrete scorefactor offset rebased3  (:pr:`7825`)
- GLM score_test, use correct df_resid  (:pr:`7843`)
- ENH/REF generic get_prediction  (:pr:`7870`)
- Fix prediction docstrings  (:pr:`7970`)
- Handle lists and tuples  (:pr:`8010`)
- Adding logc link  (:pr:`8155`)
- GLM negative binomial warns if default used for parameter alpha  (:pr:`8371`)
- GLM predict which and get_prediction  (:pr:`8505`)
- Deprecate link aliases  (:pr:`8547`)
- PerfectSeparation, warn by default instead of raise, GLM, discrete  (:pr:`8552`)
- Tweedie loglike  (:pr:`8560`)
- Glm links  (:pr:`8569`)
- ENH/REF generic get_prediction  (:pr:`7870`)
- Get_prediction for more models and cases  (:pr:`7900`)
- Add start_params to TestPenalizedPoissonOraclePenalized2  (:pr:`7868`)


``graphics``
~~~~~~~~~~~~
- Correct limit in mean diff plot  (:pr:`7921`)
- Linear regression diagnosis  (:pr:`8102`)
- Fix bug #8248  (:pr:`8249`)
- Fixed minor typo on matplotlib import alias  (:pr:`8271`)
- Fix `histogram`  (:pr:`8299`)



``io``
~~~~~~
- Add _repr_latex_ methods to iolib tables  (:pr:`8134`)
- Determine if all rows have same length  (:pr:`8257`)
- Possibility of not printing r-squared in summary_col  (:pr:`8658`)
- Adding extra text in html of summary2.Summary #8663  (:pr:`8664`)



``maintenance``
~~~~~~~~~~~~~~~
- Switch to new codecov upload method  (:pr:`7799`)
- Update setup to build normally when NumPy availble  (:pr:`7801`)
- Clean up usage of private SciPy APIs as much as possible  (:pr:`7820`)
- Fix for deprecation  (:pr:`7832`)
- Protect against future pandas changes  (:pr:`7844`)
- Merge pull request #7787 from gmcmacran/loglogDoc  (:pr:`7845`)
- Merge pull request #7791 from Wooqo/fix-hw  (:pr:`7846`)
- Merge pull request #7795 from bashtage/bug-none-kpss  (:pr:`7847`)
- Merge pull request #7801 from bashtage/change-setup  (:pr:`7850`)
- Merge pull request #7812 from joaomacalos/zivot-andrews-docs  (:pr:`7852`)
- Merge pull request #7799 from bashtage/update-codecov  (:pr:`7853`)
- Merge pull request #7820 from rgommers/scipy-imports  (:pr:`7854`)
- BACKPORT Merge pull request #7844 from bashtage/future-pandas  (:pr:`7855`)
- Merge pull request #7816 from tncowart/unalias_links  (:pr:`7857`)
- Merge pull request #7832 from larsoner/dep  (:pr:`7858`)
- Merge pull request #7874 from bashtage/scalar-wald  (:pr:`7876`)
- Merge pull request #7842 from bashtage/deprecate-cols  (:pr:`7877`)
- Merge pull request #7839 from guilhermesilveira/main  (:pr:`7878`)
- Merge pull request #7868 from josef-pkt/tst_penalized_convergence  (:pr:`7879`)
- Silence warning  (:pr:`7904`)
- Remove Future and Deprecation warnings  (:pr:`7914`)
- Start removing pytest warns with None  (:pr:`7943`)
- Prevent future issues with pytest  (:pr:`7965`)
- Relax tolerance on VAR test  (:pr:`7988`)
- Modify setup requirements  (:pr:`7993`)
- Add slim to summary docstring  (:pr:`8004`)
- Add conditional models to API  (:pr:`8011`)
- Add stacklevel to warnings  (:pr:`8014`)
- Pin numpydoc  (:pr:`8041`)
- Unpin numpydoc  (:pr:`8043`)
- Add backport action  (:pr:`8052`)
- Correct upstream target  (:pr:`8074`)
- Cleanup CI  (:pr:`8083`)
- [maintenance/0.13.x] Merge pull request #7950 from bashtage/cond-number  (:pr:`8084`)
- Correct backport errors  (:pr:`8085`)
- Stop using conda temporarily  (:pr:`8088`)
- Correct small future issues  (:pr:`8089`)
- Correct setup for oldest supported  (:pr:`8092`)
- Release note for 0.13.2  (:pr:`8107`)
- Use correct setuptools backend  (:pr:`8109`)
- Update examples in python  (:pr:`8146`)
- Avoid divide by 0 in aicc  (:pr:`8176`)
- Correct linting  (:pr:`8181`)
- Use requirements  (:pr:`8210`)
- Relax overly tight tolerance  (:pr:`8215`)
- Auto bug report  (:pr:`8244`)
- Small code quality and modernizations  (:pr:`8246`)
- Further class clean  (:pr:`8247`)
- Upper bound on Cython for CI  (:pr:`8258`)
- Remove distutils  (:pr:`8266`)
- Correct clean command  (:pr:`8268`)
- Update used actions, cache pip deps, Python 3.10  (:pr:`8278`)
- Correct requirements-dev  (:pr:`8285`)
- Update lint  (:pr:`8296`)
- Remove pandas warning from pytest errors  (:pr:`8320`)
- Remove unintended print statements  (:pr:`8347`)
- Fix lint and upstream induced changes  (:pr:`8366`)
- Relax tolerance due to Scipy changes  (:pr:`8368`)
- GitHub Workflows security hardening  (:pr:`8411`)
- Fix Matplotlib deprecation of `loc` as a positional keyword in legend functions  (:pr:`8429`)
- Add a weekly scheduled run to the Azure pipelines  (:pr:`8430`)
- Add Python 3.11 jobs  (:pr:`8431`)
- Fix future warnings  (:pr:`8434`)
- Fix Windows and SciPy issues  (:pr:`8455`)
- Fix develop installs  (:pr:`8462`)
- Refactor doc build  (:pr:`8464`)
- Use stable Python 3.11 on macOS  (:pr:`8466`)
- Replave setup with setup_method in tests  (:pr:`8469`)
- Relax tolerance on tests that marginally fail  (:pr:`8470`)
- Future fixes for 0.13  (:pr:`8473`)
- Try to fix object issue  (:pr:`8474`)
- Update doc build instructions  (:pr:`8479`)
- Update doc build instructions  (:pr:`8480`)
- Backport Python 3.11 to 0.13.x branch  (:pr:`8484`)
- Set some Pins  (:pr:`8489`)
- Refine pins  (:pr:`8491`)
- Refine pins  (:pr:`8492`)
- Remove redundant wheel dep from pyproject.toml  (:pr:`8498`)
- Add Dependabot configuration for GitHub Actions updates  (:pr:`8499`)
- Bump actions/setup-python from 3 to 4  (:pr:`8500`)
- Add CodeQL workflow  (:pr:`8509`)
- Fix pre testing errors  (:pr:`8540`)
- Remove deprecated alias  (:pr:`8566`)
- Clean up deprecations  (:pr:`8588`)
- Disable failing random test, imputation, mediation  (:pr:`8597`)
- Fix style in sandbox/distributions  (:pr:`8603`)
- Fix test change due to pandas  (:pr:`8604`)
- Pin sphinx  (:pr:`8611`)
- Relax test tol for OSX fail  (:pr:`8612`)
- Update copyright date in docs/source/conf.py  (:pr:`8694`)
- MAINT/TST  unit test failures, compatibility changes  (:pr:`8777`)
- Update pyproject for 3.10  (:pr:`7880`)
- Simplify pyproject using oldest supported numpy  (:pr:`7989`)
- Update doc builder to Python 3.9  (:pr:`7997`)
- Resore doct build to 3.8  (:pr:`7999`)
- Switch to single threaded doc build  (:pr:`8012`)
- Improve specificity of warning check  (:pr:`8797`)
- Ensure statsmodels test suite passes with pandas CoW  (:pr:`8816`)
- Remove deprecated np.alltrue and np.product  (:pr:`8823`)
- Remove casts from array to scalar  (:pr:`8825`)
- Switch DeprecationWarn to FutureWarn  (:pr:`8834`)

``nonparametric``
~~~~~~~~~~~~~~~~~
- Check dtype for xvals in lowess  (:pr:`8047`)
- Correct description of `cut` parameter for `KDEUnivariate`  (:pr:`8340`)
- Improve specificity of warning check  (:pr:`8797`)
- Fix lowess Cython to handle read-only  (:pr:`8820`)


``othermod``
~~~~~~~~~~~~
- Get_prediction for more models and cases  (:pr:`7900`)
- Scoretest betareg  (:pr:`7907`)
- MLEInfluence for two-part models, extra params, BetaModel  (:pr:`7912`)


``regression``
~~~~~~~~~~~~~~
- Robust add MQuantileNorm  (:pr:`3183`)
- Update maxlag to maxlags  (:pr:`7916`)
- Ensure pinv_wexog is available  (:pr:`8161`)
- Enforce type check in recursive_olsresiduals  (:pr:`8225`)
- Faster whitening matrix calculation for sm.GLS()  (:pr:`8373`)
- Add GLS singular test  (:pr:`8375`)
- Adding extra text in html of summary2.Summary #8663  (:pr:`8664`)
- Mixedlm fit_regularized, missing vcomp in results  (:pr:`8682`)
- Correct assignment in different versions of pandas  (:pr:`8793`)



``robust``
~~~~~~~~~~
- Robust add MQuantileNorm  (:pr:`3183`)
- Fix robust.norm.Hampel  (:pr:`8801`)



``stats``
~~~~~~~~~
- REF/ENH  delta method and nonlinear wald test rebased  (:pr:`7758`)
- Update proportion.py  (:pr:`7777`)
- GLM score_test, use correct df_resid  (:pr:`7843`)
- Correct prop ci  (:pr:`7998`)
- Use scipy.stats.studentized_range in tukey hsd when available  (:pr:`8035`)
- Use nobs ratio in power and samplesize proportions_2indep  (:pr:`8093`)
- Ensure exog is well specified  (:pr:`8130`)
- Make ygrid work for etest_poisson_2indep  (:pr:`8137`)
- Allows arrays in porportions  (:pr:`8154`)
-  hypothesis tests,  confint, power for rates (poisson, negbin)  (:pr:`8166`)
- Clarify difference between q_stat and acorr_ljungbox  (:pr:`8191`)
- Fix #8227 wrong standard error of the mean   (:pr:`8260`)
- Fix critical values for hansen structural change test  (:pr:`8263`)
- ENH/DOC fixes in docs, missing in stats.api fpr rates  (:pr:`8324`)
- Fix max in tost_proportions_2indep, vectorize tost  (:pr:`8333`)
- Docs/add-missing-return-value-from-aggregate-raters-to-doc  (:pr:`8400`)
- Add notebook for stats poisson rates  (:pr:`8412`)
- Corrected the docstring of normal_sample_size_one_tail()  (:pr:`8414`)
- Notebook rankcompare  (:pr:`8427`)
- Fix docstrings  (:pr:`8494`)
- REF/DOC Poisson diagnostic  (:pr:`8502`)
- Normal_sample_size_one_tail, fix std_alt default, minimum nobs  (:pr:`8544`)
- Ref/ENH misc, smaller fixes or enhancements  (:pr:`8567`)
- Correct ContrastResults  (:pr:`8615`)
- Fix fdrcorrection_twostage, order, pvals>1  (:pr:`8623`)
- Add FTestPowerF2 as corrected version of FTestPower  (:pr:`8656`)
- Fix test_knockoff.py::test_sim failures and link  (:pr:`8673`)
- Doc fixes, bugs in proportion  (:pr:`8702`)



``topic.diagnostic``
~~~~~~~~~~~~~~~~~~~~
- Add CountResults.get_diagnostic  (:pr:`7895`)
- MLEInfluence for two-part models, extra params, BetaModel  (:pr:`7912`)
- Add get_influence to DiscreteResults  (:pr:`7951`)
- REF/DOC Poisson diagnostic  (:pr:`8502`)



``treatment``
~~~~~~~~~~~~~
- Treatment effect rebased  (:pr:`8034`)
- Add notebook for treatment effect  (:pr:`8418`)



``tsa``
~~~~~~~
- Incorrect HW predictions  (:pr:`7791`)
- Handle None in kpss  (:pr:`7795`)
- Fix ZivotAndrewsUnitRoot.run() docstring  (:pr:`7812`)
- Fox ACF/PACF docstrings  (:pr:`7927`)
- Option of initial values whe simulating VAR model  (:pr:`7930`)
- Correct STL api  (:pr:`7933`)
- Correct condition number  (:pr:`7950`)
- Correct incorrect initial trend access  (:pr:`7969`)
- ETS model loglike doc typo fix  (:pr:`8003`)
- Fix doc errors in MLEResults predict  (:pr:`8005`)
- Add apply to AutoRegResults  (:pr:`8006`)
- New census binaries have different tails  (:pr:`8007`)
- Add append method to AutoRegResults  (:pr:`8009`)
- Grammar  (:pr:`8023`)
- Bugfix for tsa/stattools.py grangercausalitytest with uncentered_tss  (:pr:`8026`)
- Improve testing of grangercausality  (:pr:`8036`)
- Add burg as an option for method to pacf  (:pr:`8113`)
- Fix ValueError output in lagmat when using pandas  (:pr:`8118`)
- Add typing support classes  (:pr:`8152`)
- Add MSTL algorithm for multi-seasonal time series decomposition  (:pr:`8160`)
- Move STL and MSTL tests to STL subpackage  (:pr:`8179`)
- Clarify difference between q_stat and acorr_ljungbox  (:pr:`8191`)
- Change heading levels in MSTL notebook to fix docs  (:pr:`8218`)
- Add MSTL docs  (:pr:`8221`)
- Remove print statement in MSTL test fixture  (:pr:`8226`)
- Switch to inexact match  (:pr:`8239`)
- Fix typo comment in tsa_model.py  (:pr:`8272`)
- Avoid removing directories from path in x13  (:pr:`8308`)
- Fix auto lag selection in acorr_ljungbox #8338  (:pr:`8339`)
- Fix when exog is Series and its name have multiple chars  (:pr:`8343`)
- ETS loglike indexing bug when y_hat == 0  (:pr:`8355`)
- Remove inhonogenous array constructor  (:pr:`8367`)
- Ensure x_columns is a list  (:pr:`8378`)
- Dickey Fuller constant values (issue #8471 )  (:pr:`8537`)
- X13.py option for location of temporary files  (:pr:`8564`)
- Ref/ENH misc, smaller fixes or enhancements  (:pr:`8567`)
- AR/MA creation with ArmaProcess.from_roots  (:pr:`8742`)
- Statespace: issue FutureWarning for unknown keyword args  (:pr:`8810`)
- Correct initial level, treand and seasonal  (:pr:`8831`)
- Add sharex for seasonal decompose plots  (:pr:`8835`)



``tsa.statespace``
~~~~~~~~~~~~~~~~~~
- Correct seasonal order  (:pr:`7906`)
- Add prediction results to docs  (:pr:`7932`)
- Fix heuristic and simple initial seasonals in state space ExponentialSmoothing  (:pr:`7991`)
- Remove aliasing of type punned pointers  (:pr:`7995`)
- Prevent signed and unsigned int comparison  (:pr:`8000`)
- Add information set selection (predicted, filtered, smoothed) and "signal" prediction to state space predict  (:pr:`8002`)
- Function to compute smoothed state weights (observations and prior mean) for state space models  (:pr:`8013`)
- Improve some state space docstrings.  (:pr:`8015`)
- State space: add revisions to news, decomposition of smoothed states/signals  (:pr:`8028`)
- State space: improve weights performance  (:pr:`8030`)
- Fix a typo in the documentation  (:pr:`8275`)
- SARIMAX variance starting parameter when the MA order is large relative to sample size  (:pr:`8297`)
- Fix sim smoother nan, dims / add options  (:pr:`8354`)
- Loop instead of if in SARIMAX transition init  (:pr:`8743`)
- Statespace: issue FutureWarning for unknown keyword args  (:pr:`8810`)



``tsa.vector.ar``
~~~~~~~~~~~~~~~~~
- Option of initial values whe simulating VAR model  (:pr:`7930`)
- Number of simulations on simualte var  (:pr:`7958`)






bug-wrong
---------

A new issue label `type-bug-wrong` indicates bugs that cause that incorrect
numbers are returned without warnings.
(Regular bugs are mostly usability bugs or bugs that raise an exception for
unsupported use cases.)
`see tagged issues <https://github.com/statsmodels/statsmodels/issues?q=is%3Aissue+label%3Atype-bug-wrong+is%3Aclosed+milestone%3A0.14>`_


Major Bugs Fixed
================

See github issues for a list of bug fixes included in this release

- `Closed bugs <https://github.com/statsmodels/statsmodels/pulls?utf8=%E2%9C%93&q=is%3Apr+is%3Amerged+milestone%3A0.14+label%3Atype-bug>`_
- `Closed bugs (wrong result) <https://github.com/statsmodels/statsmodels/pulls?q=is%3Apr+is%3Amerged+milestone%3A0.14+label%3Atype-bug-wrong>`_


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

Thanks to all of the contributors for the 0.14.0 release (based on git log):

- Adam Murphy
- Alex
- Alex Blackwell
- Alex Thompson
- AmarAdilovic
- Amit Anchalia
- Anthony Lee
- Bill
- Chad Fulton
- Christian Lorentzen
- Daedalos
- EC-AI
- Eitan Hemed
- Elliot A Martin
- Eric Larson
- Eva Maxfield Brown
- Evgeny Zhurko
- Ewout Ter Hoeven
- Geoffrey Oxberry
- Greg Mcmahan
- Gregory Parkes
- Guilherme Silveira
- Henry Schreiner
- Ishan Chokshi
- James Fiedler
- Jan-Frederik Konopka
- Jere Lahelma
- Joao Pedro
- Josef Perktold
- João Tanaka
- Kees Mulder
- Kerby Shedden
- Kevin Sheppard
- Kirill Milash
- Kirill Ulanov
- Kishan Manani
- Lindsay Stevens
- Malte Londschien
- Matt Spinelli
- Max Foxley-Marrable
- Michael Chirico
- Michał Górny
- Neil Zhao
- Nicholas Shea
- Nicky Sandhu
- Nikita Kostiuchenko
- Pavlo Fesenko
- Peter Stöckli
- Pierre Haessig
- Prajwal Kafle
- Ralf Gommers
- Ramon Viñas
- Rebecca N. Palmer
- Ryan Russell
- Samuel Wallan
- Stefan Vodita
- Thomas Cowart
- Tobias Gebhard
- Toshiaki Asakura
- Wainberg
- Winfield Chen
- Yiming Paul Li
- Zach Probst 
- Zachariah
- code-review-doctor
- dependabot[bot]
- enricovara
- j-svensmark
- kuritzen
- lanzariel
- mildc055ee
- oronimbus
- partev
- rambam613
- vasudeva-ram
- wisp3rwind
- zhengkai2001


These lists of names are automatically generated based on git log, and may not
be complete.

Merged Pull Requests
--------------------

The following Pull Requests were merged since the last release:

- :pr:`3183`: ENH: robust add MQuantileNorm
- :pr:`7597`: ENH: Diagnostic class rebased
- :pr:`7758`: REF/ENH  delta method and nonlinear wald test rebased
- :pr:`7777`: Update proportion.py
- :pr:`7787`: DOC: Genmod's loglog Formula Fixes
- :pr:`7791`: BUG: incorrect HW predictions
- :pr:`7795`: BUG: Handle None in kpss
- :pr:`7799`: MAINT: Switch to new codecov upload method
- :pr:`7801`: MAINT: Update setup to build normally when NumPy availble
- :pr:`7812`: DOC: fix ZivotAndrewsUnitRoot.run() docstring
- :pr:`7816`: BUG: Allow all appropriate links in a Family
- :pr:`7820`: MAINT: clean up usage of private SciPy APIs as much as possible
- :pr:`7825`: Discrete scorefactor offset rebased3
- :pr:`7832`: FIX: Fix for deprecation
- :pr:`7839`: DOC: Fixes typo "Welsh ttest" to "Welch ttest"
- :pr:`7842`: MAINT: Deprecate cols in conf_int
- :pr:`7843`: BUG: GLM score_test, use correct df_resid
- :pr:`7844`: MAINT: Protect against future pandas changes
- :pr:`7845`: BACKPORT: Merge pull request #7787 from gmcmacran/loglogDoc
- :pr:`7846`: BACKPORT: Merge pull request #7791 from Wooqo/fix-hw
- :pr:`7847`: BACKPORT: Merge pull request #7795 from bashtage/bug-none-kpss
- :pr:`7850`: BACKPORT: Merge pull request #7801 from bashtage/change-setup
- :pr:`7852`: BACKPORT: Merge pull request #7812 from joaomacalos/zivot-andrews-docs
- :pr:`7853`: BACKPORT: Merge pull request #7799 from bashtage/update-codecov
- :pr:`7854`: BACKPORT: Merge pull request #7820 from rgommers/scipy-imports
- :pr:`7855`: BACKPORT Merge pull request #7844 from bashtage/future-pandas
- :pr:`7857`: BACKPORT: Merge pull request #7816 from tncowart/unalias_links
- :pr:`7858`: BACKPORT: Merge pull request #7832 from larsoner/dep
- :pr:`7868`: TST: add start_params to TestPenalizedPoissonOraclePenalized2
- :pr:`7870`: ENH/REF generic get_prediction
- :pr:`7874`: ENH: Start move to scalar test statistics
- :pr:`7875`: BUG: Use sorted residual to calcualte _cpr
- :pr:`7876`: BACKPORT: Merge pull request #7874 from bashtage/scalar-wald
- :pr:`7877`: BACKPORT: Merge pull request #7842 from bashtage/deprecate-cols
- :pr:`7878`: BACKPORT: Merge pull request #7839 from guilhermesilveira/main
- :pr:`7879`: BACKPORT: Merge pull request #7868 from josef-pkt/tst_penalized_convergence
- :pr:`7880`: MAINT: Update pyproject for 3.10
- :pr:`7881`: RLS: Release 0.13.1 documentation
- :pr:`7890`: DOC: Issue #7889
- :pr:`7895`: REF/ENH: add CountResults.get_diagnostic
- :pr:`7900`: ENH/BUG: get_prediction for more models and cases
- :pr:`7904`: MAINT: Silence warning
- :pr:`7906`: BUG: Correct seasonal order
- :pr:`7907`: ENH/REF: Scoretest betareg
- :pr:`7912`: ENH: MLEInfluence for two-part models, extra params, BetaModel
- :pr:`7914`: MAINT: Remove Future and Deprecation warnings
- :pr:`7916`: DOC: update maxlag to maxlags
- :pr:`7921`: BUG: Correct limit in mean diff plot
- :pr:`7927`: DOC: Fox ACF/PACF docstrings
- :pr:`7929`: ENH/REF: discrete add get_distribution, add which="var" for NBP, GPP
- :pr:`7930`: ENH: Option of initial values whe simulating VAR model
- :pr:`7932`: DOC: Add prediction results to docs
- :pr:`7933`: DOC: Correct STL api
- :pr:`7939`: TST: Add tests for pandas compat
- :pr:`7940`: MAINT: Future NumPy compat
- :pr:`7943`: MAINT: Start removing pytest warns with None
- :pr:`7950`: BUG: Correct condition number
- :pr:`7951`: ENH: add get_influence to DiscreteResults
- :pr:`7954`: DOC: Fix heading level
- :pr:`7958`: ENH: Number of simulations on simualte var
- :pr:`7965`: MAINT: Prevent future issues with pytest
- :pr:`7969`: BUG: Correct incorrect initial trend access
- :pr:`7970`: DOC: Fix prediction docstrings
- :pr:`7973`: ENH: Truncated, hurdle count model rebased
- :pr:`7986`: MAINT: Remove DataFrame.append usage
- :pr:`7988`: MAINT: Relax tolerance on VAR test
- :pr:`7989`: MAINT: Simplify pyproject using oldest supported numpy
- :pr:`7991`: BUG/DOC: Fix heuristic and simple initial seasonals in state space ExponentialSmoothing
- :pr:`7993`: MAINT: Modify setup requirements
- :pr:`7995`: MAINT: Remove aliasing of type punned pointers
- :pr:`7996`: MAINT: Fix issues in future pandas
- :pr:`7997`: MAINT: Update doc builder to Python 3.9
- :pr:`7998`: BUG: Correct prop ci
- :pr:`7999`: MAINT: Resore doct build to 3.8
- :pr:`8000`: CLN: Prevent signed and unsigned int comparison
- :pr:`8001`: MAINT: Update binom_test to binomtest
- :pr:`8002`: ENH: Add information set selection (predicted, filtered, smoothed) and "signal" prediction to state space predict
- :pr:`8003`: DOC: ETS model loglike doc typo fix
- :pr:`8004`: MAINT: Add slim to summary docstring
- :pr:`8005`: DOC: Fix doc errors in MLEResults predict
- :pr:`8006`: ENH: Add apply to AutoRegResults
- :pr:`8007`: new census binaries have different tails
- :pr:`8009`: ENH: Add append method to AutoRegResults
- :pr:`8010`: GEE inputs: handle lists and tuples
- :pr:`8011`: MAINT: Add conditional models to API
- :pr:`8012`: MAINT: Switch to single threaded doc build
- :pr:`8013`: ENH: function to compute smoothed state weights (observations and prior mean) for state space models
- :pr:`8014`: MAINT: Add stacklevel to warnings
- :pr:`8015`: DOC: improve some state space docstrings.
- :pr:`8023`: Grammar
- :pr:`8026`: bugfix for tsa/stattools.py grangercausalitytest with uncentered_tss
- :pr:`8028`: ENH: state space: add revisions to news, decomposition of smoothed states/signals
- :pr:`8030`: PERF: state space: improve weights performance
- :pr:`8031`: ENH/REF/DOC  improve hurdle and truncated count models
- :pr:`8034`: ENH: Treatment effect rebased
- :pr:`8035`: ENH: use scipy.stats.studentized_range in tukey hsd when available
- :pr:`8036`: MAINT: Improve testing of grangercausality
- :pr:`8037`: MAINT: Protect against future pandas changes
- :pr:`8038`: DOC: Fix missing reference
- :pr:`8041`: MAINT: Pin numpydoc
- :pr:`8042`: DOC: Apply small docstring corrections
- :pr:`8043`: MAINT: Unpin numpydoc
- :pr:`8047`: BUG: Check dtype for xvals in lowess
- :pr:`8052`: MAINT: Add backport action
- :pr:`8053`: [maintenance/0.13.x] Merge pull request #8035 from swallan/scipy-studentized-range-qcrit-pvalue
- :pr:`8054`: [maintenance/0.13.x] Merge pull request #7989 from bashtage/try-oldest-supported-numpy
- :pr:`8055`: [maintenance/0.13.x] Merge pull request #7906 from bashtage/reverse-seasonal
- :pr:`8056`: [maintenance/0.13.x] Merge pull request #7921 from bashtage/mean-diff-plot
- :pr:`8057`: [maintenance/0.13.x] Merge pull request #7927 from bashtage/enricovara-patch-1
- :pr:`8058`: [maintenance/0.13.x] Merge pull request #7939 from bashtage/test-pandas-compat
- :pr:`8059`: [maintenance/0.13.x] Merge pull request #7954 from bashtage/recursive-ls-heading
- :pr:`8060`: [maintenance/0.13.x] Merge pull request #7969 from bashtage/hw-wrong-param
- :pr:`8061`: [maintenance/0.13.x] Merge pull request #7988 from bashtage/relax-tol-var-test
- :pr:`8062`: [maintenance/0.13.x] Merge pull request #7991 from ChadFulton/ss-exp-smth-seasonals
- :pr:`8063`: [maintenance/0.13.x] Merge pull request #7995 from bashtage/remove-aliasing
- :pr:`8064`: [maintenance/0.13.x] Merge pull request #8000 from bashtage/unsigned-int-comparrison
- :pr:`8065`: [maintenance/0.13.x] Merge pull request #8003 from pkaf/ets-loglike-doc
- :pr:`8066`: [maintenance/0.13.x] Merge pull request #8007 from rambam613/patch-1
- :pr:`8068`: [maintenance/0.13.x] Merge pull request #8015 from ChadFulton/ss-docs
- :pr:`8069`: [maintenance/0.13.x] Merge pull request #8023 from MichaelChirico/patch-1
- :pr:`8070`: [maintenance/0.13.x] Merge pull request #8026 from wirkuttis/bugfix_statstools
- :pr:`8072`: [maintenance/0.13.x] Merge pull request #8042 from bashtage/pin-numpydoc
- :pr:`8073`: [maintenance/0.13.x] Merge pull request #8047 from bashtage/fix-lowess-8046
- :pr:`8074`: MAINT: Correct upstream target
- :pr:`8075`: [maintenance/0.13.x] Merge pull request #7916 from zprobs/main
- :pr:`8077`: [maintenance/0.13.x] Merge pull request #8037 from bashtage/future-pandas
- :pr:`8078`: [maintenance/0.13.x] Merge pull request #8005 from bashtage/mle-results-doc
- :pr:`8079`: [maintenance/0.13.x] Merge pull request #8004 from bashtage/doc-slim
- :pr:`8080`: [maintenance/0.13.x] Merge pull request #7875 from ZachariahPang/Fix-wrong-order-datapoints
- :pr:`8081`: [maintenance/0.13.x] Merge pull request #7940 from bashtage/future-co…
- :pr:`8082`: [maintenance/0.13.x] Merge pull request #7946 from bashtage/remove-looseversion
- :pr:`8083`: MAINT: Cleanup CI
- :pr:`8084`: [maintenance/0.13.x] Merge pull request #7950 from bashtage/cond-number
- :pr:`8085`: MAINT: Correct backport errors
- :pr:`8086`: BUG: denominator needs to be a vector
- :pr:`8088`: MAINT: Stop using conda temporarily
- :pr:`8089`: MAINT: Correct small future issues
- :pr:`8092`: MAINT: Correct setup for oldest supported
- :pr:`8093`: BUG: use nobs ratio in power and samplesize proportions_2indep
- :pr:`8096`: [maintenance/0.13.x] Merge pull request #8093 from josef-pkt/bug_proportion_pwer_2indep
- :pr:`8097`: [maintenance/0.13.x] Merge pull request #8086 from xjcl/patch-1
- :pr:`8102`: DOC: Linear regression diagnosis
- :pr:`8104`: DOC: DEV Guide modify redundant text
- :pr:`8107`: MAINT: Release note for 0.13.2
- :pr:`8109`: fix(setup): use correct setuptools backend
- :pr:`8111`: [maintenance/0.13.x] Merge pull request #8109 from henryiii/patch-2
- :pr:`8113`: ENH: add burg as an option for method to pacf
- :pr:`8118`: BUG: Fix ValueError output in lagmat when using pandas
- :pr:`8127`: DOC: Fix spelling in ARDL
- :pr:`8130`: BUG: Ensure exog is well specified
- :pr:`8134`: ENH: Add _repr_latex_ methods to iolib tables
- :pr:`8137`: BUG: Make ygrid work for etest_poisson_2indep
- :pr:`8146`: MAINT: Update examples in python
- :pr:`8152`: TYP: Add typing support classes
- :pr:`8154`: BUG: Allows arrays in porportions
- :pr:`8155`: ENH: Adding logc link
- :pr:`8160`: ENH: Add MSTL algorithm for multi-seasonal time series decomposition
- :pr:`8161`: BUG: Ensure pinv_wexog is available
- :pr:`8166`: ENH:  hypothesis tests,  confint, power for rates (poisson, negbin)
- :pr:`8169`: DOC: Fix typos in docstring
- :pr:`8176`: BUG: Avoid divide by 0 in aicc
- :pr:`8179`: REF: Move STL and MSTL tests to STL subpackage
- :pr:`8181`: MAINT: Correct linting
- :pr:`8191`: DOC: Clarify difference between q_stat and acorr_ljungbox
- :pr:`8192`: adding weighted empirical CDF
- :pr:`8203`: DOC: Improve docs for using fleiss_kappa
- :pr:`8210`: MAINT: Use requirements
- :pr:`8215`: MAINT: Relax overly tight tolerance
- :pr:`8218`: DOC: Change heading levels in MSTL notebook to fix docs
- :pr:`8221`: DOC: Add MSTL docs
- :pr:`8225`: BUG: Enforce type check in recursive_olsresiduals
- :pr:`8226`: TST: Remove print statement in MSTL test fixture
- :pr:`8228`: Fix docs std_null twice instead of std_alternative
- :pr:`8239`: BUG: Switch to inexact match
- :pr:`8244`: Auto bug report
- :pr:`8245`: Missing `f` prefix on f-strings fix
- :pr:`8246`: MAINT: Small code quality and modernizations
- :pr:`8247`: MAINT: Further class clean
- :pr:`8249`: Fix bug #8248
- :pr:`8257`: BUG: determine if all rows have same length
- :pr:`8258`: MAINT: Upper bound on Cython for CI
- :pr:`8259`: DOC: Updated duration.rst to display output
- :pr:`8260`:  BUG: fix #8227 wrong standard error of the mean 
- :pr:`8263`: BUG: fix critical values for hansen structural change test
- :pr:`8264`: DOC: Small doc fixes
- :pr:`8266`: MAINT: Remove distutils
- :pr:`8268`: BUG: Correct clean command
- :pr:`8271`: Fixed minor typo on matplotlib import alias
- :pr:`8272`: MAINT: fix typo comment in tsa_model.py
- :pr:`8275`: DOC: fix a typo in the documentation
- :pr:`8278`: CI: Update used actions, cache pip deps, Python 3.10
- :pr:`8282`: Update book reference in ETS example
- :pr:`8285`: MAINT: Correct requirements-dev
- :pr:`8296`: MAINT: Update lint
- :pr:`8297`: BUG: SARIMAX variance starting parameter when the MA order is large relative to sample size
- :pr:`8299`: DOC: fix `histogram`
- :pr:`8305`: ENH: Add method and converged attributes to DiscreteModel.
- :pr:`8308`: BUG: Avoid removing directories from path in x13
- :pr:`8316`: Easy PR! Fix minor typos
- :pr:`8320`: MAINT: Remove pandas warning from pytest errors
- :pr:`8324`: ENH/DOC fixes in docs, missing in stats.api fpr rates
- :pr:`8333`: BUG/ENH: fix max in tost_proportions_2indep, vectorize tost
- :pr:`8335`: Update data.py
- :pr:`8339`: BUG: Fix auto lag selection in acorr_ljungbox #8338
- :pr:`8340`: DOC: Correct description of `cut` parameter for `KDEUnivariate`
- :pr:`8343`: BUG: Fix when exog is Series and its name have multiple chars
- :pr:`8347`: MAINT: Remove unintended print statements
- :pr:`8354`: BUG/ENH: Fix sim smoother nan, dims / add options
- :pr:`8355`: BUG: ETS loglike indexing bug when y_hat == 0
- :pr:`8365`: DOC: added detailed ValueError to prepare_trend_spec()
- :pr:`8366`: MAINT: Fix lint and upstream induced changes
- :pr:`8367`: MAINT: Remove inhonogenous array constructor
- :pr:`8368`: MAINT: Relax tolerance due to Scipy changes
- :pr:`8371`: GLM negative binomial warns if default used for parameter alpha
- :pr:`8373`: ENH: faster whitening matrix calculation for sm.GLS()
- :pr:`8375`: TST: Add GLS singular test
- :pr:`8378`: BUG: Ensure x_columns is a list
- :pr:`8386`: Fix typo in documentation
- :pr:`8400`: docs/add-missing-return-value-from-aggregate-raters-to-doc
- :pr:`8402`: DOC: Improvements to linear regression diagnostics example
- :pr:`8411`: GitHub Workflows security hardening
- :pr:`8412`: DOC: add notebook for stats poisson rates
- :pr:`8414`: Corrected the docstring of normal_sample_size_one_tail()
- :pr:`8418`: DOC: add notebook for treatment effect
- :pr:`8420`: DOC: add notebook for Poisson post-estimation overview
- :pr:`8424`: DOC: add notebook for hurdle count model
- :pr:`8427`: DOC: Notebook rankcompare
- :pr:`8429`: Fix Matplotlib deprecation of `loc` as a positional keyword in legend functions
- :pr:`8430`: CI: Add a weekly scheduled run to the Azure pipelines
- :pr:`8431`: CI: Add Python 3.11 jobs
- :pr:`8433`: Maint: use pandas loc in contrasts notebook
- :pr:`8434`: MAINT: Fix future warnings
- :pr:`8455`: MAINT: Fix Windows and SciPy issues
- :pr:`8462`: MAINT: fix develop installs
- :pr:`8464`: MAINT: Refactor doc build
- :pr:`8466`: CI: Use stable Python 3.11 on macOS
- :pr:`8469`: MAINT: Replave setup with setup_method in tests
- :pr:`8470`: TST: Relax tolerance on tests that marginally fail
- :pr:`8473`: MAINT: Future fixes for 0.13
- :pr:`8474`: MAINT: Try to fix object issue
- :pr:`8476`: BUG: GenericLikelihood Results hasattr for df_resid is always true, s…
- :pr:`8479`: MAINT: Update doc build instructions
- :pr:`8480`: MAINT: Update doc build instructions
- :pr:`8483`: DOC: Fix warnings
- :pr:`8484`: MAINT: Backport Python 3.11 to 0.13.x branch
- :pr:`8485`: DOC: Add release note for 0.13.3
- :pr:`8489`: MAINT: Set some Pins
- :pr:`8491`: MAINT: Refine pins
- :pr:`8492`: MAINT: Refine pins
- :pr:`8493`: DOC: Final 0.13.3 docs
- :pr:`8494`: DOC: fix docstrings
- :pr:`8498`: BLD: Remove redundant wheel dep from pyproject.toml
- :pr:`8499`: Add Dependabot configuration for GitHub Actions updates
- :pr:`8500`: Bump actions/setup-python from 3 to 4
- :pr:`8501`: DOC: Add release notes for .4 and .5
- :pr:`8502`: REF/DOC Poisson diagnostic
- :pr:`8504`: add parameter allow_singular for gaussian copula
- :pr:`8505`: REF/ENH: GLM predict which and get_prediction
- :pr:`8509`: Add CodeQL workflow
- :pr:`8521`: fix typo in fit_regularized
- :pr:`8527`: DOC: Fix typo in gmm.py
- :pr:`8537`: BUG: Dickey Fuller constant values (issue #8471 )
- :pr:`8540`: MAINT: fix pre testing errors
- :pr:`8544`: BUG: normal_sample_size_one_tail, fix std_alt default, minimum nobs
- :pr:`8545`: ENH: Nelder-Mead and Powell has bounds in scipy
- :pr:`8546`: STY: lint, pep-8 of empirical distribution, remove `__main__`
- :pr:`8547`: MAINT: Deprecate link aliases
- :pr:`8552`: REF: PerfectSeparation, warn by default instead of raise, GLM, discrete
- :pr:`8555`: Orthographic fix
- :pr:`8557`: Changes made in the documentation on endogeneity
- :pr:`8559`: DOC: add to rst docs, fix docstrings
- :pr:`8560`: ENH: Tweedie loglike
- :pr:`8564`: ENH: x13.py option for location of temporary files
- :pr:`8566`: MAINT: Remove deprecated alias
- :pr:`8567`: Ref/ENH misc, smaller fixes or enhancements
- :pr:`8569`: REF/TST: glm links
- :pr:`8571`: Add Statsmodels logo to Readme
- :pr:`8585`: DOC: added notebook links to TSA documentation and doc strings
- :pr:`8588`: MAINT: Clean up deprecations
- :pr:`8593`: DOC: fix docstring typo in rank_compare_2indep
- :pr:`8597`: TST: disable failing random test, imputation, mediation
- :pr:`8598`: MAINT/REF: remove extradoc from distribution, scipy deprecation
- :pr:`8603`: MAINT: Fix style in sandbox/distributions
- :pr:`8604`: MAINT/TST: Fix test change due to pandas
- :pr:`8608`: DOC: Fix doc build
- :pr:`8611`: MAINT: Pin sphinx
- :pr:`8612`: MAINT/TST: Relax test tol for OSX fail
- :pr:`8613`: DOC: Fix indent
- :pr:`8615`: BUG: Correct ContrastResults
- :pr:`8618`: DOC: Remove dupe section
- :pr:`8621`: DOC: Fix extlinks
- :pr:`8623`: BUG: fix fdrcorrection_twostage, order, pvals>1
- :pr:`8633`: ENH/BUG: archimedean k_dim > 2, deriv inverse in generator transform
- :pr:`8642`: ENH/TST: archimedean rvs for k_dim>2, test/gof tools
- :pr:`8648`: DOC: various doc fixes and improvements
- :pr:`8656`: ENH/BUG: add FTestPowerF2 as corrected version of FTestPower
- :pr:`8658`: ENH/TST: Possibility of not printing r-squared in summary_col
- :pr:`8662`: BUG/ENH: correct tau for small theta in FrankCopula 
- :pr:`8664`: BUG: Adding extra text in html of summary2.Summary #8663
- :pr:`8669`: BUG: fixes, discrete perfect prediction check, Multinomial fit
- :pr:`8673`: Fix test_knockoff.py::test_sim failures and link
- :pr:`8674`: BUG: MNLogit if endog is series with no name 
- :pr:`8682`: BUG: mixedlm fit_regularized, missing vcomp in results
- :pr:`8684`: DOC: Fix typo in examples/notebooks/mixed_lm_example.ipynb
- :pr:`8693`: TST: readd deleted test_package.py 
- :pr:`8694`: Update copyright date in docs/source/conf.py
- :pr:`8702`: BUG/DOC: doc fixes, bugs in proportion
- :pr:`8735`: BUG: a few more small bug fixes
- :pr:`8742`: ENH/TST: AR/MA creation with ArmaProcess.from_roots
- :pr:`8743`: BUG: loop instead of if in SARIMAX transition init
- :pr:`8744`: DOC: fix developer page linting requirements
- :pr:`8760`: DOC: Add a better description of the plot generated by plot_fit
- :pr:`8777`: MAINT/TST  unit test failures, compatibility changes
- :pr:`8780`: REF: get_distribution, return 1-d instead of column frozen distribution
- :pr:`8793`: BUG: Correct assignment in different versions of pandas
- :pr:`8797`: MAINT: Improve specificity of warning check
- :pr:`8798`: DOC: Add old release notes and draft of 0.14
- :pr:`8799`: DOC: Merge existing highlights
- :pr:`8801`: BUG: fix robust.norm.Hampel
- :pr:`8805`: DOC: Update PRs in release note
- :pr:`8806`: DOC: improve release notes highlights
- :pr:`8810`: ENH/BUG: statespace: issue FutureWarning for unknown keyword args
- :pr:`8816`: MAINT: Ensure statsmodels test suite passes with pandas CoW
- :pr:`8819`: MAINT: Cap sphinx in the doc build
- :pr:`8820`: MAINT: Fix lowess Cython to handle read-only
- :pr:`8822`: MAINT: numpy compat, indexed assignment shape in NegativeBinomial
- :pr:`8823`: Maint: remove deprecated np.alltrue and np.product
- :pr:`8825`: MAINT: Remove casts from array to scalar
- :pr:`8826`: MAINT/DOC: Fix more deprecations and restore doc build
- :pr:`8828`: Theta method bug
- :pr:`8829`: DOC: Correct docstring for diff
- :pr:`8830`: MAINT: Monkey deprecated patsy function
- :pr:`8831`: BUG: Correct initial level, treand and seasonal
- :pr:`8834`: MAINT: Switch DeprecationWarn to FutureWarn
- :pr:`8835`: ENH: Add sharex for seasonal decompose plots
- :pr:`8839`: DOC: Final changes for 0.14.0rc0 notes
- :pr:`8845`: BUG/ENH: support offset in truncated count models
- :pr:`8847`: DOC: Use JSON for versioning
- :pr:`8851`: BUG: Fix added variable plots to work with OLS
- :pr:`8857`: DOC: Small fix for STLForecast example
- :pr:`8858`: DOC: Fix example notebooks
- :pr:`8861`: DOC: Fix internet address of dataset
- :pr:`8862`: DOC: Small additional fixes
- :pr:`8863`: DOC: Add version
- :pr:`8865`: MAINT: Move from Styler.applymap to map
- :pr:`8866`: DOC: Add admonitions for changes and deprecations