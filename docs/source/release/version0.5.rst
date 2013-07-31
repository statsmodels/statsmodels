===========
0.5 Release
===========

Release 0.5.0
=============

Statsmodels 0.5 contains many new features and a large amount of bug fixes.

See the :ref:`list of fixed issues <issues_list_05>` for specific closed issues.


The following major new features appear in this version.

Support for Model Formulas via Patsy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Statsmodels now supports fitting models with a formula. This functionality is provided by `patsy <http://patsy.readthedocs.org/en/latest/>`_. Patsy is now a dependency for statsmodels. Models can be individually imported from the ``statsmodels.formula.api`` namespace or you can import them all as::

    import statsmodels.formula.api as smf

Alternatively, each model in the usual ``statsmodels.api`` namespace has a ``from_formula`` classmethod that will create a model using a formula. A typical workflow can now look something like this.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    url = 'http://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv'
    data = pd.read_csv(url)

    # Fit regression model (using the natural log of one of the regressors)
    results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=data).fit()

See :ref:`here for some more documentation of using formulas in statsmodels <formula_examples>`

Empirical Likelihood (Google Summer of Code 2012 project)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
...

Analysis of Variance (ANOVA) Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
...

Nonparameteric Regression (GSoC 2012 project)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
...

Multivariate Kernel Density Estimators (GSoC 2012 project)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
...

Quantile Regression Model
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: statsmodels.regression.quantile_regression

Quantile regression is supported via the :class:`QuantReg` class. Kernel and bandwidth selection options are available for estimating the asymptotic covariance matrix using a kernel density estimator.

Negative Binomial Regression Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: statsmodels.discrete.discrete_model

It is now possible to fit negative binomial models for count data via maximum-likelihood using the :class:`NegativeBinomial` class. ``NB1``, ``NB2``, and ``geometric`` variance specifications are available.

l1-penalized discrete choice models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(optional cvxopt dependency)

Other important new features
----------------------------
* **IPython notebook examples**

* **Improved marginal effects for discrete choice models**

* **OLS outlier tests**

* **Expanded probability and diagnostic plots**

* **New datasets**

* **Access to R datasets**

* **Improved numerical differentiation tools**

* **Consistent constant handling across models**

* **Missing value handling across models**: Users can now control what models do in the presence of missing values via the ``missing`` keyword available in the instantiation of every model. The options are ``'none'``, ``'drop'``, and ``'raise'``. The default is ``'none'``, which does no missing value checks. To drop missing values use ``'drop'``. And ``'raise'`` will raise an error in the presence of any missing data.

* **Ability to write Stata datasets**: Added the ability to write Stata ``.dta`` files.

.. currentmodule:: statsmodels.tsa.arima_model

* **ARIMA modeling**: Statsmodels now has support for fitting Autoregressive Integrated Moving Average (ARIMA) models. See :class:`ARIMA` and :class:`ARIMAResults` for more information.

* **Support for dynamic prediction in AR(I)MA models**: It is now possible to obtain dynamic in-sample forecast values in :class:`ARMA` and :class:`ARIMA` models.

* **Improved Pandas integration**: Statsmodels now supports all frequencies available in pandas for time-series modeling. These are used for intelligent dates handling for prediction. These features are available, if you pass a pandas Series or DataFrame with a DatetimeIndex to a time-series model.

Major Bugs fixed
----------------

* Post-estimation statistics for weighted least squares that depended on the centered total sum of squares were not correct. These are now correct and tested. See :ghissue:`501`.

* Regression through the origin models now correctly use uncentered total sum of squares in post-estimation statistics. This affected the :math:`R^2` value in linear models without a constant. See :ghissue:`27`.

Backwards incompatible changes and deprecations
-----------------------------------------------

* Cython code is now non-optional. You will need a C compiler to build from source. If building from github and not a source release, you will also need Cython installed. See the :ref:`installation documentation <install>`.

* The ``q_matrix`` keyword to `t_test` and `f_test` for linear models is deprecated. You can now specify linear hypotheses using formulas.

.. currentmodule:: statsmodels.tsa.stattools

* The ``conf_int`` keyword to :func:`acf` is deprecated.

.. currentmodule:: statsmodels.tsa.vector_ar.var_model

* The ``names`` argument is deprecated in :class:`VAR` and SVAR. This is now automatically detected and handled.

* The ``order`` keyword to ``ARMA.fit`` is deprecated. It is now passed in during model instantiation.

.. currentmodule:: statsmodels.distributions

* The empirical distribution function (:class:`ECDF`) and supporting functions have been moved to ``statsmodels.distributions``. Their old paths have been deprecated.

* The ``margeff`` method of the discrete choice models has been deprecated. Use ``get_margeff`` instead. See above. Also, the vague ``resid`` attribute of the discrete choice models has been deprecated in favor of the more descriptive ``resid_dev`` to indicate that they are deviance residuals.

.. currentmodule:: statsmodels.nonparametric.kde

* The class ``KDE`` has been deprecated and renamed to :class:`KDEUnivariate` to distinguish it from the new ``KDEMultivariate``. See above.

Development summary and credits
-------------------------------

The previous version (statsmodels 0.4.3) was released on July 2, 2012. Since then we have closed a total of 356 issues, 166 pull requests and 190 regular issues. The :ref:`detailed list<issues_list_05>` can be viewed.

This release is a result of the work of the following 36 authors who contributed total of 1962 commits. If for any reason, we've failed to list your name in the below, please contact us:

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

.. note:: 

   Obtained by running ``git log v0.4.3..HEAD --format='* %aN <%aE>' | sed 's/@/\-at\-/' | sed 's/<>//' | sort -u``.

